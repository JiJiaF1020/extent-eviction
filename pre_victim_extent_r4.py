"""GLM-5.1 A5/C8 物理 slot extent 粒度 Pre-Victim（R4: 块内 max 衰减计数）。

直接替换 ``pre_victim_lru.py`` 使用：类名、构造参数、模块常量、全部公开方法与
V8 完全一致，``pre_victim_runtime.py`` 等调用方零改动。启用方式二选一：
1. 把本文件内容覆盖 ``vllm_ascend/dsa_offload/pre_victim_lru.py``；
2. 或把本文件改名为 ``pre_victim_lru.py`` 放到同目录。

与 V8 的差异（淘汰语义，对应 chunk-evict-replay 的 r4 规则）：
- 淘汰单位从单个 slot 变为 ``extent_granularity`` 个物理连续 slot 的对齐 extent
  （默认 8）。每轮 victim 为 ``victim_count // extent_granularity`` 个完整 extent，
  extent 内 slot 按 ID 升序写出，CPU SGL 可以按连续块直写。
- extent 分数 = 块内各 slot 衰减计数的最大值（每次访问先按
  2^(-Δt/EXTENT_DECAY_HALFLIFE) 衰减到时基再 +1，半衰期 64 轮）——
  块内最强的一个热 token 就能保住整个 extent（钉住效应），
  分数低的 extent 先被淘汰。replay 实测 16k 容量下该规则最优。
- 同分时块内 max(last_access) 更旧的 extent 优先被淘汰（与 replay 一致）。
- 刚被覆写的 extent 有 ``EXTENT_GRACE_ROUNDS`` 轮免淘汰保护（与 replay 的
  GRACE=2 一致：写入轮 t 之后，t+1 轮不可被选为 victim）。
- g=1 时退化为逐 slot 的衰减计数淘汰（不是 LRU）。

模块不维护 token ID。LIDU 返回的 ``destination_slots`` 去重后按 slot 更新
per-slot last_access / 衰减计数。MTP 相关约束与 V8 完全相同。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    from numba import njit, prange, set_num_threads
except ImportError as exc:  # pragma: no cover - 部署环境配置错误
    raise ImportError("Extent Pre-Victim (r1) requires numba; install it before importing.") from exc


BITMAP_WORD_BITS = 32
DEFAULT_BASE_POOL_CAPACITY = 13108
DEFAULT_LARGE_POOL_CAPACITY = 20480
DEFAULT_LARGE_LAYER_BEGIN = 40
DEFAULT_VICTIM_COUNT = 2048
DEFAULT_LIDU_FEEDBACK_COUNT = 8192

MTP4_QUERIES_PER_REQUEST = 4
# MTP 算子的限制保持不变；单 query 走独立的非 MTP ABI。
LIDU_MIN_QUERIES_PER_REQUEST = 2
LIDU_MAX_QUERIES_PER_REQUEST = 4
LIDU_SPARSE_COUNT = 2048
LIDU_MAX_CACHE_TOKENS = 16256
LIDU_BITMAP_BLOCK_WORDS = 8
LIDU_MISS_SLOT_BITMAP_WORDS = (
    ((LIDU_MAX_CACHE_TOKENS + BITMAP_WORD_BITS - 1) // BITMAP_WORD_BITS + LIDU_BITMAP_BLOCK_WORDS - 1)
    // LIDU_BITMAP_BLOCK_WORDS
    * LIDU_BITMAP_BLOCK_WORDS
)

# extent 淘汰参数（与 chunk-evict-replay 对齐）。
DEFAULT_EXTENT_GRANULARITY = 8
EXTENT_GRACE_ROUNDS = 2
EXTENT_DECAY_HALFLIFE = 64.0
_EMPTY_LAST_ACCESS = -1
_UNBOUNDED_ROUND = 2**30
_PROTECTED_SCORE = np.inf


@njit(inline="always")
def _single_bit_index(bit_value):
    """返回一个正 32-bit 整数中唯一置位 bit 的下标。"""

    index = 0
    if bit_value >= (1 << 16):
        bit_value >>= 16
        index += 16
    if bit_value >= (1 << 8):
        bit_value >>= 8
        index += 8
    if bit_value >= (1 << 4):
        bit_value >>= 4
        index += 4
    if bit_value >= (1 << 2):
        bit_value >>= 2
        index += 2
    if bit_value >= (1 << 1):
        index += 1
    return index


@njit(inline="always")
def _extent_score_r4(
    slot_last_acc,
    slot_cnt,
    slot_cnt_t,
    row,
    begin,
    end,
    current_round,
    decay_halflife,
):
    """r4: extent 分数 = 块内 max(衰减计数)；la_max 供 tie-break。"""

    la_max = np.int64(_EMPTY_LAST_ACCESS)
    score = 0.0
    for slot in range(begin, end):
        value = int(slot_last_acc[row, slot])
        if value > la_max:
            la_max = value
        if value >= 0:
            decay = np.exp2(-np.float64(current_round - int(slot_cnt_t[row, slot])) / decay_halflife)
            decayed = np.float64(slot_cnt[row, slot]) * decay
            if decayed > score:
                score = decayed
    return score, la_max


@njit
def _stable_argsort2(primary, secondary):
    """按 (primary, secondary) 升序的稳定归并排序，返回索引。"""

    count = primary.shape[0]
    order = np.arange(count, dtype=np.int64)
    if count <= 1:
        return order
    scratch = np.empty(count, dtype=np.int64)
    width = 1
    while width < count:
        for run in range(0, count, 2 * width):
            left = run
            middle = min(run + width, count)
            right = min(run + 2 * width, count)
            i = left
            j = middle
            out = left
            while i < middle and j < right:
                a = order[i]
                b = order[j]
                if (primary[a] < primary[b]) or (
                    primary[a] == primary[b] and secondary[a] <= secondary[b]
                ):
                    scratch[out] = a
                    i += 1
                else:
                    scratch[out] = b
                    j += 1
                out += 1
            while i < middle:
                scratch[out] = order[i]
                i += 1
                out += 1
            while j < right:
                scratch[out] = order[j]
                j += 1
                out += 1
        order, scratch = scratch, order
        width *= 2
    return order


@njit
def _select_victim_extents(
    row,
    current_round,
    active,
    slot_last_acc,
    slot_cnt,
    slot_cnt_t,
    ext_born,
    victim_slots,
    victim_bitmap,
    victim_count,
    bitmap_words,
    granularity,
    grace_rounds,
    decay_halflife,
):
    """按 extent 分数选出下一轮 victim，并写出 slot 列表与 ABI bitmap。

    victim_slots 按 extent 分数从低到高排列，extent 内 slot ID 升序。
    """

    extent_total = (active + granularity - 1) // granularity
    extent_needed = victim_count // granularity

    scores = np.empty(extent_total, dtype=np.float64)
    la_maxes = np.empty(extent_total, dtype=np.int64)
    for extent in range(extent_total):
        begin = extent * granularity
        end = min(begin + granularity, active)
        score, la_max = _extent_score_r4(
            slot_last_acc,
            slot_cnt,
            slot_cnt_t,
            row,
            begin,
            end,
            current_round,
            decay_halflife,
        )
        if current_round - int(ext_born[row, extent]) < grace_rounds:
            score = _PROTECTED_SCORE
        scores[extent] = score
        la_maxes[extent] = la_max

    if np.isinf(scores).all():
        # 全部 extent 都在保护期：按块内最旧访问强制淘汰（与 replay 一致）。
        order = _stable_argsort2(la_maxes.astype(np.float64), np.arange(extent_total, dtype=np.int64))
    else:
        order = _stable_argsort2(scores, la_maxes)

    for word in range(bitmap_words):
        victim_bitmap[row, word] = 0

    out = 0
    for rank in range(extent_needed):
        extent = order[rank] if rank < extent_total else extent_total - 1
        begin = extent * granularity
        for offset in range(granularity):
            slot = begin + offset
            if slot >= active:
                slot = active - 1
            victim_slots[row, out] = slot
            out += 1
            word = slot >> 5
            bit = slot & 31
            victim_bitmap[row, word] = np.int32(
                np.uint32(victim_bitmap[row, word]) | (np.uint32(1) << np.uint32(bit))
            )


@njit(parallel=True, nogil=True)
def _reset_extent_rows(
    flat_rows,
    active_capacities,
    rounds,
    slot_last_acc,
    slot_cnt,
    slot_cnt_t,
    ext_born,
    victim_slots,
    victim_bitmap,
    unique_counts,
    victim_count,
    bitmap_words,
    granularity,
    grace_rounds,
    decay_halflife,
):
    """把指定稳定行恢复为全空状态，victim 为最低编号的 extent。"""

    capacity = slot_last_acc.shape[1]
    for selected_index in prange(flat_rows.shape[0]):
        row = int(flat_rows[selected_index])
        active = int(active_capacities[row])
        for slot in range(capacity):
            slot_last_acc[row, slot] = _EMPTY_LAST_ACCESS
            slot_cnt[row, slot] = np.float32(0.0)
            slot_cnt_t[row, slot] = 0
        extent_total = (capacity + granularity - 1) // granularity
        for extent in range(extent_total):
            ext_born[row, extent] = -_UNBOUNDED_ROUND
        unique_counts[row] = 0
        _select_victim_extents(
            row,
            int(rounds[row]),
            active,
            slot_last_acc,
            slot_cnt,
            slot_cnt_t,
            ext_born,
            victim_slots,
            victim_bitmap,
            victim_count,
            bitmap_words,
            granularity,
            grace_rounds,
            decay_halflife,
        )


@njit(nogil=True)
def _feedback_slots_in_range(destination_slots, feedback_offsets, active_capacities):
    """在修改任意状态之前检查所有 packed feedback，保证失败原子性。"""

    for selected_index in range(active_capacities.size):
        active = int(active_capacities[selected_index])
        for index in range(feedback_offsets[selected_index], feedback_offsets[selected_index + 1]):
            slot = int(destination_slots[index])
            if slot < 0 or slot >= active:
                return False
    return True


@njit(parallel=True, nogil=True)
def _commit_extent_rows(
    flat_rows,
    destination_slots,
    feedback_offsets,
    active_capacities,
    rounds,
    slot_last_acc,
    slot_cnt,
    slot_cnt_t,
    ext_born,
    victim_slots,
    victim_bitmap,
    dedupe_bitmap,
    unique_counts,
    victim_count,
    bitmap_words,
    granularity,
    grace_rounds,
    decay_halflife,
):
    """提交指定稳定行的 Final slot，并缓存这些行的下一代 extent victim。

    访问 slot 更新 last_access / 衰减计数；落在上一代 victim 集合内的访问
    视为该 extent 刚被覆写，刷新其免淘汰保护起点。
    """

    for selected_index in prange(flat_rows.shape[0]):
        row = int(flat_rows[selected_index])
        active = int(active_capacities[row])
        active_words = (active + BITMAP_WORD_BITS - 1) // BITMAP_WORD_BITS
        current_round = int(rounds[row])

        for word in range(bitmap_words):
            dedupe_bitmap[row, word] = 0

        unique_count = 0
        # offsets 对应 layer-major 的稳定行顺序；混合 MTP 不做固定宽度 reshape。
        for index in range(feedback_offsets[selected_index], feedback_offsets[selected_index + 1]):
            slot = int(destination_slots[index])
            word = slot >> 5
            bit = slot & 31
            bit_mask = np.uint32(1) << np.uint32(bit)
            if np.uint32(dedupe_bitmap[row, word]) & bit_mask:
                continue
            dedupe_bitmap[row, word] = np.int32(np.uint32(dedupe_bitmap[row, word]) | bit_mask)
            unique_count += 1

            # 衰减到时基 current_round 再 +1（与 replay 的分数更新一致）。
            last_update = int(slot_cnt_t[row, slot])
            decay = np.exp2(-np.float64(current_round - last_update) / decay_halflife)
            slot_cnt[row, slot] = np.float32(np.float64(slot_cnt[row, slot]) * decay + 1.0)
            slot_cnt_t[row, slot] = current_round
            slot_last_acc[row, slot] = current_round

            # 命中上一代 victim 集合 = 该 slot 本轮被覆写，extent 获得保护。
            if np.uint32(victim_bitmap[row, word]) & bit_mask:
                ext_born[row, slot // granularity] = current_round
        unique_counts[row] = unique_count

        # 完成分数更新后，重新生成下一代 victim。
        _select_victim_extents(
            row,
            current_round,
            active,
            slot_last_acc,
            slot_cnt,
            slot_cnt_t,
            ext_born,
            victim_slots,
            victim_bitmap,
            victim_count,
            bitmap_words,
            granularity,
            grace_rounds,
            decay_halflife,
        )


@njit
def _materialize_row(
    row,
    active,
    slot_last_acc,
    slot_cnt,
    slot_cnt_t,
    ext_born,
    out,
    current_round,
    granularity,
    decay_halflife,
):
    """物化单行 oldest -> newest（淘汰优先级从低到高）顺序；诊断/测试用。"""

    extent_total = (active + granularity - 1) // granularity
    scores = np.empty(extent_total, dtype=np.float64)
    la_maxes = np.empty(extent_total, dtype=np.int64)
    for extent in range(extent_total):
        begin = extent * granularity
        end = min(begin + granularity, active)
        score, la_max = _extent_score_r4(
            slot_last_acc,
            slot_cnt,
            slot_cnt_t,
            row,
            begin,
            end,
            current_round,
            decay_halflife,
        )
        scores[extent] = score
        la_maxes[extent] = la_max
    order = _stable_argsort2(scores, la_maxes)
    out_index = 0
    for rank in range(extent_total):
        extent = order[rank]
        begin = extent * granularity
        end = min(begin + granularity, active)
        for slot in range(begin, end):
            out[out_index] = slot
            out_index += 1
    return out_index


@dataclass(frozen=True, slots=True)
class MultiLayerLRUVictimSelectionV8:
    """一批 active request 对应的 Pre-Victim 快照（extent 粒度）。"""

    victim_slots: NDArray[np.int32]  # [L,A,K]，extent 分数低 -> 高，块内升序
    victim_bitmap: NDArray[np.int32]  # [L,A,512]，LIDU ABI
    generations: NDArray[np.int64]  # [L,A]，commit 时必须原样传回
    pool_rows: NDArray[np.int64]  # [A]，active row -> stable pool row


class MultiLayerLRUPreVictimV8:
    """为 ``[layer, stable_pool_row]`` 保存物理 slot 的 extent 粒度淘汰状态。

    公开接口与纯 LRU V8 完全一致；新增可选构造参数 ``extent_granularity``
    （默认 8），调用方不传即为 8 槽 extent。
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        capacity: int = DEFAULT_LARGE_POOL_CAPACITY,
        *,
        victim_count: int = DEFAULT_VICTIM_COUNT,
        feedback_count: int = DEFAULT_LIDU_FEEDBACK_COUNT,
        active_capacities: ArrayLike | int | None = None,
        max_workers: int = 1,
        extent_granularity: int = DEFAULT_EXTENT_GRANULARITY,
    ) -> None:
        """分配状态；``batch_size`` 是兼容旧接口的稳定 pool 大小。

        ``feedback_count`` 只约束 ``commit_rows`` 的定长反馈；
        ``commit_lidu_mtp`` 按本次累计 query 长度处理变长反馈。
        """

        self.num_layers = int(num_layers)
        self.pool_size = int(batch_size)  # 稳定 request 行数量
        self.capacity = int(capacity)  # 每行最多分配多少个物理 slot
        self.victim_count = int(victim_count)
        self.feedback_count = int(feedback_count)
        self.max_workers = int(max_workers)
        self.extent_granularity = int(extent_granularity)
        self.row_count = self.num_layers * self.pool_size  # 把二维坐标展平为一维 stable row
        self.bitmap_words = LIDU_MISS_SLOT_BITMAP_WORDS
        self._validate_config()  # 检查初始化参数是否合理
        self._set_num_threads()  # 设置 numba 的线程数，并行粒度是 stable row。

        # active_capacity 是某一行真正使用多少个 slot。
        self._active_capacity = self._build_active_capacities(active_capacities)
        row_slot_shape = (self.row_count, self.capacity)
        extent_count = (self.capacity + self.extent_granularity - 1) // self.extent_granularity

        # 每行的淘汰轮计数，commit 时递增，作为 last_access/衰减的时基。
        self._round = np.zeros(self.row_count, dtype=np.int64)
        # per-slot 访问状态：last_access 轮、衰减计数及其时基。
        self._slot_last_acc = np.full(row_slot_shape, _EMPTY_LAST_ACCESS, dtype=np.int32)
        self._slot_cnt = np.zeros(row_slot_shape, dtype=np.float32)
        self._slot_cnt_t = np.zeros(row_slot_shape, dtype=np.int32)
        # per-extent 最近一次被覆写的轮，用于免淘汰保护。
        self._ext_born = np.full((self.row_count, extent_count), -_UNBOUNDED_ROUND, dtype=np.int32)
        # 每一行的 generation，commit 时必须原样传回，否则会报错。
        self._generation = np.zeros(self.row_count, dtype=np.int64)

        # 保存下一轮 victim 的有序列表（extent 分数低 -> 高）。
        self._victim_slots = np.empty((self.row_count, self.victim_count), dtype=np.int32)
        # 保存同一批 victim 的 bitmap 形式，供 LIDU 使用。
        self._victim_bitmap = np.zeros((self.row_count, self.bitmap_words), dtype=np.int32)
        # commit 用的去重 scratch（victim_bitmap 要保留用于覆写判定）。
        self._dedupe_bitmap = np.zeros((self.row_count, self.bitmap_words), dtype=np.int32)
        # 记录最近一次 commit 中，LIDU 返回了多少个不重复的 destination slot。
        self._unique_counts = np.zeros(self.row_count, dtype=np.int32)

        # 重置所有 (layer, pool_row) 为全空状态。
        _reset_extent_rows(
            np.arange(self.row_count, dtype=np.int64),
            self._active_capacity,
            self._round,
            self._slot_last_acc,
            self._slot_cnt,
            self._slot_cnt_t,
            self._ext_born,
            self._victim_slots,
            self._victim_bitmap,
            self._unique_counts,
            self.victim_count,
            self.bitmap_words,
            self.extent_granularity,
            EXTENT_GRACE_ROUNDS,
            EXTENT_DECAY_HALFLIFE,
        )

    def _set_num_threads(self) -> None:
        try:
            set_num_threads(self.max_workers)
        except ValueError as exc:
            raise ValueError("max_workers exceeds Numba's configured thread limit") from exc

    def _validate_config(self) -> None:
        if self.num_layers <= 0 or self.pool_size <= 0:
            raise ValueError("num_layers and batch_size/pool_size must be positive")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.victim_count <= 0 or self.victim_count > self.capacity:
            raise ValueError("victim_count must be in [1, capacity]")
        if self.feedback_count <= 0:
            raise ValueError("feedback_count must be positive")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if self.extent_granularity <= 0 or self.extent_granularity > self.victim_count:
            raise ValueError("extent_granularity must be in [1, victim_count]")
        if self.victim_count % self.extent_granularity != 0:
            raise ValueError("victim_count must be a multiple of extent_granularity")

    def _build_active_capacities(self, values: ArrayLike | int | None) -> NDArray[np.int32]:
        expected = (self.num_layers, self.pool_size)
        if values is None:
            if self.capacity < DEFAULT_BASE_POOL_CAPACITY:
                raise ValueError("small test capacities must pass active_capacities explicitly")
            if self.num_layers > DEFAULT_LARGE_LAYER_BEGIN:
                raise ValueError(
                    "the legacy 20480-slot layers exceed LIDU's 16256-token "
                    "limit; pass LIDU-compatible active_capacities explicitly"
                )
            array = np.full(
                expected,
                min(DEFAULT_BASE_POOL_CAPACITY, self.capacity),
                dtype=np.int32,
            )
        else:
            raw = np.asarray(values)
            if not np.issubdtype(raw.dtype, np.integer):
                raise TypeError("active_capacities must contain integers")
            if raw.ndim == 0:
                array = np.full(expected, raw.item())
            elif raw.shape == expected:
                array = raw.copy()
            elif raw.shape == (self.num_layers,):
                array = np.broadcast_to(raw[:, None], expected).copy()
            else:
                raise ValueError("active_capacities must be scalar, [L], or [L,pool_size]")

        if np.any(array < self.victim_count) or np.any(array > self.capacity):
            raise ValueError("active_capacities must be within [victim_count, capacity]")
        if np.any(array > LIDU_MAX_CACHE_TOKENS):
            raise ValueError(f"active_capacities exceed LIDU max {LIDU_MAX_CACHE_TOKENS}")
        return np.ascontiguousarray(array, dtype=np.int32).reshape(-1)

    @property
    def last_unique_final_counts(self) -> NDArray[np.int32]:
        view = self._unique_counts.reshape(self.num_layers, self.pool_size).view()
        view.flags.writeable = False
        return view

    @property
    def persistent_state_nbytes(self) -> int:
        arrays = (
            self._active_capacity,
            self._round,
            self._slot_last_acc,
            self._slot_cnt,
            self._slot_cnt_t,
            self._ext_born,
            self._generation,
            self._victim_slots,
            self._victim_bitmap,
            self._dedupe_bitmap,
            self._unique_counts,
        )
        return sum(array.nbytes for array in arrays)

    def _checked_pool_rows(self, pool_rows: ArrayLike) -> NDArray[np.int64]:
        raw = np.asarray(pool_rows)
        if raw.ndim != 1 or raw.size == 0:
            raise ValueError("pool_rows must be a non-empty 1-D integer array")
        if not np.issubdtype(raw.dtype, np.integer):
            raise TypeError("pool_rows must contain integers")
        rows = np.ascontiguousarray(raw, dtype=np.int64)
        if np.any(rows < 0) or np.any(rows >= self.pool_size):
            raise ValueError("pool_rows contains an out-of-range stable row")
        if np.unique(rows).size != rows.size:
            raise ValueError("pool_rows must not contain duplicates")
        return rows

    def _flat_rows(self, pool_rows: NDArray[np.int64]) -> NDArray[np.int64]:
        layer_offsets = np.arange(self.num_layers, dtype=np.int64)[:, None] * self.pool_size
        return np.ascontiguousarray((layer_offsets + pool_rows[None, :]).reshape(-1))

    def _selection(
        self,
        pool_rows: NDArray[np.int64],
        flat_rows: NDArray[np.int64],
    ) -> MultiLayerLRUVictimSelectionV8:
        active_count = pool_rows.size

        # 从持久状态中取出这些行，reshape 成 LIDU/调用方需要的形状
        slots = self._victim_slots[flat_rows].reshape(self.num_layers, active_count, self.victim_count)
        bitmap = self._victim_bitmap[flat_rows].reshape(self.num_layers, active_count, self.bitmap_words)
        generations = self._generation[flat_rows].reshape(self.num_layers, active_count)

        slots.flags.writeable = False
        bitmap.flags.writeable = False
        generations.flags.writeable = False
        selected_rows = pool_rows.copy()
        selected_rows.flags.writeable = False
        return MultiLayerLRUVictimSelectionV8(
            victim_slots=slots,
            victim_bitmap=bitmap,
            generations=generations,
            pool_rows=selected_rows,
        )

    def prepare_rows(self, pool_rows: ArrayLike) -> MultiLayerLRUVictimSelectionV8:
        """按当前 active-batch 次序 gather 稳定 pool row 的 victim。"""

        checked_rows = self._checked_pool_rows(pool_rows)
        return self._selection(checked_rows, self._flat_rows(checked_rows))

    def prepare_rows_into(
        self,
        pool_rows: ArrayLike,
        *,
        victim_slots: NDArray[np.int32],
        victim_bitmap: NDArray[np.int32],
        output_rows: ArrayLike,
    ) -> NDArray[np.int64]:
        """Gather victims directly into caller-owned host buffers."""

        checked_rows = self._checked_pool_rows(pool_rows)
        checked_output_rows = np.asarray(output_rows)
        active_count = checked_rows.size
        if (
            checked_output_rows.shape != (active_count,)
            or not np.issubdtype(checked_output_rows.dtype, np.integer)
        ):
            raise ValueError(f"output_rows must be an integer array with shape ({active_count},)")
        checked_output_rows = np.ascontiguousarray(
            checked_output_rows,
            dtype=np.int64,
        )

        slots_out = np.asarray(victim_slots)
        bitmap_out = np.asarray(victim_bitmap)
        if (
            slots_out.ndim != 3
            or slots_out.shape[0] != self.num_layers
            or slots_out.shape[2] != self.victim_count
            or slots_out.dtype != np.int32
        ):
            raise ValueError(
                "victim_slots must be int32 [num_layers, output_rows, victim_count]"
            )
        if (
            bitmap_out.ndim != 3
            or bitmap_out.shape[0] != self.num_layers
            or bitmap_out.shape[2] != self.bitmap_words
            or bitmap_out.dtype != np.int32
        ):
            raise ValueError(
                "victim_bitmap must be int32 [num_layers, output_rows, bitmap_words]"
            )
        output_capacity = slots_out.shape[1]
        if bitmap_out.shape[1] != output_capacity:
            raise ValueError("victim_slots and victim_bitmap output row counts differ")
        if (
            np.any(checked_output_rows < 0)
            or np.any(checked_output_rows >= output_capacity)
            or np.unique(checked_output_rows).size != active_count
        ):
            raise ValueError("output_rows contains an invalid or duplicate destination row")

        slots = self._victim_slots.reshape(
            self.num_layers,
            self.pool_size,
            self.victim_count,
        )
        bitmap = self._victim_bitmap.reshape(
            self.num_layers,
            self.pool_size,
            self.bitmap_words,
        )
        for pool_row, output_row in zip(checked_rows, checked_output_rows):
            slots_out[:, output_row] = slots[:, pool_row]
            bitmap_out[:, output_row] = bitmap[:, pool_row]

        flat_rows = self._flat_rows(checked_rows)
        generations = self._generation[flat_rows].reshape(
            self.num_layers,
            active_count,
        )
        generations.flags.writeable = False
        return generations

    def _checked_generations(
        self,
        generations: ArrayLike,
        flat_rows: NDArray[np.int64],
        active_count: int,
    ) -> None:
        array = np.asarray(generations)
        expected = (self.num_layers, active_count)
        if array.shape != expected or array.dtype != np.int64:
            raise ValueError(f"generations must be int64 with shape {expected}")
        if not np.array_equal(array.reshape(-1), self._generation[flat_rows]):
            raise ValueError("generation mismatch for one or more stable pool rows")

    def _checked_feedback(
        self,
        destination_slots: ArrayLike,
        active_count: int,
    ) -> NDArray[np.int32]:
        array = np.asarray(destination_slots)
        if array.ndim == 4 and array.shape[2] == 1:
            array = array[:, :, 0, :]
        expected = (self.num_layers, active_count, self.feedback_count)
        if array.shape != expected:
            raise ValueError(f"destination_slots must have shape {expected} or [L,A,1,{self.feedback_count}]")
        if array.dtype != np.int32:
            raise TypeError("destination_slots must use int32")
        if not array.flags.c_contiguous:
            raise ValueError("destination_slots must be C-contiguous")
        return array

    def _checked_miss_counts(
        self,
        miss_counts: ArrayLike,
        active_count: int,
        miss_limits: NDArray[np.int64],
    ) -> NDArray[np.int32]:
        array = np.asarray(miss_counts)
        expected = (self.num_layers, active_count)
        raw_shape = (self.num_layers, active_count + 1)
        if array.shape not in (expected, raw_shape) or array.dtype != np.int32:
            raise ValueError(f"miss_counts must be int32 with shape {expected} or {raw_shape}")
        counts = array[:, :active_count]
        if np.any(counts < 0) or np.any(counts > miss_limits):
            raise ValueError("miss_counts exceed the per-request query/capacity limit or contain failure status")
        # !23 的最后一项是全 batch 的 compact descriptor 有效长度，不是请求反馈。
        # 在任何状态更新前验证；int64 求和避免恶意/损坏计数溢出。
        if array.shape == raw_shape and np.any(array[:, -1] != counts.sum(axis=1, dtype=np.int64)):
            raise ValueError("miss_counts total does not match the sum of per-request counts")
        return counts

    def _checked_cache_tokens(self, cache_tokens: ArrayLike | int, active_count: int) -> NDArray[np.int32]:
        raw = np.asarray(cache_tokens)
        if not np.issubdtype(raw.dtype, np.integer):
            raise TypeError("cache_tokens must contain integers")
        expected = (self.num_layers, active_count)
        if raw.ndim == 0:
            array = np.full(expected, raw.item(), dtype=np.int32)
        elif raw.shape == (active_count,):
            array = np.broadcast_to(raw[None, :], expected).astype(np.int32, copy=True)
        elif raw.shape == expected:
            array = raw.astype(np.int32, copy=False)
        else:
            raise ValueError("cache_tokens must be scalar, [A], or [L,A]")
        return np.ascontiguousarray(array)

    def commit_rows(
        self,
        *,
        pool_rows: ArrayLike,
        generations: ArrayLike,
        destination_slots: ArrayLike,
        miss_counts: ArrayLike,
        cache_tokens: ArrayLike | int,
    ) -> MultiLayerLRUVictimSelectionV8:
        """提交 LIDU 反馈到稳定行，并返回同一批行的下一代 victim。"""

        # check校验输入是否有误
        checked_rows = self._checked_pool_rows(pool_rows)
        feedback = self._checked_feedback(destination_slots, checked_rows.size)
        feedback_offsets = np.arange(self.num_layers * checked_rows.size + 1, dtype=np.int64) * self.feedback_count
        return self._commit_feedback(
            checked_rows=checked_rows,
            generations=generations,
            flat_feedback=feedback.reshape(-1),
            feedback_offsets=feedback_offsets,
            miss_counts=miss_counts,
            cache_tokens=cache_tokens,
        )

    def _commit_feedback(
        self,
        *,
        checked_rows: NDArray[np.int64],
        generations: ArrayLike,
        flat_feedback: NDArray[np.int32],
        feedback_offsets: NDArray[np.int64],
        miss_counts: ArrayLike,
        cache_tokens: ArrayLike | int,
        max_miss_counts: NDArray[np.int64] | None = None,
        return_selection: bool = True,
    ) -> MultiLayerLRUVictimSelectionV8 | None:
        """定长和 packed MTP 共用的全层校验、原子提交入口。"""

        active_count = checked_rows.size
        flat_rows = self._flat_rows(checked_rows)
        self._checked_generations(generations, flat_rows, active_count)
        checked_cache_tokens = self._checked_cache_tokens(cache_tokens, active_count)

        selected_capacities = self._active_capacity[flat_rows].reshape(self.num_layers, active_count)
        miss_limits = np.minimum(
            selected_capacities,
            self.victim_count if max_miss_counts is None else max_miss_counts,
        ).astype(np.int64)
        self._checked_miss_counts(miss_counts, active_count, miss_limits)
        if not np.array_equal(checked_cache_tokens, selected_capacities):
            raise ValueError("cache_tokens does not match pre_victim active capacity for one or more stable rows")
        if not _feedback_slots_in_range(flat_feedback, feedback_offsets, selected_capacities.reshape(-1)):
            raise ValueError("destination_slots contains a slot outside its row's cache_tokens")

        # 所有输入在状态修改前完成校验，失败不会留下部分更新。
        self._set_num_threads()
        # 淘汰轮计数在 kernel 之前递增，kernel 内以新轮次作为时基。
        self._round[flat_rows] += 1
        # 进入真正的 Numba kernel
        _commit_extent_rows(
            flat_rows,
            flat_feedback,
            feedback_offsets,
            self._active_capacity,
            self._round,
            self._slot_last_acc,
            self._slot_cnt,
            self._slot_cnt_t,
            self._ext_born,
            self._victim_slots,
            self._victim_bitmap,
            self._dedupe_bitmap,
            self._unique_counts,
            self.victim_count,
            self.bitmap_words,
            self.extent_granularity,
            EXTENT_GRACE_ROUNDS,
            EXTENT_DECAY_HALFLIFE,
        )
        self._generation[flat_rows] += 1
        if not return_selection:
            return None
        # 返回同一批稳定行的下一轮 victim。
        return self._selection(checked_rows, flat_rows)

    def commit_lidu(
        self,
        *,
        pool_rows: ArrayLike,
        generations: ArrayLike,
        destination_slots: ArrayLike,
        miss_counts: ArrayLike,
        cache_tokens: ArrayLike | int,
        actual_seq_lengths_q: ArrayLike | None = None,
    ) -> MultiLayerLRUVictimSelectionV8:
        """统一反馈入口；不传 query 累计长度表示不开 MTP。

        全 1 query（累计 [1,2,...,A]）对应非 MTP LIDU；每请求 2--4 query
        对应 MTP LIDU。这里只提交反馈，不执行或替调用方选择 NPU 算子。
        两个算子的 ABI 不同，1 query 与 2--4 query 不能混在一次调用中。
        miss_counts 接受 !23 原生 [L,A+1] 或已去掉总数的 [L,A]。
        """

        commit = self.commit_lidu_decode
        extra = {}
        if actual_seq_lengths_q is not None:
            checked_rows = self._checked_pool_rows(pool_rows)
            lengths = np.asarray(actual_seq_lengths_q)
            if lengths.shape != (checked_rows.size,) or lengths.dtype != np.int32:
                raise ValueError(f"actual_seq_lengths_q must be int32 with shape ({checked_rows.size},)")
            query_counts = np.diff(np.concatenate((np.zeros(1, dtype=np.int64), lengths.astype(np.int64))))
            if not np.all(query_counts == 1):
                if np.any(query_counts == 1):
                    raise ValueError("non-MTP (1 query) and MTP (2--4 queries) require separate LIDU calls")
                commit = self.commit_lidu_mtp
                extra["actual_seq_lengths_q"] = lengths
        return commit(
            pool_rows=pool_rows,
            generations=generations,
            destination_slots=destination_slots,
            miss_counts=miss_counts,
            cache_tokens=cache_tokens,
            **extra,
        )

    def commit_lidu_decode(
        self,
        *,
        pool_rows: ArrayLike,
        generations: ArrayLike,
        destination_slots: ArrayLike,
        miss_counts: ArrayLike,
        cache_tokens: ArrayLike | int,
    ) -> MultiLayerLRUVictimSelectionV8:
        """提交非 MTP LIDU 的 int32[L,A,1,2048]（也接受 [L,A,2048]）。

        每请求恰好 1 条 query；独立于构造参数 feedback_count，因此同一份
        状态可以跨 step 切换 decode/MTP，不需要重建状态或补齐反馈。
        """

        if self.victim_count != LIDU_SPARSE_COUNT:
            raise ValueError(f"non-MTP LIDU requires victim_count={LIDU_SPARSE_COUNT}")
        checked_rows = self._checked_pool_rows(pool_rows)
        raw = np.asarray(destination_slots)
        if raw.dtype != np.int32:
            raise TypeError("destination_slots must use int32")
        expected = (self.num_layers, checked_rows.size, LIDU_SPARSE_COUNT)
        if raw.shape not in (expected, (self.num_layers, checked_rows.size, 1, LIDU_SPARSE_COUNT)):
            raise ValueError(f"non-MTP destination_slots must have shape {expected} or [L,A,1,2048]")
        return self._commit_feedback(
            checked_rows=checked_rows,
            generations=generations,
            flat_feedback=np.ascontiguousarray(raw).reshape(-1),
            feedback_offsets=np.arange(self.num_layers * checked_rows.size + 1, dtype=np.int64) * LIDU_SPARSE_COUNT,
            miss_counts=miss_counts,
            cache_tokens=cache_tokens,
        )

    def commit_lidu_mtp(
        self,
        *,
        pool_rows: ArrayLike,
        generations: ArrayLike,
        destination_slots: ArrayLike,
        miss_counts: ArrayLike,
        actual_seq_lengths_q: ArrayLike,
        cache_tokens: ArrayLike | int,
    ) -> MultiLayerLRUVictimSelectionV8:
        """提交 LIDU 原生 ``int32[L,T,1,2048]`` 输出，T 为总 query 数。

        ``actual_seq_lengths_q`` 对应算子的 ``actual_seq_lengths_query``，
        必须是 int32[A] 累计结束位置（不含开头的 0），每段长度为 2/3/4。
        例如 [2,5,9] 对应三个请求的 2/3/4 条 query。所有层共用该分段。
        同质 batch 还接受 [L,A,Q,2048] 或 [L,A,Q*2048]，Q 为 2/3/4；
        混合 batch 只接受原生 packed 布局，不接受补齐或猜测的请求边界。
        每请求 miss 上限为 min(Q*2048, cache_tokens)，与 2048 个候选槽位数无关。
        """

        if self.victim_count != LIDU_SPARSE_COUNT:
            raise ValueError(f"MTP LIDU requires victim_count={LIDU_SPARSE_COUNT}")

        checked_rows = self._checked_pool_rows(pool_rows)
        active_count = checked_rows.size
        lengths = np.asarray(actual_seq_lengths_q)
        if lengths.shape != (active_count,) or lengths.dtype != np.int32:
            raise ValueError(f"actual_seq_lengths_q must be int32 with shape ({active_count},)")
        # 先升为 int64 再求差，防止非法 int32 累计长度减法溢出。
        query_offsets = np.concatenate((np.zeros(1, dtype=np.int64), lengths.astype(np.int64)))
        query_counts = np.diff(query_offsets)
        if np.any(query_counts < LIDU_MIN_QUERIES_PER_REQUEST) or np.any(query_counts > LIDU_MAX_QUERIES_PER_REQUEST):
            raise ValueError("actual_seq_lengths_q must be cumulative lengths with 2--4 queries per request")

        raw = np.asarray(destination_slots)
        if raw.dtype != np.int32:
            raise TypeError("destination_slots must use int32")
        total_queries = int(query_offsets[-1])
        native_shape = (
            self.num_layers,
            total_queries,
            1,
            LIDU_SPARSE_COUNT,
        )
        allowed_shapes = [native_shape]
        if np.all(query_counts == query_counts[0]):
            # 同质 batch 保留已分组和已展平的输入；混合 batch 不能这样 reshape。
            queries_per_request = int(query_counts[0])
            allowed_shapes.extend(
                [
                    (self.num_layers, active_count, queries_per_request, LIDU_SPARSE_COUNT),
                    (self.num_layers, active_count, queries_per_request * LIDU_SPARSE_COUNT),
                ]
            )
        if raw.shape not in allowed_shapes:
            raise ValueError(f"MTP destination_slots must have shape in {allowed_shapes}; got {raw.shape}")

        # 每层的 packed query 连续存储；每请求消费全部 hit 和安装后的 final slot。
        # 不按 miss_counts 截断，也不填充假的 slot（0 本身是合法物理 slot）。
        layer_offsets = np.arange(self.num_layers, dtype=np.int64)[:, None] * total_queries
        feedback_starts = (layer_offsets + query_offsets[None, :-1]).reshape(-1) * LIDU_SPARSE_COUNT
        # Packed inputs reshape without another full CPU copy.
        flat_feedback = (
            raw.reshape(-1)
            if raw.flags.c_contiguous
            else np.ascontiguousarray(raw).reshape(-1)
        )
        feedback_offsets = np.append(feedback_starts, flat_feedback.size)
        # 真正修改淘汰状态的 Python 入口
        return self._commit_feedback(
            checked_rows=checked_rows,
            generations=generations,
            flat_feedback=flat_feedback,
            feedback_offsets=feedback_offsets,
            miss_counts=miss_counts,
            cache_tokens=cache_tokens,
            max_miss_counts=query_counts[None, :] * LIDU_SPARSE_COUNT,
        )

    def commit_lidu_mixed(
        self,
        *,
        pool_rows: ArrayLike,
        generations: ArrayLike,
        destination_slots: ArrayLike,
        miss_counts: ArrayLike,
        query_counts: ArrayLike,
        cache_tokens: ArrayLike | int,
        return_selection: bool = True,
    ) -> MultiLayerLRUVictimSelectionV8 | None:
        """Atomically commit feedback from native decode and MTP cohorts.

        The device runtime may split one sparse batch into the native decode
        operator for one-query requests and the native MTP operator for
        two-to-four-query requests.  Callers must concatenate the final
        destination rows back into original sparse-request order before this
        entry point.  No query is padded and neither operator ABI is changed.
        """

        if self.victim_count != LIDU_SPARSE_COUNT:
            raise ValueError(f"mixed LIDU requires victim_count={LIDU_SPARSE_COUNT}")
        checked_rows = self._checked_pool_rows(pool_rows)
        active_count = checked_rows.size
        raw_counts = np.asarray(query_counts)
        if raw_counts.shape != (active_count,) or not np.issubdtype(raw_counts.dtype, np.integer):
            raise ValueError(f"query_counts must be an integer array with shape ({active_count},)")
        checked_query_counts = np.ascontiguousarray(raw_counts, dtype=np.int64)
        if np.any(checked_query_counts < 1) or np.any(checked_query_counts > LIDU_MAX_QUERIES_PER_REQUEST):
            raise ValueError("query_counts must contain values in [1,4]")

        query_offsets = np.concatenate(
            (
                np.zeros(1, dtype=np.int64),
                np.cumsum(checked_query_counts, dtype=np.int64),
            )
        )
        total_queries = int(query_offsets[-1])
        raw = np.asarray(destination_slots)
        if raw.dtype != np.int32:
            raise TypeError("destination_slots must use int32")
        expected = (
            self.num_layers,
            total_queries,
            1,
            LIDU_SPARSE_COUNT,
        )
        if raw.shape != expected:
            raise ValueError(f"mixed destination_slots must have shape {expected}; got {raw.shape}")

        layer_offsets = np.arange(self.num_layers, dtype=np.int64)[:, None] * total_queries
        feedback_starts = (layer_offsets + query_offsets[None, :-1]).reshape(-1) * LIDU_SPARSE_COUNT
        # PreVictimRuntime stages active feedback into a packed pinned host
        # buffer, so its steady path reshapes without another full CPU copy.
        flat_feedback = (
            raw.reshape(-1)
            if raw.flags.c_contiguous
            else np.ascontiguousarray(raw).reshape(-1)
        )
        feedback_offsets = np.append(feedback_starts, flat_feedback.size)
        return self._commit_feedback(
            checked_rows=checked_rows,
            generations=generations,
            flat_feedback=flat_feedback,
            feedback_offsets=feedback_offsets,
            miss_counts=miss_counts,
            cache_tokens=cache_tokens,
            max_miss_counts=(checked_query_counts[None, :] * LIDU_SPARSE_COUNT),
            return_selection=return_selection,
        )

    def commit_lidu_mtp4(
        self,
        *,
        pool_rows: ArrayLike,
        generations: ArrayLike,
        destination_slots: ArrayLike,
        miss_counts: ArrayLike,
        actual_seq_lengths_q: ArrayLike,
        cache_tokens: ArrayLike | int,
    ) -> MultiLayerLRUVictimSelectionV8:
        """保留固定 MTP4 兼容入口；新调用方请使用 ``commit_lidu_mtp``。"""

        checked_rows = self._checked_pool_rows(pool_rows)
        expected = np.arange(1, checked_rows.size + 1, dtype=np.int64) * MTP4_QUERIES_PER_REQUEST
        if not np.array_equal(np.asarray(actual_seq_lengths_q), expected):
            raise ValueError("MTP4 requires packed query prefix lengths [4, 8, ..., 4*A]")
        return self.commit_lidu_mtp(
            pool_rows=checked_rows,
            generations=generations,
            destination_slots=destination_slots,
            miss_counts=miss_counts,
            actual_seq_lengths_q=actual_seq_lengths_q,
            cache_tokens=cache_tokens,
        )

    def reset_rows(
        self,
        pool_rows: ArrayLike,
        *,
        active_capacities: ArrayLike | int | None = None,
    ) -> MultiLayerLRUVictimSelectionV8:
        """在稳定 pool row 被新 request 复用时清空其全部层的淘汰状态。"""

        checked_rows = self._checked_pool_rows(pool_rows)
        active_count = checked_rows.size
        flat_rows = self._flat_rows(checked_rows)
        if active_capacities is not None:
            raw = np.asarray(active_capacities)
            if not np.issubdtype(raw.dtype, np.integer):
                raise TypeError("active_capacities must contain integers")
            expected = (self.num_layers, active_count)
            if raw.ndim == 0:
                capacities = np.full(expected, raw.item(), dtype=np.int32)
            elif raw.shape == (active_count,):
                capacities = np.broadcast_to(raw[None, :], expected).astype(np.int32, copy=True)
            elif raw.shape == (self.num_layers,):
                capacities = np.broadcast_to(raw[:, None], expected).astype(np.int32, copy=True)
            elif raw.shape == expected:
                capacities = raw.astype(np.int32, copy=False)
            else:
                raise ValueError("reset active_capacities must be scalar, [A], [L], or [L,A]")
            if np.any(capacities < self.victim_count) or np.any(capacities > self.capacity):
                raise ValueError("reset active_capacities must be within [victim_count, capacity]")
            if np.any(capacities > LIDU_MAX_CACHE_TOKENS):
                raise ValueError(f"active_capacities exceed LIDU max {LIDU_MAX_CACHE_TOKENS}")
            self._active_capacity[flat_rows] = np.ascontiguousarray(capacities, dtype=np.int32).reshape(-1)

        self._set_num_threads()
        _reset_extent_rows(
            flat_rows,
            self._active_capacity,
            self._round,
            self._slot_last_acc,
            self._slot_cnt,
            self._slot_cnt_t,
            self._ext_born,
            self._victim_slots,
            self._victim_bitmap,
            self._unique_counts,
            self.victim_count,
            self.bitmap_words,
            self.extent_granularity,
            EXTENT_GRACE_ROUNDS,
            EXTENT_DECAY_HALFLIFE,
        )
        self._generation[flat_rows] += 1
        return self._selection(checked_rows, flat_rows)

    def materialize_lru_order(self, pool_rows: ArrayLike | None = None) -> NDArray[np.int32]:
        """物化 oldest -> newest（淘汰优先级）顺序；只用于测试和诊断。

        extent 版本按 extent 分数排序，extent 内 slot ID 升序；容量外为 -1。
        """

        if pool_rows is None:
            checked_rows = np.arange(self.pool_size, dtype=np.int64)
        else:
            checked_rows = self._checked_pool_rows(pool_rows)
        flat_rows = self._flat_rows(checked_rows)
        order = np.full((flat_rows.size, self.capacity), -1, dtype=np.int32)
        for selected_index, row in enumerate(flat_rows):
            active = int(self._active_capacity[row])
            written = _materialize_row(
                int(row),
                active,
                self._slot_last_acc,
                self._slot_cnt,
                self._slot_cnt_t,
                self._ext_born,
                order[selected_index],
                int(self._round[row]),
                self.extent_granularity,
                EXTENT_DECAY_HALFLIFE,
            )
            if written != active:
                raise RuntimeError("extent materialization did not cover the active capacity")
        return order.reshape(self.num_layers, checked_rows.size, self.capacity)


__all__ = [
    "BITMAP_WORD_BITS",
    "DEFAULT_BASE_POOL_CAPACITY",
    "DEFAULT_EXTENT_GRANULARITY",
    "DEFAULT_LARGE_LAYER_BEGIN",
    "DEFAULT_LARGE_POOL_CAPACITY",
    "DEFAULT_LIDU_FEEDBACK_COUNT",
    "DEFAULT_VICTIM_COUNT",
    "EXTENT_DECAY_HALFLIFE",
    "EXTENT_GRACE_ROUNDS",
    "LIDU_MAX_CACHE_TOKENS",
    "LIDU_MAX_QUERIES_PER_REQUEST",
    "LIDU_MIN_QUERIES_PER_REQUEST",
    "LIDU_MISS_SLOT_BITMAP_WORDS",
    "LIDU_SPARSE_COUNT",
    "MTP4_QUERIES_PER_REQUEST",
    "MultiLayerLRUPreVictimV8",
    "MultiLayerLRUVictimSelectionV8",
]
