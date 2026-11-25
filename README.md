# extent-eviction

KVAF hot-slot 淘汰策略：extent 粒度 Pre-Victim 的三个可直接替换实现（R1 / R2 / R4）。

背景：CPU SGL 直写 HBM hot slot 时，淘汰必须以物理连续的 8/16 个 slot
（extent）为粒度，才能按连续块直写。逐 slot LRU 无法保证这一点。
本仓三个文件把 `pre_victim_lru.py`（纯 LRU V8）替换为 extent 粒度打分淘汰，
replay 验证结果见实验仓（16k 容量、g=8 时 R4 的 miss 5.86%，逐 token LRU 9.18%）。

## 文件

| 文件 | 规则 | extent 分数 |
|---|---|---|
| `pre_victim_extent_r1.py` | extent-LRU | 块内 max(last_access) |
| `pre_victim_extent_r2.py` | 衰减频率和 | 块内 Σ 衰减计数（半衰期 64 轮） |
| `pre_victim_extent_r4.py` | 最强 token 保护 | 块内 max（衰减计数） |

## 用法

三个文件都是 `pre_victim_lru.py` 的 drop-in 替换：类名
`MultiLayerLRUPreVictimV8`、构造参数、全部公开方法、LIDU victim bitmap ABI
与 V8 完全一致。想用哪个策略，把哪个文件内容覆盖目标仓的
`vllm_ascend/dsa_offload/pre_victim_lru.py` 即可，其他代码零改动：

```bash
cp pre_victim_extent_r4.py /path/to/vllm_ascend/dsa_offload/pre_victim_lru.py
```

默认 extent 粒度 8 槽，16 槽在构造时传 `extent_granularity=16`；要求
`victim_count % extent_granularity == 0`（默认 victim_count=2048）。

## 语义（与 replay 实验逐条对齐）

- 淘汰单位 = 对齐的连续 `g` 个物理 slot；每轮 victim 为
  `victim_count // g` 个完整 extent，块内 slot ID 升序；
- 分数低者先淘汰；同分取块内 max(last_access) 更旧的；
- 刚被覆写的 extent 有 2 轮免淘汰保护（写入轮 t 之后，t+1 轮不可当选）；
- 全部 extent 都在保护期时按块内最旧访问强制淘汰。

## 验证

在 numba 0.60 + Python 3.12 语义下编译通过，以下检查全部通过：
初始 victim 对齐、bitmap 与 slot 列表一致、访问 extent 内任一 slot 整块钉住、
热点 extent 不进场且冷 extent 被淘汰、victim 恒为完整对齐 extent、
generation 校验、materialize 覆盖全量、reset 复原。

依赖：`numpy`、`numba`（Python ≥ 3.10）。
