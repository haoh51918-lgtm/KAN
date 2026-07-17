# 🧪 MIRAGE-KAN 阶段报告：v3 在严格因子库门槛处终止

## ⚠️ 一句话结论

v3 完成了真实 KAN、标签置换 KAN 和 GP/SR 搜索与精确硬公式评分，但最终只有 7 个 KAN 因子通过全部严格规则，低于预注册的最小库规模 8，因此按规则终止。2022–2025 开发期没有打开，Quanta 回测没有运行。

## 📊 发生在哪里

| 项目 | 结果 |
|---|---:|
| 真实 KAN miner | 256 次已完成 |
| 标签置换 KAN miner | 256 次已完成 |
| typed GP/SR 对照 | 256 次已完成 |
| 严格入选 KAN 因子 | 7 个 |
| 原最小库规模 | 8 个 |
| profile 覆盖门槛 | 通过 |
| MLP 黑盒控制 | 未开始 |
| 开发期数据 | 未打开 |
| Quanta 回测 | 未开始 |

硬 AST 指把 KAN 学到的分类边离散成可独立执行的抽象语法树公式。GP/SR 是遗传编程与符号回归，用作同预算方法对照。失败信息明确显示 `minimum_size_met=False`、`profile_quota_met=True`，即问题只是库规模差 1，不是因子集中在单一 profile。

## 🔒 证据边界

七个 v3 目标已经全部写入 terminal failure 回执，不能删除、恢复或改名为成功产物。一次性 mining entitlement、preclaim、authority ledger、base lock 和 implementation lock 都已哈希封存。

由于 mining bundle 采用全有或全无发布，7 个因子的身份与分数没有作为可复用因子库发布。因此我们没有看到任何开发期收益、Information Ratio、RankIC 或组合结果，也不能声称这些因子优于或劣于 Alpha158。

## 🧭 为什么不直接把门槛改成 7

项目的主要横向指标是“严格因子库在固定 Quanta 框架中的真实回测质量”，因子数量只是辅助分析。让一个任意的 8 因子门槛永久阻止第一次完整回测，与这个目标不一致。

但把门槛恰好改成刚看到的 7 会显得针对结果拟合。因此下一代 v4 将使用 6 个因子的探索性最低规模：要求至少 3 个 profile，平均每个必要 profile 有 2 个因子，同时保留一定冗余。库上限仍是 16。

## ✅ v4 不会放宽什么

- 不降低 train/validation RankIC 阈值；
- 不降低覆盖率、soft-hard 保真度或 gate margin；
- 不放宽符号公式合法性或独立 AST 重放；
- 不放宽因子间相关性上限；
- 不减少 KAN、置换 KAN 或 GP/SR 的完整预算；
- 不允许手写因子、旧因子成员复用或近似实现；
- 不改变 Quanta 成本、回测、bootstrap 和性能判断标准；
- 仍然禁止在本阶段正式 promotion。

## 🛠️ 同时修复的开发运行时问题

开发期开封前审计还发现，v3 implementation lock 没有完整记录 Quanta 延迟导入的 QLib 与 LightGBM。我们在开发期开封前发现了它，因此没有污染任何回测结果。

v4 正在构建独立 Python 3.12.3 环境，精确固定 220 个包、所有 wheel SHA-256、Python 可执行文件、Torch/CUDA/cuDNN、QLib、LightGBM、MLflow、Quanta commit 和 QLib provider。不会把两个旧环境的 site-packages 拼接在一起。

## 🚀 下一步

1. 完成隔离 v4 运行时的离线安装、`pip check` 和导入 smoke；
2. 建立 v4 新协议、全新产物路径和新的 implementation lock；
3. 重跑完整 mining，因为 v3 没有发布可复用的部分因子库；
4. 若严格库规模达到至少 6，完成 MLP、机制卡、盲审包；
5. 只开一次 2022–2025，并运行五臂真实 Quanta 回测；
6. 以因子库回测质量而不是因子数量作为主要结论。

下一次飞书更新将在 v4 implementation lock 签发并准备真实重跑，或出现新的实质阻塞时发送。
