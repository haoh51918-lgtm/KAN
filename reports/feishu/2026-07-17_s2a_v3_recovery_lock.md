# 🧭 MIRAGE-KAN 阶段报告：v3 已恢复到可安全运行状态

## ✅ 一句话结论

v2 的失败已被完整封存，纠正性继任协议 v3 已通过全部锁前审计并签发只读 implementation lock。当前可以开始真实因子挖掘，且没有读取或继承任何 v2 科学结果。

## 🔍 为什么需要 v3

v2 在第一次标签访问之前，因为 YAML 日期被解析为日期对象而无法写入 JSON 凭证，随后按治理规则终止。它没有加载标签、没有训练 KAN、没有产生候选因子，也没有运行回测。

v3 只修复基础设施边界，不改变科学设计：日期统一转成 ISO 格式，例如 2021-12-31；完整 JSON 先编码成功，再创建只写一次的凭证文件；所有 v3 输出路径与 v2 严格隔离；Quanta recorder 使用 v3 协议名，避免结果混写。

## 🧪 锁前验证

| 检查项 | 结果 |
|---|---:|
| 完整测试 | 314 / 314 通过 |
| Ruff 代码规范检查 | 全部通过 |
| v2 与 v3 科学配置 | 移除协议名和输出路径后完全相等 |
| v2 与 v3 可写路径 | 完全不相交 |
| v2 失败证据实时复核 | 15 / 15 哈希匹配 |
| 实际 YAML 日期回归测试 | 通过 |
| 失败序列化不遗留空文件 | 通过 |

其中，KAN 是 Kolmogorov-Arnold Network，本项目用它学习类型化时序原子之间的可微分类边；GP/SR 是遗传编程与符号回归，用作同预算方法对照；PIT 是 point-in-time，表示严格按当时可获得信息构造数据，防止未来信息泄漏。

## 🔐 新冻结链

| 冻结对象 | 身份 |
|---|---|
| v3 base lock | `07faa9b04368…bb3567` |
| v3 implementation lock | `cce2660f56eb…2ff235` |
| 固定文件 | 23 个 |
| MIRAGE-KAN 源文件 | 52 个 |
| QLib 数据文件 | 60,168 个 |
| QLib provider tree | `1babf2a6ac46…bcdca1` |
| Quanta commit | `b7ceb27b1001…8ab23` |

确定性运行已锁定：PyTorch deterministic algorithms 开启、cuDNN deterministic 开启、cuDNN benchmark 关闭、CUBLAS 工作区配置固定。

## 🛡️ v2 证据如何保护

v2 的 base lock、implementation lock、失败说明、预占据凭证、零字节失败凭证、authority ledger、单次 authority receipt 与 claim，以及七个终止产物回执，合计 15 个文件都进入 v3 实现闭包。

AuthorityGuard 会在首次标签访问、每个科学或控制臂、每次产物发布、开发期开封和最终决策发布前重新散列这些文件。任何一项被修改，后续科学回执与发布会立即停止。

## 🚀 下一阶段

马上启动真实 v3 mining，优先跑通完整链路：

1. 消费一次性 mining entitlement，并确认真实日期凭证成功落盘；
2. 双 GPU 并行运行 256 个真实 KAN miner 与 256 个完整标签置换 KAN miner；
3. CPU 并行运行 256 次同 DSL、同预算 GP/SR 对照；
4. 对入选 KAN 因子训练配对的两单元 SiLU MLP 黑盒对照；
5. 发布因子库、机制卡和盲审包后，只开一次 2022–2025 开发期；
6. 使用固定 Quanta 框架跑五臂回测，并以因子库回测质量作为主要横向指标。

下一次飞书更新会在真实 mining 产物完成，或出现需要人类决策的实质阻塞时发送。
