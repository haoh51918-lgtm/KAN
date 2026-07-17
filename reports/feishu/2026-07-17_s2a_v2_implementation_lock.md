# 🔐 MIRAGE-KAN 阶段检查点：完整链路已锁定，真实挖掘即将启动

## ✅ 一句话结论

MIRAGE-KAN S2a v2 的 KAN 挖掘、三类对照、因子库发布、五臂 Quanta 回测和机器判决已经形成可执行完整链路，并通过两轮独立审计与确定性环境下的 304 项测试。Implementation lock（实现锁）现已正式签发；从这一刻起，源码、配置、数据、Quanta、Qlib provider、Python/依赖和 GPU 运行时发生任何变化，实验都会失败关闭。

当前仍然没有 v2 科学结果：真实 train/validation 标签尚未打开，2022–2025 development 也尚未打开。

## 🧭 这次锁定了什么

| 层级 | 已锁内容 | 作用 |
|---|---|---|
| Proposal | `KAN_Alpha_PR.md` 唯一权威 | WIKI 不进入协议依据 |
| 数据 | PIT cache、标签定义、60 日 raw-only warm-up | 防止滚动特征从 split 起点错误重置 |
| 标签 | train/validation 各自末两交易日物理 purge | `fwd` 使用未来两日价格，禁止跨 split 泄漏 |
| KAN | 4 profiles × 64 miners × 300 updates | 共 256 个真实 KAN 尝试，固定预算与 seed |
| 对照 | 256 次 typed GP/SR、完整标签置换 KAN、容量匹配 MLP | 对照不能进入生产因子库 |
| 回测 | Alpha158 + KAN + GP/SR + MLP + permutation 五臂 | 统一 Quanta、LightGBM、成本、日历和指标 |
| 运行环境 | Python 3.12.3、Torch 2.9.1、CUDA 12.9、两张 A800 | 运行时漂移直接拒绝执行 |
| 确定性 | PyTorch deterministic、cuDNN deterministic、CUBLAS 固定 workspace | 固定 seed 不再只是名义上的固定 |

`profile` 是不同变量/窗口族的 KAN 搜索子空间；`typed GP/SR` 是带类型约束的遗传编程/符号回归离散搜索对照；`MLP` 是多层感知机黑盒容量对照。

## 🛡️ 最后审计实际拦下的问题

这轮审计不是简单检查格式，而是修复了数个会令真实链路失真或直接失败的问题：

- Mining 阶段禁止提前读取或发布 2022–2025 原始行情；开发期因子只能在 development opening 后从冻结 AST 或 MLP 参数独立重放。
- 训练前必须恰好保留最后 60 个交易日原始数据作为滚动特征 warm-up；这些日期的标签保持为空，也不能进入 bootstrap。
- `fwd` 的两日价格跨度不能越过 train/validation 边界；边界标签在 Parquet 读取层就被排除。
- Decision assembler 现在区分“达到效力阈值的候选”和“经过容量上限、多样性约束后最终选中的因子”，不会把二者混为一谈。
- Decision 文件在组装、发布前和发布后都会重新核验；中间篡改、增删文件或运行环境漂移都会终止整套 topology。
- Proposal authority 漂移在 development、decision assembly 和 final publication 三条路径上统一记为 `superseded_authority`，不会被通用错误分类覆盖。

## 🧪 验证结果

| 检查 | 结果 |
|---|---:|
| 全量测试 | 304 passed |
| Ruff 静态检查 | 全部通过 |
| Qlib provider 文件 | 60,168 个全部纳入内容哈希 |
| Qlib provider tree SHA-256 | `1babf2a6ac46…f01bcdca1` |
| Implementation lock SHA-256 | `f22e592379f8…778863bd` |
| Proposal SHA-256 | `1ceb575843c0…ac5218` |
| Quanta commit | `b7ceb27b1001…f8ab23` |

## 🚀 下一步

接下来将消耗唯一一次 v2 mining entitlement：两张 A800 并行运行 8 个 KAN profile 作业（4 个真实标签、4 个标签置换），CPU 同时运行 256 次 typed GP/SR；随后完成共同评分、8–16 因子去相关选库、配对 MLP、机制卡和匿名盲评包。

只有所有 mining/control artifacts 不可变发布成功后，才会一次性打开 2022–2025 development，运行五臂 Quanta 回测。主横评仍是因子库带来的扣成本组合质量，候选数、轮数和耗时只作为过程分析，不替代回测结论。

图控制模块继续锁定；在真实人类盲评完成前，KAN 主库最多称为“计算透明的候选因子库”，不会提前宣称正式可解释。
