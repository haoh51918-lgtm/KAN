# 🔒 MIRAGE-KAN v4 实现锁阶段报告

> 日期：2026-07-17 UTC  
> 状态：✅ 实现闭包已锁定，可进入真实挖掘  
> 证据边界：尚未开启任何真实标签、开发期数据或 Quanta 回测

## 🎯 本阶段解决了什么

我们已经把 MIRAGE-KAN 的完整执行链冻结为一个可离线重建、可逐文件核验、锁后漂移即失败的运行闭包。`KAN_Alpha_PR.md` 仍是唯一 proposal 权威；WIKI 未参与方案或阈值决策。

| 检查项 | 最终结果 |
|---|---:|
| Python / Torch | 3.12.3 / 2.9.1+cu129 |
| CUDA / cuDNN | 12.9 / 9.10.2.21 |
| GPU | 2 × NVIDIA A800-SXM4-80GB，UUID 与显存均已锁定 |
| 离线包仓 | 223 个制品，4.88 GB；当前平台使用 217 个包 |
| 全量测试 | 368 / 368 通过 |
| 代码检查 | Ruff lint 通过；本轮 7 个文件格式检查通过 |
| KAN 兼容烟测 | 每张 GPU：64 个生产 KAN × 300 次更新 |
| MLP 兼容烟测 | 每张 GPU：6 个匹配 MLP × 300 次更新 |
| 精确重放 | AST 数值与 support mask 精确一致；MLP 公开预测 0/0 容差精确一致 |
| Qlib 跟踪 | 指向分类后的 tracking 目录；provider 未初始化 |

## 🧪 为什么这次检查比较严格

独立审计先后拦截了旧 CLI 误接、GPU/CPU 预测语义混淆、GPU 身份未绑定，以及锁后可通过 `.pth` 注入外部路径等风险。最终实现锁会实时重验：虚拟环境、`sys.path`、`pyvenv.cfg`、全部 `.pth`、217 个包的 RECORD、GPU UUID、TF32 与确定性状态、数据、Quanta、Qlib provider 和所有源码。

这里的缩写含义：

- **AST**：抽象语法树，即可执行的符号因子公式。
- **MLP**：多层感知机，本研究中只作为参数量匹配的黑盒对照，不进入生产因子库。
- **PIT**：Point-in-Time，按当时可见信息构造的数据，防止未来信息泄漏。
- **TF32**：NVIDIA 的加速数值模式；本实验关闭它以保持精确、可复现的 float64 路径。
- **RECORD**：Python 包安装清单；其哈希用于发现锁后包内容漂移。

## 🔐 已签发身份

| 锁 | SHA-256 |
|---|---|
| v4 base lock | `a90c84f2d59b147686169824903016ac2b1f3c68056bc37fd06027e20d4b51d3` |
| v4 implementation lock | `37d345c1ab5520d9e7c18d6137fad0a91109cdc9d20319162ddea170b5d95a41` |
| runtime closure summary | `bb6e2cfa9b4e2507c4e3d1adf58f468cdfa7ff1706b1bbbdbc123c9bb7d08ea5` |

Implementation lock 使用独占创建并设为只读，连续两次 live verify 的内容哈希均与签发文件完全相同。

## 🚀 下一步

立即消费一次 mining entitlement，然后按冻结预算运行真实训练/验证期挖掘：KAN、置换 KAN 与 GP/SR 并行；只有生产库达到 6–16 个严格因子、覆盖至少 3 个 profile，才训练匹配 MLP、生成机制卡并发布完整矿池。矿池全部不可变发布成功之前，2022–2025 开发期和五臂 Quanta 回测保持封闭。

主横评指标仍然是最终因子库在固定 Quanta 框架下的回测质量，而不是挖掘数量、训练轮数或耗时。
