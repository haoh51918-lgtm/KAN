# 🔐 MIRAGE-KAN v5 纠正性后继：实现锁检查点

> 日期：2026-07-17 UTC  
> 状态：✅ 独立审计通过，可进入完整真实挖掘  
> 证据属性：纠正性、适应性、重复开发期筛选；不是新的前瞻一次性确认

## 🎯 一句话结论

v4 的问题已经被精确修复在数据集分段边界，而不是通过裁剪预测、放宽日历或掩盖结果绕过。v5 重新冻结了 proposal、科学配置、源码、运行环境、数据、Quanta、Qlib provider、GPU 和全部 v4 隔离证据。两层锁均已独占写入并连续两次活体验证，独立审计结论为 **PASS**。

当前尚未产生任何 v5 因子、回测、MLflow run 或开发期结果。下一步是从零完整重跑挖掘链路。

## 🧯 v4 根因与 v5 精确修复

| 项目 | v4 错误行为 | v5 行为 |
|---|---|---|
| YAML 日期区间 | 解析为 list，但自定义 Quanta handler 只识别 tuple | 构造自定义数据集时临时转换为等值 tuple |
| train / valid / test | 三段都错误返回 2016–2025 全期 | 三段各自只返回冻结日期 |
| 配置状态 | 分段类型不兼容 | 数据集构造后在 finally 中恢复原配置 |
| Alpha158 官方路径 | 正常 | 完全不经过这项修复，保持不变 |
| prediction coverage | 严格检查发现污染 | 仍要求原始预测信号精确匹配 966 个开发交易日 |

真实 pinned Quanta seam 的合成测试分别准备了一个 train 日、一个 valid 日和一个 test 日；三个 `dataset.prepare` 调用只返回各自日期，配置成功恢复，Qlib provider 未初始化，也没有读取真实标签。

## 🧊 科学设计没有改变

v4 与 v5 的 YAML 在移除协议编号、21 个可写产物路径和 evidence class 后严格相等。以下项目全部保持冻结：

- 2016–2020 训练期、2021 验证期、2022–2025 重复开发期；
- KAN、标签置换 KAN、GP/SR 各 256 次完整尝试；
- 每个 KAN 300 次 float64 更新；
- 6–16 个生产因子、至少 3 个 profile；
- Alpha158、MIRAGE-KAN、GP/SR、匹配 MLP、标签置换 KAN 五臂顺序；
- Quanta commit、LightGBM、TopkDropout、交易成本、metric、bootstrap 与全部决策门槛。

v5 不复用 v4 的因子成员或任何污染的开发期输出，而是在新身份下重跑完整挖掘拓扑。

## 🔒 新冻结链

| 对象 | 结果 |
|---|---|
| v5 base lock | `aa36b6ae9cfae850284f3c8bbac83c401981787dbfb9c3b9f7e4acf362192983` |
| v5 implementation lock | `03dfd21c545feb0cddfd0599bbed8ec51903f852389b4c7642386860fdedc394` |
| MIRAGE-KAN 源码树 | `5b5dae638311fc69bc6ef1fd23f0f084a982f05625a17d37a125d99869d67865` |
| Qlib provider 树 | `1babf2a6ac4643df141c46f410ce3bb1f3b51ea6570b4ebe4d02658f01bcdca1` |
| v4 前序保管文件 | 160 个，全部逐文件 SHA-256 |
| v5 运行时文件 | 258 个，全部逐文件 SHA-256 |
| 固定文件集合 | 426 / 426 与实现锁一致 |

160 个前序文件包括 v4 的 15 份 authority receipt、15 个 claim、完整 ledger、两个 arm consumption、六个 terminal failure、七部分挖掘产物、两棵隔离 MLflow run、三份阶段报告以及更早 v2/v3 保管链。隔离 tracking 只作字节级故障证据，禁止注册为 v5 科学输入。

## 🧪 验证与环境

| 检查 | 结果 |
|---|---:|
| 全量 pytest | 372 passed |
| 全量回归警告 | 933，均未导致失败 |
| 当前 `src tests runtime/s2a_v5_eval/tools` Ruff | ✅ 通过 |
| Base lock 确定性构建 | 连续两次摘要完全相同 |
| Base / implementation live verify | 各连续两次通过 |
| GPU | 2 × NVIDIA A800-SXM4-80GB |
| Torch / CUDA / cuDNN | 2.9.1+cu129 / 12.9 / 9.10.2.21 |
| v5 tracking | 只有 README，无 mlruns 或数据库 |

全量 372 项回归先于 base-lock builder 文件完成；builder 随后单独通过 Ruff、实际完成独占写锁和确定性双验证，并被 base lock 与 implementation lock 逐文件绑定。报告不把原 regression evidence 扩大解释为覆盖 builder。

## 🛡️ 证据边界

- `KAN_Alpha_PR.md` 仍是唯一 proposal 权威，WIKI 只是可信度存疑的参考。
- v4 开发期 opening 已消费，污染中间结果永久作废，不进入 v5 阈值、种子或叙事。
- v5 使用同一 2022–2025 时期只能形成纠正性的重复开发证据，不能称为前瞻一次性确认或正式晋级。
- 即使 v5 全部自动门槛通过，仍需要人类盲审与真正未见时期或市场完成确认。

## 🔎 缩写说明

- **KAN**：Kolmogorov-Arnold Network，本项目把其学习结果硬化为可执行符号因子。
- **GP/SR**：Genetic Programming / Symbolic Regression，遗传编程 / 符号回归对照。
- **MLP**：Multi-Layer Perceptron，多层感知机；这里只作容量匹配黑盒对照。
- **PIT**：Point-in-Time，严格按当时可见信息构造数据，防止未来泄漏。
- **Qlib provider**：Qlib 的行情与特征文件树；整棵树被内容哈希锁定。
- **Implementation lock**：实现锁；绑定源码、依赖、数据、运行设备和外部框架身份，任一漂移都会拒绝实验。

## 🚀 下一步

消费 v5 唯一 mining entitlement，在两张 A800 上并行运行真实 KAN 与标签置换 KAN，CPU 同时运行 typed GP/SR。只有完整发布 KAN 因子库、两类控制库、匹配 MLP、机制卡和盲审包，并通过独立挖掘审计后，才会打开五臂 Quanta 重复开发回测。

主要横向判断仍是因子库在冻结 Quanta 框架下的扣成本组合表现，而不是挖掘因子数量、训练轮数或运行时间。
