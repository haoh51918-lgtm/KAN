# ⛏️ MIRAGE-KAN v5 完整真实挖掘报告

> 日期：2026-07-17 UTC  
> 状态：✅ 完整挖掘与独立审计通过；⏳ Quanta 五臂回测尚未打开  
> 证据属性：纠正性、适应性、重复开发期筛选

## 🎯 一句话结论

MIRAGE-KAN v5 已在全新协议身份下从零重跑完整挖掘链，产出 **7 个严格合格的 KAN 符号因子**。同一轮还发布了等规模的 GP/SR 控制库、标签置换控制库、匹配 MLP 黑盒控制、机制卡和匿名盲审包。七部分 topology、28 个 payload 文件和 12 条 authority receipt 已由独立代理逐项复算通过。

这说明挖掘链路再次稳定产出合格因子，但还不能说明这些因子优于 baseline。项目的主要横向问题仍需下一阶段固定 Quanta 五臂回测回答。

## 📊 完整预算与产出

| 模块 | 冻结预算 | 最终产出 | 状态 |
|---|---:|---:|---|
| 真实 KAN | 256 次尝试、每次 300 次更新 | 7 个生产因子 | ✅ |
| 标签置换 KAN | 256 次尝试、每次 300 次更新 | 7 个控制因子 | ✅ |
| typed GP/SR | 256 次尝试 | 7 个控制因子 | ✅ |
| 匹配 MLP | 每个生产因子 1 个、各 300 次更新 | 7 个控制模型 | ✅ |
| 机制卡 | 每个生产因子 1 张 | 7 张 | ✅ |
| 匿名盲审包 | 每个生产因子 1 项 | 7 项 | ✅ |

真实 KAN 和标签置换 KAN 各完成 76,800 次优化更新；匹配 MLP 共完成 2,100 次更新。生产库规模 7 位于冻结范围 6–16 内，所有三类因子库都覆盖至少 3 个 profile。

`profile` 是不同变量与窗口族组成的 KAN 搜索子空间，用于防止最终因子全部集中在同一种模式。

## 🧬 7 个生产因子

| Profile | 因子数 | 因子 ID |
|---|---:|---|
| 价格与成交量联合 | 5 | `kan_price_volume_023`、`030`、`044`、`052`、`059` |
| 反转 | 1 | `kan_reversal_060` |
| 短窗价格 | 1 | `kan_short_price_020` |

v5 是完整重跑，没有复制 v4 因子文件、模型参数或开发期结果。确定性协议再次得到相同的 7 个因子身份，这属于重跑一致性；v5 的 manifest、topology、authority 和所有 payload 哈希均为新的协议身份。

## 🛡️ 反伪阳性与完整性

| 检查 | 结果 |
|---|---:|
| 真实 KAN scoring / disposition | 256 / 256 |
| 标签置换 scoring / disposition | 256 / 256 |
| GP/SR scoring / disposition | 256 / 256 |
| 标签置换达到真实生产门槛 | 0 / 256 |
| 因子库 / MLP / 机制卡 / 盲包规模 | 全部 K = 7 |
| Profile 数 | 三类库均 ≥ 3 |
| 七个 manifests | 7 / 7 通过 |
| Payload 哈希 | 28 / 28 一致 |
| Authority receipt | 12 条连续闭合 |
| `.INCOMPLETE` / terminal failure / staging | 0 / 0 / 0 |

12 条授权链依次包含：首次标签访问 1 条、四个科学/控制臂 4 条、六个 child 发布 6 条、顶层发布 1 条。Mining entitlement、preclaim、base lock、implementation lock 与 topology 身份全部一致。

## 🔐 可审计身份

| 对象 | SHA-256 |
|---|---|
| v5 implementation lock | `03dfd21c545feb0cddfd0599bbed8ec51903f852389b4c7642386860fdedc394` |
| mining topology | `c23f1578d7c11f72a2ccd9ee139b6b9c5640975c0ae1f75b5b5cf879f0358969` |
| 顶层 mining manifest | `b0e3b0894b5e909af8ce1158cce6c54014d5a6db90df72ab3e1ef41c9f5a56dd` |
| KAN 生产库 manifest | `6245f2ba91649ba6274dda1e28c4752e6fc16229a38cb1daefe7b106e48b7361` |
| GP/SR 控制库 manifest | `b0b164a041e5c4b3b54eadb7c51cc27320b882ac83aa7ea47cc15a997e21adf6` |
| 置换控制库 manifest | `abe83f4ebd3867434bb68640c6bc6dee7118f90f51c86abfb67bf895df2abfab` |
| 匹配 MLP manifest | `389579dd3164c17c67c51be4e52c22ecf5b7f5bab22bbb67d1ba0339d0d6dabd` |

## 🔎 缩写说明

- **KAN**：Kolmogorov-Arnold Network；本项目将其学习结果硬化为可独立执行的符号因子。
- **GP/SR**：Genetic Programming / Symbolic Regression，遗传编程 / 符号回归控制方法。
- **MLP**：Multi-Layer Perceptron，多层感知机；这里只作参数量匹配黑盒对照。
- **AST**：Abstract Syntax Tree，抽象语法树，即严格可执行的因子公式。
- **Manifest**：机器清单，记录产物身份、文件集合和 SHA-256。
- **Authority receipt**：不可覆盖的授权回执，用于证明每次标签访问、实验臂和发布动作的合法顺序。

## 🧭 证据边界

- `KAN_Alpha_PR.md` 是唯一 proposal 权威；WIKI 未用于改变方法或门槛。
- v4 污染的开发期中间结果没有进入 v5 挖掘、阈值、种子或叙事。
- 当前没有 v5 development preclaim、opening、evaluation、decision 或 MLflow run。
- 人类盲审尚未完成，机制解释不能声明通过。
- v5 使用相同 2022–2025 时期只能作为纠正性的重复开发证据，不是最终确认。

## 🚀 下一步

在第二路治理审计也通过后，才消费唯一一次 v5 development opening，并按冻结顺序运行五个 Quanta 实验臂：

1. Alpha158 官方 replay；
2. MIRAGE-KAN 生产因子库；
3. typed GP/SR 控制库；
4. 匹配 MLP 黑盒控制；
5. 标签置换 KAN 控制库。

所有臂必须先在私有 staging 中完整通过日历、identity、成本和 metric 检查，才允许原子发布和组装唯一 decision。主要判断是 MIRAGE-KAN 相对 baseline 与 MLP 的配对 Information Ratio 差异，而不是因子数量、更新次数或运行时间。
