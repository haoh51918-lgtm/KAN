# S1 Gate A 正式科学报告

> 日期：2026-07-16 UTC  
> Proposal authority：`KAN_Alpha_PR.md`  
> 正式 run：`s1_gate_a_v1_scientific_formal_v1`  
> 结论：**有效的 scientific fail；触发 Plan C**

## 🎯 一句话结论

Symbolic-Residual KAN（E4，符号原语加样条残差）在三个随机种子上都准确恢复了正确变量和五日窗口，并显著优于自由样条 KAN、纯符号 KAN、符号回归和同参数量 MLP；但联合训练没有把“未知机制”单独隔离到样条残差中，因此无法把残差治理成一个新颖、低复杂度、可执行的 HARD 原语。Gate 1、2、3、4、7 通过，Gate 5、6 失败，完整 Gate A 判定失败。

这证明 E4 具有很强的半符号数值恢复能力，但没有证明 KAN 能稳定完成“未知原语发现 → 硬化公式”的完整链路，更没有证明真实 Alpha 盈利能力。

## 🧾 证据完整性

| 检查 | 结果 |
|---|---:|
| 正式 seeds | 1729、2718、31415 |
| 顶层 artifact index | 208 / 208 文件精确闭合 |
| Merkle aggregate | `63436d31217e92cc8b7bb196fd54954ff7d4109bea04002228f7b83d05a90a6b` |
| Arm test receipts | 24 / 24，输入 manifest 前后 SHA 一致 |
| Matrix claims | 3 / 3 |
| Selected checkpoints | 18 / 18，SHA 校验通过 |
| Seal 与 implementation | 运行前后及当前完全一致 |
| Terminal failure | 无 |
| 独立结果审计 | PASS |

`NRMSE` 是归一化均方根误差，越低越好。`HARD` 指不再依赖自由样条、能够独立执行和审计的硬化程序。`nondup` 指新原语不能只是已有原语经过输入、输出仿射变换后的重复版本。

## 📊 七项 Gate

| Gate | 结果 | 核心证据 |
|---:|:---:|---|
| 1 容量健全性 | ✅ PASS | E1 clean NRMSE 中位数 0.0803；C6 为 0.0759，均优于 0.15 门槛 |
| 2 稳定恢复 | ✅ PASS | E4 三个 seed 均选择 `Return(Close,5)`，输入贡献质量约 0.99998—0.99999 |
| 3 形状质量 | ✅ PASS | E4 shape NRMSE 中位数 0.0160；相对 E3/E5 中较好者改善 64.85% |
| 4 数值竞争力 | ✅ PASS | E4 clean NRMSE 中位数 0.0233，优于最佳数值上界 0.0759 |
| 5 可执行提升 | ❌ FAIL | 没有候选通过原语治理，未生成 governed HARD model |
| 6 解释性 Pareto | ❌ FAIL | HARD 的误差、忠实度、描述长度三轴均不可用 |
| 7 Null 安全 | ✅ PASS | 三个置换标签控制均未提升原语，null promotions = 0 |

## 📈 主要模型比较

| Arm | 角色 | Clean NRMSE 中位数 | Shape NRMSE 中位数 | 结论 |
|---|---|---:|---:|---|
| E1 | 自由三次 B 样条 KAN | 0.0803 | 0.0521 | 数值上界之一 |
| E2 | E1 后验符号化 | 0.1317 | 0.2019 | 符号化损失明显 |
| E3 | 纯 Symbolic-KAN | 0.0415 | 0.0455 | 强符号基线，但硬化忠实度有限 |
| E4 | Symbolic + Spline Residual KAN | **0.0233** | **0.0160** | 数值与形状最佳，但仅半符号 |
| E5 | Typed symbolic regression | 0.0963 | 0.0790 | 可执行但表达受冻结字典限制 |
| C6 | 361 参数 SiLU MLP | 0.0759 | 0.0769 | 同容量数值控制 |
| HARD | 治理后新原语模型 | N/A | N/A | 未生成 |

E4 的 residual-spline energy ratio 中位数仅约 0.00694，但“能量小”不等于“残差就是未知原语”。这一点正是本轮最重要的负向发现。

## 🔬 为什么 promotion 失败

### 排除实现错误

- 冻结指数 family 对精确参考曲线可恢复参数 `[0.6, 2.5, 1.8, 0.25]`，NRMSE 为 0。
- 四个冻结起点都收敛；bounds、方程与 SciPy 优化器正常。
- 正确参考候选通过 nondup：最佳 affine-Tanh 的 Pearson 虽为 0.995368，但 NRMSE 为 0.096137，高于重复门槛 0.05。
- 落盘 residual 与 checkpoint 重载输出的最大误差为 `1.11e-16`。

### 真正断点：加性分解不可辨识

联合训练学到的是：

```text
软解析原语混合 + 非单调样条补偿 = 高精度总函数
```

而不是：

```text
已知解析部分 + 与隐藏机制同形的样条残差
```

| Seed | E4 总函数 shape NRMSE | 仅软解析部分 NRMSE | 样条与“所需修正”相关 |
|---:|---:|---:|---:|
| 1729 | 0.015990 | 0.26731 | 0.99801 |
| 2718 | 0.014267 | 0.19246 | 0.99718 |
| 31415 | 0.016125 | 0.22627 | 0.99801 |

三个 residual 都是“两端为负、中心为正”的稳定补偿曲线，约一半区间递增、一半递减；它们不属于冻结的单调、零点锚定饱和 family。三个 family 对 residual 的中位 NRMSE 均约 0.669，拟合后退化为与 `Clip(-1,1)` 重复的曲线，因此同时失败 `low_complexity_approximation` 与 `non_duplication`。

这属于架构假设失败，不满足 prereg 中“reference sanity 失败才允许 implementation recovery”的条件。不得通过放宽门槛、换起点或重复打开 test 来挽救结论。

## 🧭 研究决策

按照 proposal 第 21.2、21.4、22.4 节：

1. 封存 Gate A 为 `scientific_fail`。
2. 激活 Plan C：KAN 从必要骨架降为可选矿机实例，不再把“可执行原语提升”作为 S2 的前置依赖。
3. S2 优先搭建真实异构因子库：typed GP / symbolic miner 为主，MLP / semi-symbolic KAN 为辅助，统一进入现有 Quanta evaluator。
4. 先用真实因子库回测回答用户的首要问题：相对冻结 baseline 的净 Information Ratio 是否提升，并同时审计最大回撤、换手成本、RankIC、覆盖率和稳定性。
5. 图模块继续锁定。只有 S2 证明因子库价值、S3 证明矿机冗余问题后，才进入 Gate B，并与 Flat-Controller 和 Bandit-Budget 同台比较。
6. 可选 S1b 只作为独立新实验：研究“完整 selected edge promotion”或 hard-first / sequential decomposition；必须重新预注册并使用新的 test seeds，且不得阻塞 S2。

## ⚠️ 治理事故披露

正式运行前曾有一次红测错误，导致 seed 1729 在 pytest 临时目录生成数据并开始部分 E1 训练。该尝试在 test opening 前被中断，没有 claim、prediction、test metric、Gate 或顶层 manifest，已被裁定为无效的 pre-test partial attempt。后续 pytest 自动清理了临时目录；原始八文件哈希清单与 custody addendum 均被 implementation lock 绑定。本次唯一正式 one-shot opening 经独立审计仍然有效。

## 🔗 关键产物

- 正式 matrix：`artifacts/s1_gate_a_scientific/s1_gate_a_v1_scientific_formal_v1/manifests/matrix.json`
- Gate 报告：`artifacts/s1_gate_a_scientific/s1_gate_a_v1_scientific_formal_v1/reports/gate_a/gate_a.json`
- Promotion 审计：`artifacts/s1_gate_a_scientific/s1_gate_a_v1_scientific_formal_v1/models/promotion/manifest.json`
- Matrix SHA：`a37552836253577b49d4e39900e9a9be42579fd7987af60a674b4cdcfd9a1f44`
- Gate report SHA：`9bd1e3235875040f30d9dbb43dfa51fe39c54c6828b87100d7333874ffe67328`
- Promotion SHA：`b946f42ae6c679c237fbad303c5c1692068492a817e6582f370db98fc841e460`

