# 🧭 MIRAGE-KAN 研究启动检查点

## ✅ 当前结论

我们不会机械照搬 proposal 的 Phase 0→6，而是按“什么证据允许做下一步”推进。

| 顺序 | 要回答的问题 | 通过后才能做什么 |
|---|---|---|
| S0 真实纵向链路 | PIT 数据到因子库、LightGBM、真实组合回测是否完整可信？ | 允许评价项目自产因子库 |
| S1 KAN 快速证伪 | KAN 的样条残差是否真的补足纯符号字典？ | 允许把 KAN 保留为主角 |
| S2 因子库价值 | 单矿机挖出的库是否比匹配 baseline 回测更好？ | 允许扩大到多矿机 |
| S3 冗余诊断 | 多矿机是否真的重复挖掘，简单异质化能否解决？ | 允许引入图反馈 |
| S4–S5 图控闭环 | 图是否优于 boosting、bandit 和同信息无图控制器？ | 才能声称 MIRAGE 图控有效 |

## 🎯 主指标

主横屏仍是固定 QuantaAlpha 协议下、扣交易成本后的组合 Information Ratio（信息比率）。因子数、训练轮数和耗时只用于解释，不替代回测质量。

## 🧪 五个优先攻击点

1. KAN 可能只是品牌装饰：用字典外机制 E1–E5 同预算比较。
2. 增益可能来自 LightGBM、因子数或样本支持：统一 joint support、公共因子上限和 500 轮/早停 50。
3. 图可能只是 boosting 或预算分配包装：必须比较 Boost-Sequential、Bandit-Budget、Flat-Controller。
4. 结果可能是泄漏或赢家偏差：分离 membership/observed/tradable，记录全部尝试并跑完整标签置换。
5. 公式可能可读但不忠实：检查软硬一致、变量删除、滞后屏蔽和函数形状。

## ♻️ Baseline 复用

已确认 Alpha158 锚点的第 14 轮是 500 轮上限下 early stopping 的自然结果，不是人为 fixed14。首轮直接复用其 hash-bound 结果，不重复跑完整 baseline；只有新预计算因子适配器 parity 失败才做最小复现。

## 🔧 基础设施

飞书未续跑的原因是旧长连接桥已经退出，收件记录停在 10:55。凭证和线程绑定正常，现已使用当前 EvoSci 版本恢复桥接并重新激活当前 session。

## 🚀 下一步

进入 More Effort 的 S0：先写行为测试，再实现 PIT→Typed AST→不可变因子库→Quanta 回测适配器；每轮都运行测试并追加 iteration log。与此同时为 S1 预留硬化接口，避免 S0 完成后返工。
