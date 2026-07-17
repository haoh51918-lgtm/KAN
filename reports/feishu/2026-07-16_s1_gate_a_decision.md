# 🧪 MIRAGE-KAN S1 Gate A 正式结果

## 🎯 结论先行

正式实验有效完成，但科学判定是 **scientific fail**。这不是“模型完全没学会”：E4 在三个 seed 上都准确找到了 `Return(Close,5)`，总函数形状和测试误差都明显优于其他方法；真正失败的是最后一步——没有把自由样条残差升级成一个新颖、低复杂度、可独立执行的 HARD 原语。

因此按 proposal 启动 **Plan C**：KAN 降为可选矿机，下一阶段优先做真实因子库和 Quanta 回测，图模块继续锁定。

## 📊 七项 Gate

| Gate | 结果 | 直观解释 |
|---:|:---:|---|
| 1 容量健全性 | ✅ | 自由样条 KAN 和同参数 MLP 都能学会任务 |
| 2 稳定恢复 | ✅ | 三个 seed 都找到正确五日收益输入 |
| 3 形状质量 | ✅ | E4 shape NRMSE 中位数 0.0160，比 E3/E5 较好者改善 64.85% |
| 4 数值竞争力 | ✅ | E4 clean NRMSE 中位数 0.0233，明显优于最佳数值控制 0.0759 |
| 5 可执行提升 | ❌ | 没有合格 HARD 原语 |
| 6 解释性 Pareto | ❌ | HARD 不存在，无法比较误差、忠实度和复杂度 |
| 7 Null 安全 | ✅ | 置换标签控制没有伪提升 |

`NRMSE` 是归一化误差，越低越好；`HARD` 是完全关闭自由样条后仍可独立执行的公式程序。

## 🔬 失败原因

独立诊断排除了 fitter、bounds、起点和 nondup 审计错误。精确指数参考曲线可以被原样恢复，NRMSE 为 0。

E4 实际学到的是：

**软解析原语混合 + 非单调样条补偿 = 高精度总函数**

样条残差本身不是隐藏的单调饱和机制。三个 seed 都出现同一现象，所以这是冻结架构的分解不可辨识，不是简单调参能修好的问题。

## 🧭 下一步

1. 封存 Gate A，不改阈值、不重复打开 test。
2. 进入 Plan C：typed GP / symbolic miner 做可执行主路径，MLP / semi-symbolic KAN 做辅助矿机。
3. 立即冻结 S2 的真实因子库价值协议，然后用同一 Quanta evaluator 比较净 Information Ratio、最大回撤、换手成本、RankIC、覆盖率和稳定性。
4. 只有 S2 证明因子库有价值、S3 证明存在矿机冗余问题，才解锁图控 Gate B。
5. “完整 selected edge promotion”可作为独立 S1b，但必须新 prereg、新 seeds，且不能拖慢 S2。

## 🛡️ 完整性

- 208 个正式文件全部进入 Merkle 索引。
- 24 份 arm receipt、18 个 checkpoint、3 个 matrix claim 全部闭合。
- 独立审计确认 Gate 重算与报告一致。
- 此结果不构成真实 Alpha 盈利证据。

详细报告：`reports/s1_gate_a_scientific_report.md`

