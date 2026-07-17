# 🔐 MIRAGE-KAN v6 预标签锁定检查点

## 📌 一句话结论

v6 的科学设计与 v5 完全一致，血缘纠偏已通过真实 v5 清单验证，两层只读锁和两路独立预标签审计均已 PASS。当前没有消费任何 v6 opening，也没有访问真实科学标签，可以安全进入完整挖掘。

## 🧪 锁前验证

| 检查 | 结果 |
|---|---|
| 全量 Pytest | ✅ 377 passed，0 failed |
| Ruff 静态检查 | ✅ All checks passed |
| Quanta 日历 smoke | ✅ train、valid、test 严格分离，配置在 finally 中恢复 |
| 真实 v5 血缘 smoke | ✅ 7/7 映射接受，交叉错配拒绝 |
| 真实标签访问 | ✅ 无 |
| v6 MLflow run | ✅ 无，tracking 目录只有说明文件 |

933 条测试 warning 均为已有的 pandas FutureWarning，集中在 DSL 支持掩码的未来类型转换提示，不影响本次测试结论。

## 🔒 两层不可变锁

| 锁 | SHA-256 | 绑定范围 |
|---|---|---|
| Base lock | `6937ca80885f2dda8467360a45e4d38a664dbd3d5e4588526f5cdd7ba6b74114` | 312 份前序隔离文件，260 份当前运行闭包文件 |
| Implementation lock | `05fc2f7ea02d98966cbd2e665322a911505fbadd2702d67183d72bcd6558ef2d` | 580 份固定文件、52 个源码文件、217 个 Python distributions、完整 provider |

两份锁权限均为 0444，即只读。Implementation lock 绑定的源码树 SHA-256 为 `0e968b7c89c6d5d3f9b22eab9a3aeb5c44a7dde304594fca595ce9c9c46457fc`；Qlib provider 共 60,168 个文件，树 SHA-256 为 `1babf2a6ac4643df141c46f410ce3bb1f3b51ea6570b4ebe4d02658f01bcdca1`。两次实时重建验证输出逐字节一致。

## 🧬 血缘纠偏的严格含义

新的门禁比较完整的 `KAN 因子 ID → 全局尝试序号` 映射，不再错误要求 MLP 选择顺序等于规范字典顺序。预测列、训练回执、300 步轨迹、随机种子和 bootstrap 权重仍保留原始位置绑定，没有排序或近似处理。顶层发布器也已补强：即使 ID 集合和序号集合分别相同，只要一一对应关系被交叉，就会拒绝发布。

## 🛡️ 两路独立审计

| 审计 | 结论 | 关键核查 |
|---|---|---|
| Audit A | ✅ PASS | 配置等价、21 个可写路径零重叠、v5 完整 custody、两锁 live verify、v6 路径空置 |
| Audit B | ✅ PASS | 580 文件逐项哈希、CUDA/cuDNN/双 A800 确认、真实映射对抗测试、tracking preimport、无标签访问 |

v5 的 17 份 receipts、17 份 claims、4 个 development arm consumption、6 个只读失败终态和 3 个隔离 MLflow run 已全部纳入 custody。v6 不复用 v5 因子成员或开发指标，将在自己的身份下重新运行完整确定性挖掘拓扑。

## 🚀 下一步

消费 v6 mining entitlement 后，使用两张 A800 并行运行 256 个生产 KAN miner，并完成 256 个 typed GP/SR、256 个标签置换 KAN、匹配 MLP、机制卡和盲审包。七个挖掘位点全部不可变发布并通过双路审计后，才会打开一次性的 966 交易日 development。
