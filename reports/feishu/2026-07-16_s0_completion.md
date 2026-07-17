# 🔗 MIRAGE-KAN S0 完整链路报告

## ✅ 结论

第一条真实纵向链路已经跑通，不是 mock、代理回测或采样近似：

PIT OHLCV → Typed AST → 不可变因子库 → Quanta 预计算因子入口 → LightGBM → 成本组合回测

| 验收项 | 结果 |
|---|---:|
| 行为测试 | 11 / 11 通过 |
| PIT 数据行 | 1,572,483 |
| 动态成分行 | 801,001 |
| 成分内但 OHLCV 全缺失 | 15,394，原样保留为不可观测 |
| 1 日 / 20 日标签逐值误差 | 0 / 0 |
| 发布控制因子 | 4 |
| LightGBM 协议 | 最多 500 轮，早停 50 |
| 真实组合回测 | 966 个交易步，Top50 / Drop5，含成本 |

## 📊 链路观测

| 指标 | 数值 |
|---|---:|
| IC | 0.042781 |
| RankIC | 0.041031 |
| 净 Information Ratio | 0.528703 |
| 年化超额 | 0.042857 |
| 最大回撤 | -0.155985 |

⚠️ 这些数值不能解释成 MIRAGE-KAN 已经击败 baseline。当前 4 个公式是手工固定的 wiring control，只验证接口；manifest 已写死 `kan_mined=false` 与 `scientific_result=false`。

## 🧾 可审计证据

- `factor_libraries/seed_ast_v1/`：AST、完整 panel、support 和全部身份哈希。
- `evaluations/s0_vertical_slice/`：真实 Quanta 指标与 966 日累计超额序列。
- `evaluations/runtime/seed_ast_v1/`：Qlib/MLflow 训练损失、预测、标签与 IC 序列。
- `audits/s0_real_data_audit.json`：数据、标签和掩码审计。
- `artifacts/iteration_log.md`：4 次 More Effort 迭代，最终得分 0.96。

## 🔧 过程中解决的问题

1. GPFS 不支持 `renameat2(RENAME_NOREPLACE)`：没有降级成可覆盖 rename，改用独占建目录 + O_EXCL + manifest 最后提交 + fsync。
2. 新版 MLflow 默认拒绝 filesystem tracking：按异常要求显式开启 `MLFLOW_ALLOW_FILE_STORE=true` 后，未修改模型/数据/回测语义即成功。
3. 根目录运行缓存已清理，MLflow 记录归档到 `evaluations/runtime/`，并补充 `.gitignore`。

## 🚀 下一步

进入 S1 的 KAN 最小证伪，而不是直接做图：

1. 先补 Torch 批执行器与 Pandas↔Torch 精确 parity。
2. 冻结字典外机制 E1–E5 的阈值、预算与失败条件。
3. 比较纯符号、MLP、自由样条 KAN、后验符号化与 Symbolic-Residual KAN。
4. 输出必须继续走本次已经验证的同一因子库与回测链路。

图控仍锁定；只有 KAN 必要性和真实单矿机因子库价值都成立，才进入多矿机与 MIRAGE 闭环。
