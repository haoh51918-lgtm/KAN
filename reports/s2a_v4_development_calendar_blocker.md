# 🚨 MIRAGE-KAN v4 Development 阻断报告

> 日期：2026-07-17 UTC  
> 状态：🛑 v4 已按治理规则终止；没有发布任何回测臂或最终决策  
> 影响边界：挖掘产物仍完整有效；v4 的 KAN development 中间指标无效且禁止用于科学结论

## 🎯 发生了什么

五臂 Quanta 回测已经正式开封。Alpha158 基线臂完成暂存后，MIRAGE-KAN 臂也完成了模型训练和组合回测，但在发布前被严格日历检查拦截：`prediction_coverage.parquet` 的日期集合不等于冻结的 development 日历。

流程随即清理全部 staging，并把 Alpha158、MIRAGE-KAN、GP/SR、匹配 MLP、标签置换 KAN 和最终 decision 六个拓扑目录统一标记为 terminal failure。没有把部分成功或污染结果当成正式结果。

## 🔬 已精确定位的根因

| 检查 | Alpha158 | MIRAGE-KAN |
|---|---:|---:|
| 预测日期总数 | 966 | 2,427 |
| 正确的 2022-01-04 至 2025-12-26 日期 | 966 | 966 |
| 错误混入的 2016–2021 日期 | 0 | 1,461 |

Quanta 的官方 Alpha158 handler 能正确处理 YAML 解析出的日期区间。自定义预计算因子路径中的 `PrecomputedDataHandler.fetch` 只把 `tuple` 识别为日期区间，但 `backtest.yaml` 中的 train、valid、test 区间经 YAML 解析后实际是 `list`。因此自定义路径没有应用任何分段过滤：训练、验证和测试都读取了完整的 2016–2025 面板。

这不仅让预测覆盖文件多出 1,461 个日期，也意味着模型训练接触了 development 标签，构成实质性未来信息泄漏。v4 的 MIRAGE-KAN 中间 IC、RankIC、收益和 IR 全部作废，不会进入任何报告结论或后继比较。

## 🛡️ 为什么没有继续“凑结果”

- 没有删除多余日期后继续发布，因为模型训练本身已经泄漏，裁剪预测不能修复训练污染。
- 没有放宽 exact-calendar 门槛，因为它正是发现问题的关键防线。
- 没有重启 v4，因为 development opening 已一次性消费，且 v4 实现锁不可修改。
- 没有把 Alpha158 暂存结果单独发布，因为冻结协议要求五臂齐全后才能形成可比较结论。

## 🔧 不可变后继计划

创建新的 v5 adaptive successor，只修复日期区间表示兼容性，不改变数据日期、模型、因子、metric、回测策略或科学门槛：

1. 先写回归测试，稳定复现 YAML `list` 区间导致 train/valid/test 都返回全期数据。
2. 在受控适配层把三个冻结区间规范化为等值的二元 `tuple`，同时逐值断言日期内容完全未改变。
3. 验证 train 仅为 2016–2020、valid 仅为 2021、test 恰好为冻结的 966 个 development 交易日。
4. 验证 Alpha158 路径的输入、预测和回测语义不变。
5. 跑全量测试和独立审计，签发新的 base/implementation lock。
6. 重新运行完整受治理链路，再做五臂 Quanta 比较；不复用 v4 污染的 development 中间结果。

## 🔎 缩写说明

- **PIT**：Point-in-Time，按当时可见信息构造的数据。
- **IC**：Information Coefficient，因子值与未来收益的相关性。
- **RankIC**：每日横截面上因子排序与未来收益排序的相关性。
- **IR**：Information Ratio，信息比率，用于衡量超额收益稳定性。
- **staging**：发布前临时区；只有完整校验通过才允许原子发布。
- **terminal failure**：不可恢复的协议终止状态，防止同一 opening 被重复使用。

## 📌 当前结论

MIRAGE-KAN v4 的真实挖掘阶段仍然成功并保有 7 个严格因子；development 结论为空。当前问题是已经明确定位的 Quanta 自定义因子分段兼容缺陷，不是 MIRAGE-KAN 因子优劣的证据。后继实验必须在无泄漏条件下重新回答核心问题：该因子库是否在固定 Quanta 框架中优于 baseline。
