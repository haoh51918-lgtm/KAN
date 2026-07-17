# 🧭 MIRAGE-KAN v6 挖掘完成与开放前演练报告

## 📌 一句话结论

v6 的七部分挖掘拓扑已经完整、原子化发布，第一路独立审计检查 282 项并通过；但新增的真实无标签开放前演练发现评估加载器缺少一次精确索引投影。该问题已在任何开发标签访问前被截获，因此 v6 挖掘证据保持有效，开发集盲性没有受损。主线将冻结 v6，并通过 v7 的正式 verify-and-rebind（校验并重绑定）协议复用这套确定性挖掘拓扑。

## ✅ v6 挖掘交付

| 项目 | 结果 |
|---|---|
| KAN 候选尝试 | ✅ 256 / 256 |
| Typed GP/SR 对照尝试 | ✅ 256 / 256 |
| 标签置换 KAN 尝试 | ✅ 256 / 256 |
| KAN 因子库 | ✅ 7 个 |
| GP/SR 因子库 | ✅ 7 个 |
| 置换因子库 | ✅ 7 个 |
| 匹配 MLP 黑盒控制 | ✅ 7 个，全部保留 300 步轨迹 |
| 置换假阳性 | ✅ 0 / 256 |
| 七个拓扑目标 | ✅ 无 `.INCOMPLETE`，无 terminal failure |
| 顶层 manifest SHA-256 | `b02b71436bad1cb1d8872d3ea6299bbd8925f768be2df2e9c9ecd35399f1d4af` |
| 拓扑 SHA-256 | `5194c5c9193620c2948d6b0cfce23e63707e5840724203a4238e0e0886327668` |

这里的 KAN 是 Kolmogorov–Arnold Network；GP/SR 是 Genetic Programming / Symbolic Regression，即遗传编程 / 符号回归；MLP 是 Multi-Layer Perceptron，即多层感知机。MLP 只作为不可晋升的反证控制，不是可发布因子库。

## 🛡️ 独立审计

第一路只读审计核查了 282 项：12 份 authority receipts / claims、全部 manifest 与 payload 哈希、顶层和子清单绑定、预算、库规模、KAN–MLP 一一配对以及置换假阳性。第二路对抗式审计独立重算拓扑，精确重放 3 个因子库的全部 AST，并核查 512 个 KAN miner、155,136 个 tensor references、90,599,424 个连续 tensor bytes、全部 MLP receipts 和 300 步轨迹。两路结论均为 PASS，未发现软链接、越界路径、残留占位、权限异常或反馈文档回写污染。

## 🧯 开放前演练截获的问题

真实 PIT（Point-in-Time，严格时点）数据不是完整的“日期 × 股票”矩形：部分日期和股票组合在源数据中不存在。原子面板为了批量计算会形成完整笛卡尔网格，MLP 发布端随后把结果精确投影回原始 PIT 索引；加载器在校验 checkpoint 时漏做了同一个投影。

| 证据 | 数值 / 结论 |
|---|---|
| v6 mining 原始 PIT 行数 | 868,920 |
| 原子笛卡尔网格行数 | 964,314 |
| 网格额外组合 | 95,394 |
| 原始 PIT 不在网格中的行 | 0 |
| 失败类型 | 仅 shape 不一致，不是参数或预测数值失配 |
| 开发标签访问 | ✅ 0 |
| v6 development opening | ✅ 未消费 |

只读 monkeypatch 复核在真实数据上把 checkpoint 重放结果按原始 PIT 索引精确 `reindex` 后，7 个控制全部通过；扩展到完整无标签面板时 1,572,483 行索引逐项一致。该操作不会生成缺失样本、不会插值、不会改变值，也不会使用近似容差。

## 🧪 红灯测试与最小修复

新增生产 seam 回归测试使用缺少一个“日期 × 股票”组合的 ragged PIT 面板。修复前稳定复现 `123` 对 `124` 的 shape 失败；加载器只增加一次 `.reindex(prediction.index)` 后通过。额外负测故意改变一个有限 checkpoint 重放值，零容差校验仍会拒绝，因此索引修复没有掩盖数值篡改。

当前聚焦回归为 5 passed。既有“预测越过 validation 末日”“缺失或额外 artifact 行”“manifest / payload 哈希变化”门禁继续保留。

## 🔁 为什么转为 v7，而不是修改 v6

v6 implementation lock 已冻结且 mining entitlement 已消费，不能追溯修改其软件身份。v7 将建立正式跨协议重绑定凭证：实时重哈希 v6 顶层和全部子产物，复核原始 entitlement、AST、MLP 参数与轨迹、种子、bootstrap 和精确重放，并绑定新的 v7 base lock 与 implementation lock。禁止重选、重排、调参或伪造新的 label-mining entitlement。

## 🚦下一阶段门禁

1. 完成第二路 v6 挖掘审计。
2. 为跨协议重绑定 schema 先写 fail-closed 测试，再实现最小验证器。
3. 建立 v7 base / implementation locks 和独占 rebind receipt。
4. 运行全量测试、完整合成 shadow 演练、真实无标签加载演练和两路独立预开放审计。
5. 仅当以上全部通过，才消费 v7 development opening 并运行 Quanta 因子库回测。

## 🧠 路线更新

本次事件直接采纳 `Review-from-claude.md` 的关键建议：数据无关验证和完整演练必须前置到 opening 之前。它成功避免了再次烧掉一次性开发集访问权，因此该流程已经写入 Living Manual，并成为后续 MIRAGE-KAN 的硬门禁。
