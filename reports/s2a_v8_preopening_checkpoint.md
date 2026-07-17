# 🧭 MIRAGE-KAN v8 开标签前综合检查点

**时间：** 2026-07-17 UTC  
**阶段：** S2a 开发集五臂回测前  
**结论：** ✅ 所有开标签前门禁通过，可以执行唯一的正式 development 入口

## 🎯 这一步解决了什么

v6 已产出完整且可信的挖掘拓扑，但真实无标签彩排发现 matched MLP（匹配黑盒多层感知机）在稀疏股票交易日面板上的索引口径与原始 PIT 面板不一致。v7 随后又在生成重绑定回执时暴露了权威回执哈希算法不一致。两次问题都在开发集 opening 之前被拦截，因此没有消耗开发标签，也没有把异常指标当作科学结果。

v8 完成了两项最小修复：

- matched MLP 回放严格重新对齐原始 PIT 索引，并保留零容差 checkpoint 校验。
- 权威回执按原实现的规范字节格式重算：带缩进、键排序、末尾换行。

同时补齐了锁内 v8 正式启动入口，避免复用写死 v6 目录的旧启动器；并消除了最终 decision 与 manifest 中三处残留的 graph 硬编码关闭状态。

## 🔐 冻结身份

| 对象 | 状态 | 关键身份 |
|---|---:|---|
| v8 Base Lock（基础协议锁） | ✅ | `d5e17ebea346…5472` |
| v8 Implementation Lock（实现与环境锁） | ✅ | `cd558a954e65…a6c0` |
| v8 Mining Rebind Receipt（挖掘重绑定回执） | ✅ | `4354f1bb6ad3…be59` |
| QLib/MLflow 预导入回执 | ✅ | `32f61494c651…4900` |

实现锁覆盖 90 个固定文件、54 个项目源码文件，以及 60,168 个 QLib 数据文件；运行环境记录为 Torch 2.9.1、CUDA 12.9、2 张 GPU，确定性算法开启，TF32 关闭。

## 🔗 v6 → v8 精确重绑定

重绑定不是复制或重新挖矿，而是对 v6 已冻结完整拓扑进行逐文件活体复验，再把同一候选成员资格绑定到 v8 评估协议。

| 检查项 | 结果 |
|---|---:|
| 来源拓扑 | `5194c5c91936…7668` |
| 完整文件库存 | 35 个文件，313,892,483 bytes |
| 文件库存哈希 | `57af46519418…beed` |
| 重新选择 / 重排 / 重新调参 | 均未发生 |
| 复制来源 payload | 未发生 |
| 因重绑定读取标签 | 未发生 |

## 🧪 真实无标签全链路回放

PIT 是 point-in-time 的缩写，表示严格按当时可见信息构造的数据。回放只请求 Open、High、Low、Close、Volume 五个原始字段；虽然缓存文件中存在 `fwd` 标签列，但本次没有读取它。

| 因子臂 | 全量面板 | 冻结挖掘段精确复放 | 结果 |
|---|---:|---:|---:|
| MIRAGE-KAN | 1,572,483 × 7 | 868,920 行逐 AST | ✅ |
| Typed GP/SR 控制 | 1,572,483 × 7 | 868,920 行逐 AST | ✅ |
| 标签置换控制 | 1,572,483 × 7 | 868,920 行逐 AST | ✅ |
| Matched MLP 黑盒控制 | 1,572,483 × 7 | 7 个 checkpoint、完整 300 步轨迹、零容差 | ✅ |

AST 是 abstract syntax tree，即可独立执行的因子公式树；GP/SR 是 genetic programming / symbolic regression，即遗传编程与符号回归；MLP 是 multi-layer perceptron，即多层感知机。

本轮无标签回放耗时 293.147 秒。四类面板全部精确匹配原始 PIT 行索引，没有访问 development opening，也没有写入科学产物。

## 🕸️ Graph 决策已落实

后续协议中的 graph 控制器已从延后选项提升为协同挖掘主线组件。v8 冻结配置、正常 decision、基础设施失败 decision 和最终 manifest 现在都会一致传播 `graph_unlock_allowed: true`。这只是允许开发，不代表预先承认 graph 优于简单控制器；后续仍必须在相同数据、搜索空间、预算、选库和回测规则下比较 random、flat、bandit、graph 四种控制器。

## 🛡️ 开发集 opening 前状态

- ✅ 三路独立 post-lock 审计全部通过。
- ✅ 399 项全量测试通过，Ruff 静态检查通过。
- ✅ QLib 首次导入发生在 v8 专属 tracking 目录，provider 尚未初始化。
- ✅ v8 development preclaim、opening、五臂回测目录、decision 和暂存目录均不存在。
- ✅ v6、v7 历史冻结身份未被改写。
- ℹ️ 1045 条 warning 均为既有 pandas FutureWarning，不是本阶段阻塞项。

## 🚀 下一步

现在执行唯一的 v8 `development` 入口。它会按固定顺序完成：实时复核实现锁与重绑定 → 一次性消耗 development opening → 载入原始 PIT → 顺序运行 Alpha158、MIRAGE-KAN、Typed GP/SR、Matched MLP、置换控制五臂 Quanta 回测 → 装配并发布 decision。

主要横评指标仍是因子库经过统一 Quanta 框架后的成本感知回测质量；因子数、训练轮数、耗时和资源利用率只作为辅助诊断。
