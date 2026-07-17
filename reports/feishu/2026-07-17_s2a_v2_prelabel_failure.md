# ⚠️ MIRAGE-KAN 运行通报：v2 在标签打开前终止

## 结论

真实 v2 运行没有进入挖掘，也没有读取任何标签。它在写入一次性 mining entitlement（挖掘授权收据）时，因为真实 YAML 日期被解析为 `date` 对象、无法直接写入 JSON 而终止。

这属于基础设施失败，不是 MIRAGE-KAN 科学失败。两张 A800 没有开始 KAN 训练，GP/SR 与 MLP 也没有运行，2022–2025 development 完全未打开。

## 现场状态

| 项目 | 状态 |
|---|---|
| Proposal / implementation 校验 | 通过 |
| 七个 mining 目标预占位 | 已创建并全部终态化 |
| mining entitlement | 写入失败；保留 0 字节不可变事故见证 |
| train / validation 标签 | 未读取 |
| KAN / permutation / GP / MLP | 未启动 |
| 因子库与回测结果 | 未产生 |

## 处置

我不会删除空收据、修改 v2 implementation lock 或偷偷重跑。v2 将永久保留为 `terminal_non_scientific_run`。当前正在建立新协议 ID 的 v3 修正版，只修复两点：

1. mining 与 development opening 中的日期统一规范为 ISO 字符串；
2. 完整 JSON 必须先编码成功，之后才允许 `O_EXCL` 创建不可替换文件。

科学设计、阈值、预算和 seed 保持不变，因为本次没有打开标签或产生候选。修复将先用真实 YAML date scalar 做红灯测试，再跑完整确定性测试、签发新实现锁，最后才启动新 topology。

事故与 IVE（实验失败后的经验进化记录）已经分别落盘到 governance 和 memory 目录。
