# Governance custody: S2a v3 corrective successor

- **Date**: 2026-07-17 UTC
- **Predecessor**: `s2a_kan_e3_vertical_v2`
- **Successor**: `s2a_kan_e3_vertical_v3`
- **Authorization class**: corrective successor after pre-label infrastructure failure
- **Scientific-design change**: none

## Authorization boundary

S2a v3 is authorized only because v2 failed before any PIT label was loaded and
before KAN, GP/SR, permutation, MLP, factor admission, or Quanta evaluation
began. The sole proposal authority remains `KAN_Alpha_PR.md`. The WIKI and all
predecessor results remain non-authoritative references.

The successor changes only the protocol identity, every writable output path,
typed YAML-date canonicalization at opening-receipt boundaries, exclusive-write
ordering, and Quanta recorder namespace. Seeds, budgets, data splits, purges,
warm-up, search space, training, admission, controls, mechanism tests, backtest,
metrics, and decision thresholds are unchanged from v2.

## Immutable v2 custody pins

| Evidence | SHA-256 |
|---|---|
| v2 base lock | `65d00b5ea05336fd28c5277340aa3ff38f602f6c7d07952b1af6dd04fb504ae7` |
| v2 implementation lock | `f22e592379f8045aef0f690e770eee54d1dd27415404760ecd99d27f778863bd` |
| v2 failure incident | `800bcc967b44d2d7fef378742564a6740ecd53d08584c398f88ba1fda756ce12` |
| v2 mining preclaim | `04d62cae982e5bfb02c72d8c8b6b3deb0dc420ebad8b8816266295875756f5d2` |
| v2 zero-byte mining entitlement | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| v2 authority ledger | `0f3fa79b511960047176f67bf8bad0319cd50d23e4b580e2d80a42349b9a407c` |

The zero-byte entitlement is mode `0444` and is preserved as the direct failure
witness. It must never be deleted, repaired, or reused.

| Terminal v2 target | `terminal_failure.json` SHA-256 |
|---|---|
| mining bundle | `afe8bc304d09fbf8534e92852dcc085e6f6ae75177f62d1129ec981a39ba97ef` |
| production KAN library | `cd81ed7b30bbe3e24ece907ff5edf9eb11fa626dc5bf459b08935a40b77ca437` |
| typed GP/SR control | `9384dfd283dcabdad2f6da672c029c35c4ca23113218050fc96680747aa55cf0` |
| permutation control | `af7ea16cb402a91536838373834bdf8e146560daf29eab0b46463bd92c498d6f` |
| matched MLP control | `e72b872f81c207d2af4f892b68b793bfaeef4622dc637dda3dfb7ec0764cbdd6` |
| mechanism cards | `d06825e138668d33858852a3d288aa34f6246d645c0a543272b37aa3d77fc2a6` |
| blind-review package | `716a53ee94dcf4ef6e052e588e932529531383e2fc22f2b2966e082655eee731` |

## Required v3 gates

1. Real unquoted YAML dates must serialize to ISO strings in both mining and
   development receipts.
2. Serialization failure must occur before `O_EXCL` creates a destination.
3. Parsed v3 scientific settings must equal v2 after removing protocol identity
   and artifact paths.
4. Every v3 writable path must be disjoint from every v2 writable path.
5. The complete deterministic test and lint suites must pass before the v3
   implementation lock is issued and before first label access.

