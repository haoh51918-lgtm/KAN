# MIRAGE-KAN Figure Sketches

## Method figure

```mermaid
flowchart LR
    A[PIT OHLCV\nmembership ≠ observed] --> B[Typed financial DSL\ncausality · domain · masks]
    B --> C[Symbolic-Residual KAN miner\nanalytic gates + penalized spline]
    C --> D[Hardening and canonical AST]
    D --> E[Immutable factor library]
    E --> F[Pinned QuantaAlpha\nLightGBM + portfolio backtest]
    E --> G[Factor nodes\nstructure · behavior · stability]
    H[Miner nodes\nsearch policy · budget · lineage] --> C
    G --> I[Novel: Miner–Factor feedback\nresidual task · diversity · evolution · budget]
    H --> I
    I --> H
```

Visual priority: gray out the standard PIT and Quanta blocks; use one color for the Symbolic-Residual miner and another for the feedback loop. The exported AST/library path must visibly bypass graph state at inference.

## Evidence dependency figure

```mermaid
flowchart TD
    S0[Real vertical slice] --> S2[Single-miner library value]
    S0 --> S1[KAN out-of-dictionary kill test]
    S1 -->|KAN supported| S2
    S1 -->|KAN unsupported| PC[KAN de-centered fallback]
    S2 -->|Backtest value| S3[Independent-miner redundancy]
    S2 -->|No value| B1[Backtrack objective / hardening / selection]
    S3 -->|Redundancy exists| S4[Minimal graph feedback]
    S3 -->|No redundancy| PB[Single-miner paper]
    S4 -->|Beats non-graph controls| S5[Typed evolution and budget]
    S4 -->|No graph advantage| PF[Non-graph closed-loop framing]
    S5 --> S6[Robustness and lockbox confirmation]
```

## Teaser placeholder

Use a two-panel result figure only after evidence exists:

- Left: net Information Ratio with paired uncertainty for baseline, single miner, independent miners, and MIRAGE-KAN.
- Right: effective rank or unique valid AST yield at the same full-evaluation budget.

Do not draw expected bars or placeholder gains that could anchor later interpretation.

