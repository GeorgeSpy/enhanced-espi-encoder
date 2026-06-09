# LeFFT Development Summary

This document summarizes H2 and H3 LeFFT development tracks as development-only diagnostic evidence.

## H2 Standalone Clean LeFFT

H2 tested standalone CE-only clean LeFFT variants under grouped transfer protocols. The track implemented design documents, module smoke tests, loss tests, pair audits, and reduced/factorial CE-only training protocols.

Final status: standalone CE-only LeFFT failed grouped transfer. The best H2.4C condition, S2_v002_W0, reached LOBO Macro-F1 0.294294 versus the 0.608 go/no-go threshold. H2.5 remains blocked.

## H3 Auxiliary LeFFT

H3 tested the LeFFT auxiliary branch as Phase-1 self-supervised/physics-aware pretraining, without v6.2-A checkpoints, fusion, acoustic targets, CE, SupCon, or frequency auxiliary. Mechanics and safety repeatedly passed, but representation transfer remained insufficient or negative.

Final status: no further simple H3.1 scaling is recommended. H3.2 run remains blocked. H3.2 no-harm fusion design-only may be drafted separately.

## Future Work

Future LeFFT work requires stronger objectives, anchored/fusion design with no-harm criteria, downstream acoustic-response target acquisition, or new transfer-positive evidence.
