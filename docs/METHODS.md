# Methods Overview

This repository supports an encoder-oriented ESPI workflow:

```text
v6.2 classifier backbone
  -> feature extraction / embeddings
  -> encoder audit
  -> grouped generalization
  -> future acoustic-response prediction
```

## Included Evidence

- v6.1 vs v6.2 summary comparison
- v6.2 encoder audit summary
- grouped encoder generalization summary
- technical appendix notes for the v6.2-to-encoder transition

## Claim Boundary

Currently supported:

```text
v6.2 can be reused as an ESPI representation encoder and audited under grouped domain splits.
```

Not yet supported:

```text
ESPI embeddings improve acoustic-response prediction beyond metadata, frequency,
geometry, and material baselines.
```

That future claim requires acoustic targets, baseline models, ablation studies, and uncertainty evaluation.
