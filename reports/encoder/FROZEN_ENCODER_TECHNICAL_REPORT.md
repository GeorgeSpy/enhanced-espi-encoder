# Encoder Evidence v001 - Frozen v6.2-A Embeddings

## 1. Source checkpoint and scope

This report summarizes **post-hoc representation-geometry evidence** from the frozen embedding space of the **v6.2-A reportable baseline**. The source is the epoch-25 checkpoint associated with the internal reference result `96.18% Accuracy / 90.97% Macro-F1`.

No new classifier was trained, no new feature extraction is implied by this report, and the result must not be presented as **true LOBO generalization proof** or as a fully validated Physics-Aligned Encoder.

The technical objective is narrow: evaluate whether frozen pre-head embeddings from v6.2-A retain useful modal structure after removing the final classifier head.

## 2. Feature extraction audit

- Samples: `12944`
- Embedding dimension: `1280`
- Embedding layer: `MCDropoutClassifier.global_pool`
- Load audit: missing keys `0`, unexpected keys `0`
- Metadata fields: `sample_id`, `label`, `label_name`, `board`, `source_group`, `material`, `frequency_hz`, `domain`, `path`, `split`, `split_group`
- NaN embeddings: `0`
- Inf embeddings: `0`
- Duplicate paths: `0`
- The report generator does not reload raw images, run GPU inference, retrain the CNN, or duplicate the embedding array.

## 3. Frozen embedding evaluation

| Evaluation | Method | Accuracy | Macro Recall | Macro-F1 | Comment |
| --- | --- | --- | --- | --- | --- |
| Stratified train-to-val | kNN cosine k=10 | 97.37% | 91.97% | 93.44% | Best kNN result over frozen embeddings. |
| Stratified train-to-val | Linear probe - balanced logistic regression | 96.83% | 93.84% | 92.85% | Tests linear availability of representation information. |
| Stratified train-to-val | Nearest class prototype - cosine | 93.47% | 91.98% | 86.42% | Tests whether classes form clean prototype centers. |
| Leave-split-group-out | kNN cosine k=10 | 97.26% | 92.36% | 93.89% | Mean over 6 groups. |
| Leave-split-group-out | Minimum grouped Macro-F1 | - | - | 91.28% | Worst group for the best aggregate kNN method. |

Optional PCA diagnostics were generated externally with CPU-only SVD on `3000` of `12944` embeddings. The tracked repository keeps only summary evidence, not the full feature dump or generated figure binaries.

## 4. Interpretation

The frozen v6.2-A embedding space shows strong post-hoc modal geometry. The best kNN result and the linear probe are close to or above the classifier reference Macro-F1 of the v6.2-A reportable baseline. This supports using v6.2-A as a **frozen ESPI encoder candidate**.

The prototype result is lower but still informative: it indicates that class information exists in the embedding space, while pure centroid geometry is not equally strong for all classes.

## 5. Caveat

This is **not true LOBO/LOMO encoder proof**. The frozen checkpoint was trained on the original stratified split. The leave-split-group audit removes groups from the retrieval/probe reference set, but the frozen encoder itself was not trained with the target board or material excluded.

The result should therefore be framed as **post-hoc representation-geometry evidence**, not as definitive LOBO/LOMO generalization proof.

## 6. Next step

The next strict validation step is one of the following:

- true LOBO/LOMO encoder training, where the target board or material is excluded during training,
- supervised contrastive or domain-aware fine-tuning, so samples from the same modal class but different boards/materials are pulled together while frequency-adjacent hard negatives remain separable.

## 7. Appendix-ready claim

Frozen pre-head embeddings extracted from the v6.2-A reportable baseline retain strong classification and modal information without retraining the CNN or using the final classifier head. kNN, prototype, and linear-probe results provide post-hoc representation-geometry evidence that v6.2-A can be used as a **frozen ESPI encoder candidate**. Because the checkpoint was trained on the original stratified split, the grouped retrieval/probe results are not yet true LOBO/LOMO generalization proof. Strict validation requires LOBO/LOMO encoder training or supervised contrastive/domain-aware fine-tuning.
