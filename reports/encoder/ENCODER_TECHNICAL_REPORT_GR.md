# Encoder Evidence v001 — Frozen v6.2-A embeddings

## 1. Source checkpoint and scope

Η παρούσα αναφορά συνοψίζει **post-hoc representation-geometry evidence** από το παγωμένο embedding space του **v6.2-A reportable baseline**. Η πηγή είναι το epoch25 checkpoint `C:\ESPI\FIXED_PACKAGE\baseline_v62_5class\checkpoints\checkpoint_epoch25_20260211_035150.pt`, δηλαδή το checkpoint που συνδέεται με το reference αποτέλεσμα `96.18% Accuracy / 90.97% Macro-F1`. Δεν εκπαιδεύτηκε νέος classifier, δεν έγινε νέο feature extraction, και δεν πρόκειται για **true LOBO generalization proof** ή για πλήρως validated Physics-Aligned Encoder.

Ο στόχος του ελέγχου είναι στενός και τεχνικός: να εξεταστεί αν τα frozen pre-head embeddings του v6.2-A περιέχουν χρήσιμη modal δομή όταν αφαιρεθεί η τελική ταξινομητική κεφαλή.

## 2. Feature extraction audit

- Δείγματα: `12944`
- Διάσταση embedding: `1280`
- Embedding layer: `MCDropoutClassifier.global_pool`
- Load audit: missing keys `0`, unexpected keys `0`
- Metadata fields: `sample_id`, `label`, `label_name`, `board`, `source_group`, `material`, `frequency_hz`, `domain`, `path`, `split`, `split_group`
- NaN embeddings: `0`
- Inf embeddings: `0`
- Duplicate paths: `0`
- Δεν έγινε reload raw images, δεν έγινε GPU inference, δεν έγινε retraining, και δεν δημιουργήθηκε αντίγραφο των embeddings.

## 3. Frozen embedding evaluation

| Έλεγχος | Μέθοδος | Accuracy | Macro Recall | Macro-F1 | Σχόλιο |
| --- | --- | --- | --- | --- | --- |
| Stratified train→val | kNN cosine k=10 | 97.37% | 91.97% | 93.44% | Καλύτερο kNN πάνω στα frozen embeddings. |
| Stratified train→val | Linear probe — balanced logistic regression | 96.83% | 93.84% | 92.85% | Ελέγχει γραμμική διαθεσιμότητα πληροφορίας. |
| Stratified train→val | Nearest class prototype — cosine | 93.47% | 91.98% | 86.42% | Ελέγχει αν οι κλάσεις έχουν καθαρά κέντρα. |
| Leave-split-group-out | kNN cosine k=10 | 97.26% | 92.36% | 93.89% | Μέσος όρος σε 6 groups. |
| Leave-split-group-out | Minimum grouped Macro-F1 | — | — | 91.28% | Χειρότερο group για το καλύτερο aggregate kNN. |


Προαιρετικά PCA diagnostics δημιουργήθηκαν με CPU-only SVD σε `3000` από `12944` embeddings:
- `C:\Users\George mini-PC\Documents\Playground\v62_encoder_evidence\embedding_audit_v001\pca_by_class.png`
- `C:\Users\George mini-PC\Documents\Playground\v62_encoder_evidence\embedding_audit_v001\pca_by_board.png`
- `C:\Users\George mini-PC\Documents\Playground\v62_encoder_evidence\embedding_audit_v001\pca_by_material.png`

## 4. Interpretation

Ο frozen v6.2-A embedding χώρος δείχνει ισχυρή post-hoc modal geometry. Το καλύτερο kNN αποτέλεσμα και το linear probe είναι κοντά ή πάνω από το classifier reference Macro-F1 του v6.2-A reportable baseline, κάτι που υποστηρίζει τη χρήση του v6.2-A ως **frozen ESPI encoder candidate**. Το prototype αποτέλεσμα είναι χαμηλότερο αλλά παραμένει χρήσιμο, επειδή δείχνει ότι η πληροφορία υπάρχει στον χώρο, ενώ η καθαρή centroid geometry δεν είναι εξίσου ισχυρή για όλες τις κλάσεις.

## 5. Caveat

Αυτό **δεν είναι true LOBO/LOMO encoder proof**. Το frozen checkpoint εκπαιδεύτηκε στο αρχικό stratified split. Το leave-split-group audit αφαιρεί groups από το retrieval/probe reference set, αλλά ο ίδιος ο frozen encoder δεν έχει εκπαιδευτεί με το target board/material αποκλεισμένο. Επομένως το αποτέλεσμα πρέπει να διατυπώνεται ως **post-hoc representation-geometry evidence** και όχι ως οριστική απόδειξη γενίκευσης σε LOBO/LOMO.

## 6. Next step

Το επόμενο αυστηρό βήμα είναι ένα από τα δύο:

- true LOBO/LOMO encoder training, όπου το target board ή material αποκλείεται ήδη κατά την εκπαίδευση,
- supervised contrastive ή domain-aware fine-tuning, ώστε δείγματα ίδιας modal κλάσης από διαφορετικά boards/materials να έρχονται κοντά και hard negatives κοντινής συχνότητας να παραμένουν διαχωρίσιμα.

## 7. Appendix-ready claim

Η εξαγωγή frozen pre-head embeddings από το v6.2-A reportable baseline έδειξε ότι ο χώρος αναπαραστάσεων διατηρεί ισχυρή ταξινομητική και modal πληροφορία χωρίς επανεκπαίδευση του CNN ή χρήση της τελικής ταξινομητικής κεφαλής. Τα αποτελέσματα kNN, prototype και linear probe παρέχουν post-hoc representation-geometry evidence ότι το v6.2-A μπορεί να χρησιμοποιηθεί ως **frozen ESPI encoder candidate**. Ωστόσο, επειδή το checkpoint εκπαιδεύτηκε στο αρχικό stratified split, τα grouped retrieval/probe αποτελέσματα δεν αποτελούν ακόμη true LOBO/LOMO generalization proof· η αυστηρή επικύρωση απαιτεί LOBO/LOMO encoder training ή supervised contrastive/domain-aware fine-tuning.
