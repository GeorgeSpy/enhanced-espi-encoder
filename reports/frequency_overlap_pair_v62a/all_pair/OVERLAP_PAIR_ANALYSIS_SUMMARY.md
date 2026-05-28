# Frequency-Overlap Pair Analysis

This analysis isolates the target modal pair and evaluates whether image-derived embeddings help when frequency information is ambiguous.

- Labels: `[2, 3]` / `['2', '3']`
- Overlap only: `False`
- Selected samples: `1507`
- Frequency overlap range: `550.0000` to `556.0000` Hz

## Frequency ranges by class

| label | label_name | freq_min | freq_max | n   |
| ----- | ---------- | -------- | -------- | --- |
| 2     | 2          | 500.0000 | 556.0000 | 738 |
| 3     | 3          | 550.0000 | 605.0000 | 769 |

## Overall metrics

| accuracy | balanced_accuracy | macro_f1 | weighted_f1 | protocol      | family | feature_kind | n    |
| -------- | ----------------- | -------- | ----------- | ------------- | ------ | ------------ | ---- |
| 0.9257   | 0.9256            | 0.9256   | 0.9257      | board_lobo    | lr     | embed        | 1507 |
| 0.9456   | 0.9459            | 0.9456   | 0.9456      | board_lobo    | lr     | freq         | 1507 |
| 0.9801   | 0.9801            | 0.9801   | 0.9801      | board_lobo    | lr     | fusion       | 1507 |
| 0.9502   | 0.9500            | 0.9502   | 0.9502      | board_lobo    | rf     | embed        | 1507 |
| 0.9270   | 0.9264            | 0.9268   | 0.9269      | board_lobo    | rf     | freq         | 1507 |
| 0.9602   | 0.9600            | 0.9602   | 0.9602      | board_lobo    | rf     | fusion       | 1507 |
| 0.9383   | 0.9378            | 0.9382   | 0.9382      | material_lomo | lr     | embed        | 1507 |
| 0.8567   | 0.8572            | 0.8567   | 0.8566      | material_lomo | lr     | freq         | 1507 |
| 0.9681   | 0.9681            | 0.9681   | 0.9681      | material_lomo | lr     | fusion       | 1507 |
| 0.9396   | 0.9399            | 0.9396   | 0.9396      | material_lomo | rf     | embed        | 1507 |
| 0.8567   | 0.8572            | 0.8567   | 0.8566      | material_lomo | rf     | freq         | 1507 |
| 0.9476   | 0.9477            | 0.9476   | 0.9476      | material_lomo | rf     | fusion       | 1507 |
| 0.9402   | 0.9405            | 0.9402   | 0.9402      | stratified    | lr     | embed        | 301  |
| 0.9236   | 0.9239            | 0.9236   | 0.9236      | stratified    | lr     | freq         | 301  |
| 0.9834   | 0.9836            | 0.9834   | 0.9834      | stratified    | lr     | fusion       | 301  |
| 0.9203   | 0.9207            | 0.9203   | 0.9203      | stratified    | rf     | embed        | 301  |
| 0.9668   | 0.9663            | 0.9667   | 0.9667      | stratified    | rf     | freq         | 301  |
| 0.9269   | 0.9273            | 0.9269   | 0.9269      | stratified    | rf     | fusion       | 301  |

## Fold metrics

| accuracy | balanced_accuracy | macro_f1 | weighted_f1 | protocol      | family | feature_kind | heldout_group | n_train | n_test |
| -------- | ----------------- | -------- | ----------- | ------------- | ------ | ------------ | ------------- | ------- | ------ |
| 0.8955   | 0.8909            | 0.8926   | 0.8953      | board_lobo    | lr     | embed        | C01           | 1373    | 134    |
| 0.9631   | 0.9610            | 0.9628   | 0.9630      | board_lobo    | lr     | embed        | C02           | 1209    | 298    |
| 0.9395   | 0.9353            | 0.9385   | 0.9392      | board_lobo    | lr     | embed        | C03           | 1292    | 215    |
| 0.8950   | 0.8953            | 0.8950   | 0.8950      | board_lobo    | lr     | embed        | W01           | 1145    | 362    |
| 0.9421   | 0.9417            | 0.9417   | 0.9421      | board_lobo    | lr     | embed        | W02           | 1196    | 311    |
| 0.9037   | 0.8744            | 0.8886   | 0.9015      | board_lobo    | lr     | embed        | W03           | 1320    | 187    |
| 1.0000   | 1.0000            | 1.0000   | 1.0000      | board_lobo    | lr     | freq         | C01           | 1373    | 134    |
| 0.9362   | 0.9326            | 0.9353   | 0.9358      | board_lobo    | lr     | freq         | C02           | 1209    | 298    |
| 0.9488   | 0.9439            | 0.9478   | 0.9485      | board_lobo    | lr     | freq         | C03           | 1292    | 215    |
| 0.9475   | 0.9486            | 0.9475   | 0.9474      | board_lobo    | lr     | freq         | W01           | 1145    | 362    |
| 0.9421   | 0.9467            | 0.9421   | 0.9422      | board_lobo    | lr     | freq         | W02           | 1196    | 311    |
| 0.9198   | 0.8828            | 0.9049   | 0.9168      | board_lobo    | lr     | freq         | W03           | 1320    | 187    |
| 0.9478   | 0.9409            | 0.9459   | 0.9474      | board_lobo    | lr     | fusion       | C01           | 1373    | 134    |
| 0.9899   | 0.9894            | 0.9899   | 0.9899      | board_lobo    | lr     | fusion       | C02           | 1209    | 298    |
| 0.9907   | 0.9898            | 0.9906   | 0.9907      | board_lobo    | lr     | fusion       | C03           | 1292    | 215    |
| 0.9807   | 0.9806            | 0.9807   | 0.9807      | board_lobo    | lr     | fusion       | W01           | 1145    | 362    |
| 0.9904   | 0.9911            | 0.9903   | 0.9904      | board_lobo    | lr     | fusion       | W02           | 1196    | 311    |
| 0.9572   | 0.9375            | 0.9509   | 0.9565      | board_lobo    | lr     | fusion       | W03           | 1320    | 187    |
| 0.9104   | 0.9061            | 0.9080   | 0.9102      | board_lobo    | rf     | embed        | C01           | 1373    | 134    |
| 0.9732   | 0.9720            | 0.9730   | 0.9731      | board_lobo    | rf     | embed        | C02           | 1209    | 298    |
| 0.9628   | 0.9608            | 0.9624   | 0.9627      | board_lobo    | rf     | embed        | C03           | 1292    | 215    |
| 0.9365   | 0.9359            | 0.9363   | 0.9364      | board_lobo    | rf     | embed        | W01           | 1145    | 362    |
| 0.9582   | 0.9570            | 0.9578   | 0.9582      | board_lobo    | rf     | embed        | W02           | 1196    | 311    |
| 0.9412   | 0.9178            | 0.9322   | 0.9400      | board_lobo    | rf     | embed        | W03           | 1320    | 187    |
| 0.8284   | 0.7982            | 0.8087   | 0.8178      | board_lobo    | rf     | freq         | C01           | 1373    | 134    |
| 0.8691   | 0.8617            | 0.8645   | 0.8659      | board_lobo    | rf     | freq         | C02           | 1209    | 298    |
| 0.9395   | 0.9337            | 0.9382   | 0.9390      | board_lobo    | rf     | freq         | C03           | 1292    | 215    |
| 0.9503   | 0.9514            | 0.9502   | 0.9502      | board_lobo    | rf     | freq         | W01           | 1145    | 362    |
| 0.9453   | 0.9497            | 0.9453   | 0.9454      | board_lobo    | rf     | freq         | W02           | 1196    | 311    |
| 1.0000   | 1.0000            | 1.0000   | 1.0000      | board_lobo    | rf     | freq         | W03           | 1320    | 187    |
| 0.9254   | 0.9168            | 0.9225   | 0.9247      | board_lobo    | rf     | fusion       | C01           | 1373    | 134    |
| 0.9765   | 0.9755            | 0.9764   | 0.9765      | board_lobo    | rf     | fusion       | C02           | 1209    | 298    |
| 0.9674   | 0.9643            | 0.9670   | 0.9673      | board_lobo    | rf     | fusion       | C03           | 1292    | 215    |
| 0.9586   | 0.9585            | 0.9585   | 0.9586      | board_lobo    | rf     | fusion       | W01           | 1145    | 362    |
| 0.9678   | 0.9665            | 0.9675   | 0.9678      | board_lobo    | rf     | fusion       | W02           | 1196    | 311    |
| 0.9412   | 0.9178            | 0.9322   | 0.9400      | board_lobo    | rf     | fusion       | W03           | 1320    | 187    |
| 0.9459   | 0.9427            | 0.9451   | 0.9457      | material_lomo | lr     | embed        | carbon        | 860     | 647    |
| 0.9326   | 0.9329            | 0.9325   | 0.9326      | material_lomo | lr     | embed        | wood          | 647     | 860    |
| 0.8671   | 0.8547            | 0.8605   | 0.8630      | material_lomo | lr     | freq         | carbon        | 860     | 647    |
| 0.8488   | 0.8445            | 0.8438   | 0.8446      | material_lomo | lr     | freq         | wood          | 647     | 860    |
| 0.9614   | 0.9580            | 0.9608   | 0.9612      | material_lomo | lr     | fusion       | carbon        | 860     | 647    |
| 0.9733   | 0.9725            | 0.9732   | 0.9732      | material_lomo | lr     | fusion       | wood          | 647     | 860    |
| 0.9474   | 0.9487            | 0.9472   | 0.9475      | material_lomo | rf     | embed        | carbon        | 860     | 647    |
| 0.9337   | 0.9334            | 0.9336   | 0.9337      | material_lomo | rf     | embed        | wood          | 647     | 860    |
| 0.8671   | 0.8547            | 0.8605   | 0.8630      | material_lomo | rf     | freq         | carbon        | 860     | 647    |
| 0.8488   | 0.8445            | 0.8438   | 0.8446      | material_lomo | rf     | freq         | wood          | 647     | 860    |
| 0.9629   | 0.9626            | 0.9626   | 0.9629      | material_lomo | rf     | fusion       | carbon        | 860     | 647    |
| 0.9360   | 0.9358            | 0.9360   | 0.9360      | material_lomo | rf     | fusion       | wood          | 647     | 860    |
| 0.9402   | 0.9405            | 0.9402   | 0.9402      | stratified    | lr     | embed        | validation    | 1206    | 301    |
| 0.9236   | 0.9239            | 0.9236   | 0.9236      | stratified    | lr     | freq         | validation    | 1206    | 301    |
| 0.9834   | 0.9836            | 0.9834   | 0.9834      | stratified    | lr     | fusion       | validation    | 1206    | 301    |
| 0.9203   | 0.9207            | 0.9203   | 0.9203      | stratified    | rf     | embed        | validation    | 1206    | 301    |
| 0.9668   | 0.9663            | 0.9667   | 0.9667      | stratified    | rf     | freq         | validation    | 1206    | 301    |
| 0.9269   | 0.9273            | 0.9269   | 0.9269      | stratified    | rf     | fusion       | validation    | 1206    | 301    |

## Output files

- `reports\frequency_overlap_pair_v62a\all_pair\overlap_pair_overall_metrics.csv`
- `reports\frequency_overlap_pair_v62a\all_pair\overlap_pair_fold_metrics.csv`
- `reports\frequency_overlap_pair_v62a\all_pair\overlap_pair_confusion_counts.csv`
- `reports\frequency_overlap_pair_v62a\all_pair\overlap_pair_key_numbers.json`
