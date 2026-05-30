# Spectral Descriptor Ablation Evaluation

## Scope

This report evaluates deterministic spectral / LeFFT-inspired descriptors under the same information-budget protocols used for frequency and v6.2-A controls. It uses saved normalized feature dumps only. No raw images, deep-model training, encoder fine-tuning, trained LeFFT model, or acoustic-response prediction is used.

## Conditions

- `frequency_only`: Frequency only.
- `spectral_descriptors_only`: Spectral descriptors only.
- `v6_2_a_embedding_only`: v6.2-A embedding only.
- `frequency_spectral_fusion`: Frequency + spectral descriptors.
- `v6_2_a_spectral_fusion`: v6.2-A + spectral descriptors.
- `v6_2_a_frequency_fusion`: Frequency + v6.2-A.
- `frequency_v6_2_a_spectral_fusion`: Frequency + v6.2-A + spectral descriptors.

## Models and preprocessing

- `balanced_logistic_regression`: Balanced logistic regression on train/reference-fold standardized inputs. Parameters: `{"class_weight": "balanced", "max_iter": 5000, "random_state": 42, "preprocessing": "train/reference-fold standardization"}`
- `random_forest`: Random forest control on train/reference-fold standardized inputs. Parameters: `{"n_estimators": 80, "class_weight": "balanced_subsample", "min_samples_leaf": 2, "max_features": "sqrt", "random_state": 42, "preprocessing": "train/reference-fold standardization"}`

All standardization is fitted only on the training/reference fold and then applied to the held-out fold. Grouped held-out samples are never used to fit preprocessing.

## Summary metrics

| protocol | condition | display_name | method | macro_f1 | mean_macro_f1 | accuracy | macro_recall | worst_group | worst_group_macro_f1 | best_group | best_group_macro_f1 | n_groups |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stratified_train_val | frequency_only | Frequency only | balanced_logistic_regression | 0.9694338664635694 |  | 0.9911162611046737 | 0.9695732838589983 |  |  |  |  |  |
| stratified_train_val | frequency_only | Frequency only | random_forest | 0.9866861288039631 |  | 0.9961375048281189 | 0.9865182436611007 |  |  |  |  |  |
| stratified_train_val | spectral_descriptors_only | Spectral descriptors only | balanced_logistic_regression | 0.37757487894297204 |  | 0.52954808806489 | 0.5393321414329508 |  |  |  |  |  |
| stratified_train_val | spectral_descriptors_only | Spectral descriptors only | random_forest | 0.529208462087193 |  | 0.8412514484356894 | 0.4481336537993009 |  |  |  |  |  |
| stratified_train_val | v6_2_a_embedding_only | v6.2-A embedding only | balanced_logistic_regression | 0.9260198668851682 |  | 0.9675550405561993 | 0.9354991877973319 |  |  |  |  |  |
| stratified_train_val | v6_2_a_embedding_only | v6.2-A embedding only | random_forest | 0.9348100151719869 |  | 0.9721900347624566 | 0.9180915394515223 |  |  |  |  |  |
| stratified_train_val | frequency_spectral_fusion | Frequency + spectral descriptors | balanced_logistic_regression | 0.9853435959740521 |  | 0.9945925067593665 | 0.9887698394865968 |  |  |  |  |  |
| stratified_train_val | frequency_spectral_fusion | Frequency + spectral descriptors | random_forest | 0.9729612583113866 |  | 0.9891850135187331 | 0.962568419498057 |  |  |  |  |  |
| stratified_train_val | v6_2_a_spectral_fusion | v6.2-A + spectral descriptors | balanced_logistic_regression | 0.9328787705827952 |  | 0.9706450366937042 | 0.9389177851011349 |  |  |  |  |  |
| stratified_train_val | v6_2_a_spectral_fusion | v6.2-A + spectral descriptors | random_forest | 0.934808068951973 |  | 0.9725762842796446 | 0.9182522454451497 |  |  |  |  |  |
| stratified_train_val | v6_2_a_frequency_fusion | Frequency + v6.2-A | balanced_logistic_regression | 0.9674609129599487 |  | 0.988798764001545 | 0.9682407312851632 |  |  |  |  |  |
| stratified_train_val | v6_2_a_frequency_fusion | Frequency + v6.2-A | random_forest | 0.9403228145499579 |  | 0.9756662804171495 | 0.9230986972315629 |  |  |  |  |  |
| stratified_train_val | frequency_v6_2_a_spectral_fusion | Frequency + v6.2-A + spectral descriptors | balanced_logistic_regression | 0.9641356080871841 |  | 0.9872537659327926 | 0.9626517364922427 |  |  |  |  |  |
| stratified_train_val | frequency_v6_2_a_spectral_fusion | Frequency + v6.2-A + spectral descriptors | random_forest | 0.9381700367287722 |  | 0.9748937813827733 | 0.9188454238929994 |  |  |  |  |  |
| board_grouped | frequency_only | Frequency only | balanced_logistic_regression |  | 0.9748609579888732 | 0.9920388104708425 | 0.9736275011524164 | W02 | 0.9549541161012653 | C01 | 1.0 | 6 |
| board_grouped | frequency_only | Frequency only | random_forest |  | 0.9671234211089278 | 0.9920352170805402 | 0.9663117802963735 | C01 | 0.9234618488855777 | W03 | 1.0 | 6 |
| board_grouped | spectral_descriptors_only | Spectral descriptors only | balanced_logistic_regression |  | 0.2179926112879648 | 0.36883334983256316 | 0.3217510871880099 | C03 | 0.09712015823880564 | W01 | 0.31396767801434294 | 6 |
| board_grouped | spectral_descriptors_only | Spectral descriptors only | random_forest |  | 0.22773641065070063 | 0.762053138577611 | 0.24399832316837025 | W02 | 0.1801384127390984 | W03 | 0.2834753354306149 | 6 |
| board_grouped | v6_2_a_embedding_only | v6.2-A embedding only | balanced_logistic_regression |  | 0.9150209184272143 | 0.9589800606239431 | 0.9318690542604107 | C02 | 0.8837789055610414 | C03 | 0.9476322666655153 | 6 |
| board_grouped | v6_2_a_embedding_only | v6.2-A embedding only | random_forest |  | 0.9428118238555557 | 0.9745517802230447 | 0.9249693733103382 | C02 | 0.9325992413312496 | C03 | 0.9602754585994809 | 6 |
| board_grouped | frequency_spectral_fusion | Frequency + spectral descriptors | balanced_logistic_regression |  | 0.9346054422895418 | 0.9790081796632769 | 0.9389666782491517 | C03 | 0.9092640997065976 | C01 | 0.9707111814978099 | 6 |
| board_grouped | frequency_spectral_fusion | Frequency + spectral descriptors | random_forest |  | 0.8620335122846564 | 0.9484004274081769 | 0.829078420001584 | W02 | 0.7551064756909031 | W01 | 0.9646738997442134 | 6 |
| board_grouped | v6_2_a_spectral_fusion | v6.2-A + spectral descriptors | balanced_logistic_regression |  | 0.8904279281676635 | 0.9474507557047981 | 0.9247726547563592 | W01 | 0.8766442739497927 | C03 | 0.9094818806765457 | 6 |
| board_grouped | v6_2_a_spectral_fusion | v6.2-A + spectral descriptors | random_forest |  | 0.9435480245622432 | 0.9746276426551431 | 0.9258302665370745 | W01 | 0.9338681219250358 | C03 | 0.9592429329354175 | 6 |
| board_grouped | v6_2_a_frequency_fusion | Frequency + v6.2-A | balanced_logistic_regression |  | 0.9423828105576045 | 0.9772013946008129 | 0.9489942645502972 | C02 | 0.9201529293819523 | W02 | 0.9635238020709849 | 6 |
| board_grouped | v6_2_a_frequency_fusion | Frequency + v6.2-A | random_forest |  | 0.9478550859511922 | 0.9770509232454768 | 0.9291075530986235 | C01 | 0.9378499810309217 | C03 | 0.9602754585994809 | 6 |
| board_grouped | frequency_v6_2_a_spectral_fusion | Frequency + v6.2-A + spectral descriptors | balanced_logistic_regression |  | 0.9307482579886867 | 0.9730281694167408 | 0.9466427550238338 | C02 | 0.919145351747348 | W02 | 0.9428176168976012 | 6 |
| board_grouped | frequency_v6_2_a_spectral_fusion | Frequency + v6.2-A + spectral descriptors | random_forest |  | 0.949738643728922 | 0.9784026938991301 | 0.9298888282222059 | C01 | 0.9378499810309217 | C03 | 0.9632054872969553 | 6 |
| material_grouped | frequency_only | Frequency only | balanced_logistic_regression |  | 0.9408561364311898 | 0.983179848705876 | 0.9398454674770464 | wood | 0.9375317788915524 | carbon | 0.9441804939708274 | 2 |
| material_grouped | frequency_only | Frequency only | random_forest |  | 0.9408561364311898 | 0.983179848705876 | 0.9398454674770464 | wood | 0.9375317788915524 | carbon | 0.9441804939708274 | 2 |
| material_grouped | spectral_descriptors_only | Spectral descriptors only | balanced_logistic_regression |  | 0.08906281372113617 | 0.11866923265095503 | 0.18374612824022624 | wood | 0.08767471996615621 | carbon | 0.09045090747611614 | 2 |
| material_grouped | spectral_descriptors_only | Spectral descriptors only | random_forest |  | 0.19720081570542955 | 0.7441575398673437 | 0.21444150112437135 | wood | 0.19515713707300453 | carbon | 0.19924449433785457 | 2 |
| material_grouped | v6_2_a_embedding_only | v6.2-A embedding only | balanced_logistic_regression |  | 0.9110766490505552 | 0.9568585884768042 | 0.9294017309163742 | wood | 0.8899964004860413 | carbon | 0.932156897615069 | 2 |
| material_grouped | v6_2_a_embedding_only | v6.2-A embedding only | random_forest |  | 0.9405498432710986 | 0.9729580844219825 | 0.9219056682417639 | wood | 0.9377956484963115 | carbon | 0.9433040380458856 | 2 |
| material_grouped | frequency_spectral_fusion | Frequency + spectral descriptors | balanced_logistic_regression |  | 0.7116194739926678 | 0.8872981626658607 | 0.7404418983132395 | carbon | 0.6263682554748604 | wood | 0.796870692510475 | 2 |
| material_grouped | frequency_spectral_fusion | Frequency + spectral descriptors | random_forest |  | 0.7332843852810769 | 0.9166503723318251 | 0.7100073515889932 | carbon | 0.7095624614304572 | wood | 0.7570063091316964 | 2 |
| material_grouped | v6_2_a_spectral_fusion | v6.2-A + spectral descriptors | balanced_logistic_regression |  | 0.8677392022773942 | 0.9330745360356953 | 0.9217909655101382 | wood | 0.8388886048778372 | carbon | 0.8965897996769512 | 2 |
| material_grouped | v6_2_a_spectral_fusion | v6.2-A + spectral descriptors | random_forest |  | 0.9400464998332079 | 0.9731231429865825 | 0.9226657814092363 | wood | 0.9363964400144245 | carbon | 0.9436965596519912 | 2 |
| material_grouped | v6_2_a_frequency_fusion | Frequency + v6.2-A | balanced_logistic_regression |  | 0.9402706029527824 | 0.9746669520415959 | 0.9454322059652762 | wood | 0.9312595914006636 | carbon | 0.9492816145049012 | 2 |
| material_grouped | v6_2_a_frequency_fusion | Frequency + v6.2-A | random_forest |  | 0.9450017247513617 | 0.9757942905654546 | 0.9242067984304929 | wood | 0.9445749221125169 | carbon | 0.9454285273902064 | 2 |
| material_grouped | frequency_v6_2_a_spectral_fusion | Frequency + v6.2-A + spectral descriptors | balanced_logistic_regression |  | 0.9129855864952444 | 0.9615946531158986 | 0.9330691677476017 | wood | 0.8918544894827278 | carbon | 0.934116683507761 | 2 |
| material_grouped | frequency_v6_2_a_spectral_fusion | Frequency + v6.2-A + spectral descriptors | random_forest |  | 0.9463737481607151 | 0.9770088907196861 | 0.9276885405321466 | carbon | 0.9444966638869403 | wood | 0.9482508324344897 | 2 |

## Information-budget comparisons

- `frequency_only`: stratified 98.67%, Board LOBO 97.49%, Material LOMO 94.09%.
- `spectral_descriptors_only`: stratified 52.92%, Board LOBO 22.77%, Material LOMO 19.72%.
- `v6_2_a_embedding_only`: stratified 93.48%, Board LOBO 94.28%, Material LOMO 94.05%.
- `frequency_spectral_fusion`: stratified 98.53%, Board LOBO 93.46%, Material LOMO 73.33%.
- `v6_2_a_spectral_fusion`: stratified 93.48%, Board LOBO 94.35%, Material LOMO 94.00%.
- `v6_2_a_frequency_fusion`: stratified 96.75%, Board LOBO 94.79%, Material LOMO 94.50%.
- `frequency_v6_2_a_spectral_fusion`: stratified 96.41%, Board LOBO 94.97%, Material LOMO 94.64%.

## Required deltas

| protocol | comparison | left_macro_f1 | right_macro_f1 | delta_left_minus_right_pp |
| --- | --- | --- | --- | --- |
| stratified_train_val | spectral-only minus frequency-only | 0.529208462087193 | 0.9866861288039631 | -45.74776667167701 |
| stratified_train_val | frequency+spectral minus frequency-only | 0.9853435959740521 | 0.9866861288039631 | -0.13425328299110362 |
| stratified_train_val | v6.2-A+spectral minus v6.2-A | 0.934808068951973 | 0.9348100151719869 | -0.0001946220013926414 |
| stratified_train_val | frequency+v6.2-A+spectral minus frequency+v6.2-A | 0.9641356080871841 | 0.9674609129599487 | -0.3325304872764612 |
| board_grouped | spectral-only minus frequency-only | 0.22773641065070063 | 0.9748609579888732 | -74.71245473381725 |
| board_grouped | frequency+spectral minus frequency-only | 0.9346054422895418 | 0.9748609579888732 | -4.025551569933139 |
| board_grouped | v6.2-A+spectral minus v6.2-A | 0.9435480245622432 | 0.9428118238555557 | 0.0736200706687451 |
| board_grouped | frequency+v6.2-A+spectral minus frequency+v6.2-A | 0.949738643728922 | 0.9478550859511922 | 0.18835577777298607 |
| material_grouped | spectral-only minus frequency-only | 0.19720081570542955 | 0.9408561364311898 | -74.36553207257603 |
| material_grouped | frequency+spectral minus frequency-only | 0.7332843852810769 | 0.9408561364311898 | -20.757175115011293 |
| material_grouped | v6.2-A+spectral minus v6.2-A | 0.9400464998332079 | 0.9405498432710986 | -0.05033434378907131 |
| material_grouped | frequency+v6.2-A+spectral minus frequency+v6.2-A | 0.9463737481607151 | 0.9450017247513617 | 0.13720234093533845 |

## Board LOBO-style fold scores

| protocol | condition | method | held_out_group | accuracy | macro_recall | macro_f1 | n_test | stress_test_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| board_grouped | frequency_only | balanced_logistic_regression | C01 | 1.0 | 1.0 | 1.0 | 2733 | none |
| board_grouped | frequency_only | balanced_logistic_regression | C02 | 0.9918068132815869 | 0.973049645390071 | 0.9741399193870677 | 2319 | none |
| board_grouped | frequency_only | balanced_logistic_regression | C03 | 0.9932803909590715 | 0.9775510204081632 | 0.9791285162713734 | 1637 | none |
| board_grouped | frequency_only | balanced_logistic_regression | W01 | 0.9921455146754857 | 0.9794594594594596 | 0.9789861216134674 | 2419 | none |
| board_grouped | frequency_only | balanced_logistic_regression | W02 | 0.9852320675105485 | 0.9585798816568047 | 0.9549541161012653 | 2370 | none |
| board_grouped | frequency_only | balanced_logistic_regression | W03 | 0.9897680763983628 | 0.953125 | 0.9619570745600651 | 1466 | none |
| board_grouped | frequency_only | random_forest | C01 | 0.9915843395536041 | 0.9192982456140351 | 0.9234618488855777 | 2733 | none |
| board_grouped | frequency_only | random_forest | C02 | 0.9831824062095731 | 0.9446808510638298 | 0.945804917287448 | 2319 | none |
| board_grouped | frequency_only | random_forest | C03 | 0.9920586438607208 | 0.9734693877551021 | 0.9752660339373023 | 1637 | none |
| board_grouped | frequency_only | random_forest | W01 | 0.9925589086399339 | 0.9805405405405405 | 0.9800953079178886 | 2419 | none |
| board_grouped | frequency_only | random_forest | W02 | 0.9928270042194093 | 0.9798816568047337 | 0.9781124186253507 | 2370 | none |
| board_grouped | frequency_only | random_forest | W03 | 1.0 | 1.0 | 1.0 | 1466 | none |
| board_grouped | spectral_descriptors_only | balanced_logistic_regression | C01 | 0.48920600073179654 | 0.274415971149927 | 0.20844126880028346 | 2733 | none |
| board_grouped | spectral_descriptors_only | balanced_logistic_regression | C02 | 0.24536438119879259 | 0.3012008350739763 | 0.19788368407203524 | 2319 | none |
| board_grouped | spectral_descriptors_only | balanced_logistic_regression | C03 | 0.10384850335980451 | 0.2978822538757918 | 0.09712015823880564 | 1637 | none |
| board_grouped | spectral_descriptors_only | balanced_logistic_regression | W01 | 0.48160396858205873 | 0.44113807452776366 | 0.31396767801434294 | 2419 | none |
| board_grouped | spectral_descriptors_only | balanced_logistic_regression | W02 | 0.5751054852320675 | 0.31017642685000474 | 0.2822724521874248 | 2370 | none |
| board_grouped | spectral_descriptors_only | balanced_logistic_regression | W03 | 0.3178717598908595 | 0.305692961650596 | 0.20827042641489685 | 1466 | none |
| board_grouped | spectral_descriptors_only | random_forest | C01 | 0.8349798755945848 | 0.20098772775035192 | 0.18757087169434497 | 2733 | none |
| board_grouped | spectral_descriptors_only | random_forest | C02 | 0.723156532988357 | 0.2568641250488003 | 0.224177005789909 | 2319 | none |
| board_grouped | spectral_descriptors_only | random_forest | C03 | 0.7519853390348198 | 0.24967741423314926 | 0.25203732303732307 | 1637 | none |
| board_grouped | spectral_descriptors_only | random_forest | W01 | 0.7565109549400578 | 0.24182623422438132 | 0.23901951521291337 | 2419 | none |
| board_grouped | spectral_descriptors_only | random_forest | W02 | 0.7894514767932489 | 0.20155623629074956 | 0.1801384127390984 | 2370 | none |
| board_grouped | spectral_descriptors_only | random_forest | W03 | 0.7162346521145976 | 0.3130782014627893 | 0.2834753354306149 | 1466 | none |
| board_grouped | v6_2_a_embedding_only | balanced_logistic_regression | C01 | 0.977680204903037 | 0.9220966353150033 | 0.9179971537482736 | 2733 | none |
| board_grouped | v6_2_a_embedding_only | balanced_logistic_regression | C02 | 0.9340232858990944 | 0.9405644369439783 | 0.8837789055610414 | 2319 | none |
| board_grouped | v6_2_a_embedding_only | balanced_logistic_regression | C03 | 0.9725106902871106 | 0.9340430174494333 | 0.9476322666655153 | 1637 | none |
| board_grouped | v6_2_a_embedding_only | balanced_logistic_regression | W01 | 0.9396444811905746 | 0.9247875534378774 | 0.8916116827894383 | 2419 | none |
| board_grouped | v6_2_a_embedding_only | balanced_logistic_regression | W02 | 0.9729957805907173 | 0.9552685383942154 | 0.9304520813737189 | 2370 | none |
| board_grouped | v6_2_a_embedding_only | balanced_logistic_regression | W03 | 0.9570259208731241 | 0.9144541440219559 | 0.9186534204252983 | 1466 | none |
| board_grouped | v6_2_a_embedding_only | random_forest | C01 | 0.9842663739480424 | 0.9087318037988729 | 0.9335778806382031 | 2733 | none |
| board_grouped | v6_2_a_embedding_only | random_forest | C02 | 0.9646399310047434 | 0.9187196884535209 | 0.9325992413312496 | 2319 | none |
| board_grouped | v6_2_a_embedding_only | random_forest | C03 | 0.9780085522296884 | 0.9341717591336792 | 0.9602754585994809 | 1637 | none |
| board_grouped | v6_2_a_embedding_only | random_forest | W01 | 0.9669284828441504 | 0.9249024419924978 | 0.9339337890802281 | 2419 | none |
| board_grouped | v6_2_a_embedding_only | random_forest | W02 | 0.9814345991561182 | 0.9355251770340841 | 0.9574260950029168 | 2370 | none |
| board_grouped | v6_2_a_embedding_only | random_forest | W03 | 0.9720327421555253 | 0.9277653694493738 | 0.9390584784812568 | 1466 | none |
| board_grouped | frequency_spectral_fusion | balanced_logistic_regression | C01 | 0.995609220636663 | 0.9788729531338227 | 0.9707111814978099 | 2733 | none |
| board_grouped | frequency_spectral_fusion | balanced_logistic_regression | C02 | 0.9793014230271668 | 0.9387140403972374 | 0.9388058660500065 | 2319 | none |
| board_grouped | frequency_spectral_fusion | balanced_logistic_regression | C03 | 0.9718998167379352 | 0.9163052483846391 | 0.9092640997065976 | 1637 | none |
| board_grouped | frequency_spectral_fusion | balanced_logistic_regression | W01 | 0.9706490285241836 | 0.9366039414549998 | 0.929731559112063 | 2419 | none |
| board_grouped | frequency_spectral_fusion | balanced_logistic_regression | W02 | 0.9763713080168777 | 0.9278598448018816 | 0.9247053537826471 | 2370 | none |
| board_grouped | frequency_spectral_fusion | balanced_logistic_regression | W03 | 0.980218281036835 | 0.9354440413223299 | 0.9344145935881268 | 1466 | none |
| board_grouped | frequency_spectral_fusion | random_forest | C01 | 0.9754848152213684 | 0.8482285928725558 | 0.867970868310751 | 2733 | none |
| board_grouped | frequency_spectral_fusion | random_forest | C02 | 0.9150495903406641 | 0.7333155758677813 | 0.7901726591659781 | 2319 | none |
| board_grouped | frequency_spectral_fusion | random_forest | C03 | 0.9297495418448382 | 0.7894993894993895 | 0.8391493276036055 | 1637 | none |
| board_grouped | frequency_spectral_fusion | random_forest | W01 | 0.9830508474576272 | 0.9540490913413237 | 0.9646738997442134 | 2419 | none |
| board_grouped | frequency_spectral_fusion | random_forest | W02 | 0.9054852320675105 | 0.7172640492902431 | 0.7551064756909031 | 2370 | none |
| board_grouped | frequency_spectral_fusion | random_forest | W03 | 0.9815825375170532 | 0.9321138211382113 | 0.955127843192488 | 1466 | none |
| board_grouped | v6_2_a_spectral_fusion | balanced_logistic_regression | C01 | 0.9721917306988657 | 0.9198023809733857 | 0.8993160419668849 | 2733 | none |
| board_grouped | v6_2_a_spectral_fusion | balanced_logistic_regression | C02 | 0.9322984044846917 | 0.9341524881116776 | 0.8776581617857986 | 2319 | none |
| board_grouped | v6_2_a_spectral_fusion | balanced_logistic_regression | C03 | 0.9511301160659743 | 0.9310362818555756 | 0.9094818806765457 | 1637 | none |
| board_grouped | v6_2_a_spectral_fusion | balanced_logistic_regression | W01 | 0.9334435717238528 | 0.9161651901257872 | 0.8766442739497927 | 2419 | none |
| board_grouped | v6_2_a_spectral_fusion | balanced_logistic_regression | W02 | 0.950210970464135 | 0.9321915214656276 | 0.878013301098598 | 2370 | none |
| board_grouped | v6_2_a_spectral_fusion | balanced_logistic_regression | W03 | 0.9454297407912687 | 0.9152880660061016 | 0.9014539095283609 | 1466 | none |
| board_grouped | v6_2_a_spectral_fusion | random_forest | C01 | 0.9849981705085986 | 0.9123275322504366 | 0.9378499810309217 | 2733 | none |
| board_grouped | v6_2_a_spectral_fusion | random_forest | C02 | 0.9659335920655455 | 0.9202930235136364 | 0.9357630197795693 | 2319 | none |
| board_grouped | v6_2_a_spectral_fusion | random_forest | C03 | 0.9773976786805131 | 0.9324623574242775 | 0.9592429329354175 | 1637 | none |
| board_grouped | v6_2_a_spectral_fusion | random_forest | W01 | 0.9677552707730467 | 0.9253369035301043 | 0.9338681219250358 | 2419 | none |
| board_grouped | v6_2_a_spectral_fusion | random_forest | W02 | 0.9810126582278481 | 0.935193320675684 | 0.9566629448555355 | 2370 | none |
| board_grouped | v6_2_a_spectral_fusion | random_forest | W03 | 0.970668485675307 | 0.9293684618283079 | 0.9379011468469788 | 1466 | none |
| board_grouped | v6_2_a_frequency_fusion | balanced_logistic_regression | C01 | 0.9886571533113795 | 0.9591578337819373 | 0.9420947968001997 | 2733 | none |
| board_grouped | v6_2_a_frequency_fusion | balanced_logistic_regression | C02 | 0.9586028460543338 | 0.9491607681369189 | 0.9201529293819523 | 2319 | none |
| board_grouped | v6_2_a_frequency_fusion | balanced_logistic_regression | C03 | 0.9816737935247404 | 0.946299473416944 | 0.9611342041589858 | 1637 | none |
| board_grouped | v6_2_a_frequency_fusion | balanced_logistic_regression | W01 | 0.9677552707730467 | 0.9358907505724723 | 0.9254139106972843 | 2419 | none |
| board_grouped | v6_2_a_frequency_fusion | balanced_logistic_regression | W02 | 0.9890295358649789 | 0.9634729476955911 | 0.9635238020709849 | 2370 | none |
| board_grouped | v6_2_a_frequency_fusion | balanced_logistic_regression | W03 | 0.9774897680763983 | 0.9399838136979204 | 0.9419772202362197 | 1466 | none |
| board_grouped | v6_2_a_frequency_fusion | random_forest | C01 | 0.9849981705085986 | 0.9123275322504366 | 0.9378499810309217 | 2733 | none |
| board_grouped | v6_2_a_frequency_fusion | random_forest | C02 | 0.9702457956015524 | 0.9294062020593504 | 0.9431757060599042 | 2319 | none |
| board_grouped | v6_2_a_frequency_fusion | random_forest | C03 | 0.9780085522296884 | 0.9341717591336792 | 0.9602754585994809 | 1637 | none |
| board_grouped | v6_2_a_frequency_fusion | random_forest | W01 | 0.9706490285241836 | 0.9281535444636082 | 0.9424284913338485 | 2419 | none |
| board_grouped | v6_2_a_frequency_fusion | random_forest | W02 | 0.9822784810126582 | 0.9368154465935845 | 0.9576969134911228 | 2370 | none |
| board_grouped | v6_2_a_frequency_fusion | random_forest | W03 | 0.9761255115961801 | 0.9337708340910817 | 0.9457039651918748 | 1466 | none |
| board_grouped | frequency_v6_2_a_spectral_fusion | balanced_logistic_regression | C01 | 0.9871935601902672 | 0.9389844501978033 | 0.9270713108299233 | 2733 | none |
| board_grouped | frequency_v6_2_a_spectral_fusion | balanced_logistic_regression | C02 | 0.9577404053471323 | 0.9487167640611285 | 0.919145351747348 | 2319 | none |
| board_grouped | frequency_v6_2_a_spectral_fusion | balanced_logistic_regression | C03 | 0.9731215638362859 | 0.9571926190435306 | 0.9373249311528629 | 1637 | none |
| board_grouped | frequency_v6_2_a_spectral_fusion | balanced_logistic_regression | W01 | 0.9710624224886316 | 0.9334987551224831 | 0.9273891783043415 | 2419 | none |
| board_grouped | frequency_v6_2_a_spectral_fusion | balanced_logistic_regression | W02 | 0.979746835443038 | 0.9599679821917093 | 0.9428176168976012 | 2370 | none |
| board_grouped | frequency_v6_2_a_spectral_fusion | balanced_logistic_regression | W03 | 0.9693042291950886 | 0.9414959595263482 | 0.9307411590000427 | 1466 | none |
| board_grouped | frequency_v6_2_a_spectral_fusion | random_forest | C01 | 0.9849981705085986 | 0.9123275322504366 | 0.9378499810309217 | 2733 | none |
| board_grouped | frequency_v6_2_a_spectral_fusion | random_forest | C02 | 0.9724018973695558 | 0.9282228739265899 | 0.946493123421512 | 2319 | none |
| board_grouped | frequency_v6_2_a_spectral_fusion | random_forest | C03 | 0.9798411728772144 | 0.9346564117992688 | 0.9632054872969553 | 1637 | none |
| board_grouped | frequency_v6_2_a_spectral_fusion | random_forest | W01 | 0.972302604381976 | 0.9334498986149244 | 0.9443141260531611 | 2419 | none |
| board_grouped | frequency_v6_2_a_spectral_fusion | random_forest | W02 | 0.9827004219409282 | 0.9393004916436348 | 0.9590973353805726 | 2370 | none |
| board_grouped | frequency_v6_2_a_spectral_fusion | random_forest | W03 | 0.9781718963165075 | 0.931375761098381 | 0.9474718091904089 | 1466 | none |

## Material LOMO-style fold scores

| protocol | condition | method | held_out_group | accuracy | macro_recall | macro_f1 | n_test | stress_test_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| material_grouped | frequency_only | balanced_logistic_regression | carbon | 0.987143070713111 | 0.9418918918918919 | 0.9441804939708274 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | frequency_only | balanced_logistic_regression | wood | 0.9792166266986411 | 0.937799043062201 | 0.9375317788915524 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | frequency_only | random_forest | carbon | 0.987143070713111 | 0.9418918918918919 | 0.9441804939708274 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | frequency_only | random_forest | wood | 0.9792166266986411 | 0.937799043062201 | 0.9375317788915524 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | spectral_descriptors_only | balanced_logistic_regression | carbon | 0.13006428464643444 | 0.1567002052484633 | 0.09045090747611614 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | spectral_descriptors_only | balanced_logistic_regression | wood | 0.10727418065547561 | 0.21079205123198919 | 0.08767471996615621 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | spectral_descriptors_only | random_forest | carbon | 0.7271640005979967 | 0.21698885369450155 | 0.19924449433785457 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | spectral_descriptors_only | random_forest | wood | 0.7611510791366907 | 0.21189414855424116 | 0.19515713707300453 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | v6_2_a_embedding_only | balanced_logistic_regression | carbon | 0.9693526685603229 | 0.9304735644669779 | 0.932156897615069 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | v6_2_a_embedding_only | balanced_logistic_regression | wood | 0.9443645083932853 | 0.9283298973657704 | 0.8899964004860413 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | v6_2_a_embedding_only | random_forest | carbon | 0.9753326356705038 | 0.9172216823288618 | 0.9433040380458856 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | v6_2_a_embedding_only | random_forest | wood | 0.9705835331734612 | 0.9265896541546661 | 0.9377956484963115 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | frequency_spectral_fusion | balanced_logistic_regression | carbon | 0.8407833756914337 | 0.6743337187309338 | 0.6263682554748604 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | frequency_spectral_fusion | balanced_logistic_regression | wood | 0.9338129496402877 | 0.8065500778955451 | 0.796870692510475 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | frequency_spectral_fusion | random_forest | carbon | 0.9186724473015399 | 0.6737348627798065 | 0.7095624614304572 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | frequency_spectral_fusion | random_forest | wood | 0.9146282973621103 | 0.7462798403981799 | 0.7570063091316964 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | v6_2_a_spectral_fusion | balanced_logistic_regression | carbon | 0.9521602631185528 | 0.9228753142087374 | 0.8965897996769512 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | v6_2_a_spectral_fusion | balanced_logistic_regression | wood | 0.9139888089528377 | 0.920706616811539 | 0.8388886048778372 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | v6_2_a_spectral_fusion | random_forest | carbon | 0.9751831364927492 | 0.9188527775402113 | 0.9436965596519912 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | v6_2_a_spectral_fusion | random_forest | wood | 0.9710631494804157 | 0.9264787852782611 | 0.9363964400144245 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | v6_2_a_frequency_fusion | balanced_logistic_regression | carbon | 0.9790701151143669 | 0.9406055022388872 | 0.9492816145049012 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | v6_2_a_frequency_fusion | balanced_logistic_regression | wood | 0.9702637889688249 | 0.9502589096916652 | 0.9312595914006636 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | v6_2_a_frequency_fusion | random_forest | carbon | 0.97652862909254 | 0.9187266272100745 | 0.9454285273902064 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | v6_2_a_frequency_fusion | random_forest | wood | 0.9750599520383693 | 0.9296869696509112 | 0.9445749221125169 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | frequency_v6_2_a_spectral_fusion | balanced_logistic_regression | carbon | 0.973389146359695 | 0.9282206934671601 | 0.934116683507761 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | frequency_v6_2_a_spectral_fusion | balanced_logistic_regression | wood | 0.9498001598721023 | 0.9379176420280434 | 0.8918544894827278 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | frequency_v6_2_a_spectral_fusion | random_forest | carbon | 0.9760801315592764 | 0.9192181212818227 | 0.9444966638869403 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | frequency_v6_2_a_spectral_fusion | random_forest | wood | 0.9779376498800959 | 0.9361589597824704 | 0.9482508324344897 | 6255 | descriptive stress-test only; two material groups available |

## Per-class changes

| protocol | label_name | frequency_only_f1 | spectral_only_f1 | v6_2_a_embedding_only_f1 | frequency_v6_2_a_f1 | frequency_v6_2_a_spectral_f1 | spectral_minus_frequency_pp | spectral_added_to_frequency_v6_2_a_pp | minority_focus |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stratified_train_val | 1_1H | 1.0 | 0.5177664974619289 | 0.9416342412451363 | 0.9885931558935361 | 0.9846153846153846 | -48.223350253807105 | -0.3977771278151576 | False |
| stratified_train_val | 1_1T | 1.0 | 0.5615763546798028 | 0.9578544061302682 | 0.9814126394052044 | 0.9887640449438201 | -43.842364532019715 | 0.7351405538615685 | False |
| stratified_train_val | 1_2 | 0.9652777777777778 | 0.24858757062146888 | 0.8919860627177699 | 0.9315068493150684 | 0.9256756756756758 | -71.6690207156309 | -0.5831173639392673 | True |
| stratified_train_val | 2_1 | 0.9681528662420381 | 0.4036697247706422 | 0.8963210702341137 | 0.9385113268608415 | 0.9250814332247557 | -56.44831414713959 | -1.3429893636085821 | True |
| stratified_train_val | higher | 1.0 | 0.9144421629021219 | 0.9862542955326461 | 0.9972805933250928 | 0.9965415019762845 | -8.555783709787812 | -0.07390913488082296 | False |
| board_grouped | 1_1H | 1.0 | 0.16747271997836377 | 0.9422061753166808 | 0.9513272895309101 | 0.9554365953222295 | -83.25272800216362 | 0.41093057913194064 | False |
| board_grouped | 1_1T | 1.0 | 0.09201272473064241 | 0.9525833166543101 | 0.9552572848232453 | 0.9554168421835784 | -90.79872752693576 | 0.015955736033301804 | False |
| board_grouped | 1_2 | 0.9416949540041588 | 0.0026666666666666666 | 0.9122589709448518 | 0.9226108895517745 | 0.9256195158231542 | -93.90282873374922 | 0.3008626271379655 | True |
| board_grouped | 2_1 | 0.932609835940207 | 0.005050505050505051 | 0.9212001694505032 | 0.9229104521979951 | 0.9240380606142834 | -92.75593308897018 | 0.11276084162883393 | True |
| board_grouped | higher | 1.0 | 0.8714794368273253 | 0.9858104869114332 | 0.9871695136520359 | 0.9881822047013643 | -12.852056317267468 | 0.10126910493284402 | False |
| material_grouped | 1_1H | 1.0 | 0.06795662170135709 | 0.9409652584589507 | 0.9456326374873232 | 0.9471100609032665 | -93.20433782986429 | 0.14774234159433686 | False |
| material_grouped | 1_1T | 1.0 | 0.050966608084358524 | 0.9513856502875878 | 0.9561027480683592 | 0.9552395810329803 | -94.90333919156416 | -0.0863167035378809 | False |
| material_grouped | 1_2 | 0.8509171987432858 | 0.00447427293064877 | 0.9109282245008095 | 0.9152276295133437 | 0.9184020275884992 | -84.6442925812637 | 0.31743980751555156 | True |
| material_grouped | 2_1 | 0.8533634834126631 | 0.0 | 0.9142175049198309 | 0.9207021791767553 | 0.9225815326130452 | -85.33634834126632 | 0.18793534362898656 | True |
| material_grouped | higher | 1.0 | 0.8626065758107833 | 0.9852525781883137 | 0.9873434295110274 | 0.9885355386657837 | -13.739342418921673 | 0.11921091547563156 | False |

## Minority-class focus: 1_2 and 2_1

| protocol | label_name | frequency_only_f1 | spectral_only_f1 | v6_2_a_embedding_only_f1 | frequency_v6_2_a_f1 | frequency_v6_2_a_spectral_f1 | spectral_minus_frequency_pp | spectral_added_to_frequency_v6_2_a_pp | minority_focus |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stratified_train_val | 1_2 | 0.9652777777777778 | 0.24858757062146888 | 0.8919860627177699 | 0.9315068493150684 | 0.9256756756756758 | -71.6690207156309 | -0.5831173639392673 | True |
| stratified_train_val | 2_1 | 0.9681528662420381 | 0.4036697247706422 | 0.8963210702341137 | 0.9385113268608415 | 0.9250814332247557 | -56.44831414713959 | -1.3429893636085821 | True |
| board_grouped | 1_2 | 0.9416949540041588 | 0.0026666666666666666 | 0.9122589709448518 | 0.9226108895517745 | 0.9256195158231542 | -93.90282873374922 | 0.3008626271379655 | True |
| board_grouped | 2_1 | 0.932609835940207 | 0.005050505050505051 | 0.9212001694505032 | 0.9229104521979951 | 0.9240380606142834 | -92.75593308897018 | 0.11276084162883393 | True |
| material_grouped | 1_2 | 0.8509171987432858 | 0.00447427293064877 | 0.9109282245008095 | 0.9152276295133437 | 0.9184020275884992 | -84.6442925812637 | 0.31743980751555156 | True |
| material_grouped | 2_1 | 0.8533634834126631 | 0.0 | 0.9142175049198309 | 0.9207021791767553 | 0.9225815326130452 | -85.33634834126632 | 0.18793534362898656 | True |

## Zero-recall rows

- Zero-recall rows: `16`.

## Decision

- Recommendation: keep as future work / negative control; deterministic descriptors do not provide clear matched gains over the stronger controls.
- Include in the main paper only if the table is framed as a deterministic spectral descriptor control.
- Do not claim LeFFT superiority unless future matched experiments show clear, robust gains.

## Claim boundary

- Deterministic descriptors only.
- Not a trained LeFFT model.
- No LeFFT superiority claim from this evaluation.
- No acoustic-response claim.
- Material LOMO remains a descriptive stress test because only two material groups are available.