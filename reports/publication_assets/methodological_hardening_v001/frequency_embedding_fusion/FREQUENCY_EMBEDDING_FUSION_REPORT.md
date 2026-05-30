# Frequency + Embedding Information-Budget Comparison

## Purpose

This report evaluates whether frozen ESPI embeddings add predictive information beyond `frequency_hz`. It uses only normalized feature dumps and metadata. No raw images, deep-model inference, encoder fine-tuning, LeFFT evaluation, or acoustic-response prediction is performed.

## Conditions

- `frequency_only`: Frequency only (metadata_only)
- `v6_1_embedding_only`: v6.1 embedding only (embedding_only)
- `v6_1_frequency_fusion`: Frequency + v6.1 embedding (frequency_embedding_fusion)
- `v6_2_a_embedding_only`: v6.2-A embedding only (embedding_only)
- `v6_2_a_frequency_fusion`: Frequency + v6.2-A embedding (frequency_embedding_fusion)
- `random_resnet18_embedding_only`: Random ResNet-18 embedding only (embedding_only)
- `random_resnet18_frequency_fusion`: Frequency + Random ResNet-18 embedding (frequency_embedding_fusion)
- `imagenet_resnet18_embedding_only`: ImageNet ResNet-18 embedding only (embedding_only)
- `imagenet_resnet18_frequency_fusion`: Frequency + ImageNet ResNet-18 embedding (frequency_embedding_fusion)

## Models and Preprocessing

- `balanced_logistic_regression`: Balanced logistic regression on train-fold standardized inputs. Parameters: `{"class_weight": "balanced", "max_iter": 5000, "random_state": 42, "preprocessing": "train/reference-fold standardization"}`
- `random_forest`: Random forest control on train-fold standardized inputs. Parameters: `{"n_estimators": 80, "class_weight": "balanced_subsample", "min_samples_leaf": 2, "max_features": "sqrt", "random_state": 42, "preprocessing": "train/reference-fold standardization"}`

All input features, including `frequency_hz` and frozen embeddings, are standardized using the training/reference fold only and then applied to the held-out fold. Grouped held-out samples are never used to fit preprocessing.

## Summary Metrics

| protocol | condition | display_name | method | macro_f1 | mean_macro_f1 | worst_group | worst_group_macro_f1 | best_group | best_group_macro_f1 | n_groups |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stratified_train_val | frequency_only | Frequency only | balanced_logistic_regression | 0.9694338664635694 |  |  |  |  |  |  |
| stratified_train_val | frequency_only | Frequency only | random_forest | 0.9866861288039631 |  |  |  |  |  |  |
| stratified_train_val | v6_1_embedding_only | v6.1 embedding only | balanced_logistic_regression | 0.9272965397955483 |  |  |  |  |  |  |
| stratified_train_val | v6_1_embedding_only | v6.1 embedding only | random_forest | 0.950836357647718 |  |  |  |  |  |  |
| stratified_train_val | v6_1_frequency_fusion | Frequency + v6.1 embedding | balanced_logistic_regression | 0.9786365661023277 |  |  |  |  |  |  |
| stratified_train_val | v6_1_frequency_fusion | Frequency + v6.1 embedding | random_forest | 0.972254452330177 |  |  |  |  |  |  |
| stratified_train_val | v6_2_a_embedding_only | v6.2-A embedding only | balanced_logistic_regression | 0.9260198668851682 |  |  |  |  |  |  |
| stratified_train_val | v6_2_a_embedding_only | v6.2-A embedding only | random_forest | 0.9348100151719869 |  |  |  |  |  |  |
| stratified_train_val | v6_2_a_frequency_fusion | Frequency + v6.2-A embedding | balanced_logistic_regression | 0.9674609129599487 |  |  |  |  |  |  |
| stratified_train_val | v6_2_a_frequency_fusion | Frequency + v6.2-A embedding | random_forest | 0.9403228145499579 |  |  |  |  |  |  |
| stratified_train_val | random_resnet18_embedding_only | Random ResNet-18 embedding only | balanced_logistic_regression | 0.844113949749216 |  |  |  |  |  |  |
| stratified_train_val | random_resnet18_embedding_only | Random ResNet-18 embedding only | random_forest | 0.6054303800150843 |  |  |  |  |  |  |
| stratified_train_val | random_resnet18_frequency_fusion | Frequency + Random ResNet-18 embedding | balanced_logistic_regression | 0.9554643920733421 |  |  |  |  |  |  |
| stratified_train_val | random_resnet18_frequency_fusion | Frequency + Random ResNet-18 embedding | random_forest | 0.8208181216344632 |  |  |  |  |  |  |
| stratified_train_val | imagenet_resnet18_embedding_only | ImageNet ResNet-18 embedding only | balanced_logistic_regression | 0.8125247157987667 |  |  |  |  |  |  |
| stratified_train_val | imagenet_resnet18_embedding_only | ImageNet ResNet-18 embedding only | random_forest | 0.707981047203335 |  |  |  |  |  |  |
| stratified_train_val | imagenet_resnet18_frequency_fusion | Frequency + ImageNet ResNet-18 embedding | balanced_logistic_regression | 0.9028129533785243 |  |  |  |  |  |  |
| stratified_train_val | imagenet_resnet18_frequency_fusion | Frequency + ImageNet ResNet-18 embedding | random_forest | 0.8762695402875735 |  |  |  |  |  |  |
| board_grouped | frequency_only | Frequency only | balanced_logistic_regression |  | 0.9748609579888732 | W02 | 0.9549541161012653 | C01 | 1.0 | 6 |
| board_grouped | frequency_only | Frequency only | random_forest |  | 0.9671234211089278 | C01 | 0.9234618488855777 | W03 | 1.0 | 6 |
| board_grouped | v6_1_embedding_only | v6.1 embedding only | balanced_logistic_regression |  | 0.7967436199456173 | C03 | 0.6893785163522168 | C02 | 0.8801881884930796 | 6 |
| board_grouped | v6_1_embedding_only | v6.1 embedding only | random_forest |  | 0.8908803730403659 | C01 | 0.8431928677724543 | C02 | 0.9448686742033556 | 6 |
| board_grouped | v6_1_frequency_fusion | Frequency + v6.1 embedding | balanced_logistic_regression |  | 0.8681133909658709 | C01 | 0.7351389331459898 | C02 | 0.9810971230344524 | 6 |
| board_grouped | v6_1_frequency_fusion | Frequency + v6.1 embedding | random_forest |  | 0.92790139185409 | W02 | 0.8953282533997233 | C02 | 0.9777045443430762 | 6 |
| board_grouped | v6_2_a_embedding_only | v6.2-A embedding only | balanced_logistic_regression |  | 0.9150209184272143 | C02 | 0.8837789055610414 | C03 | 0.9476322666655153 | 6 |
| board_grouped | v6_2_a_embedding_only | v6.2-A embedding only | random_forest |  | 0.9428118238555557 | C02 | 0.9325992413312496 | C03 | 0.9602754585994809 | 6 |
| board_grouped | v6_2_a_frequency_fusion | Frequency + v6.2-A embedding | balanced_logistic_regression |  | 0.9423828105576045 | C02 | 0.9201529293819523 | W02 | 0.9635238020709849 | 6 |
| board_grouped | v6_2_a_frequency_fusion | Frequency + v6.2-A embedding | random_forest |  | 0.9478550859511922 | C01 | 0.9378499810309217 | C03 | 0.9602754585994809 | 6 |
| board_grouped | random_resnet18_embedding_only | Random ResNet-18 embedding only | balanced_logistic_regression |  | 0.3217028762048134 | W03 | 0.23281724837195306 | C01 | 0.41340233538526217 | 6 |
| board_grouped | random_resnet18_embedding_only | Random ResNet-18 embedding only | random_forest |  | 0.1883303415323974 | W03 | 0.17111631537861044 | C02 | 0.21857391799177858 | 6 |
| board_grouped | random_resnet18_frequency_fusion | Frequency + Random ResNet-18 embedding | balanced_logistic_regression |  | 0.5325131603198028 | W03 | 0.3767301843357092 | C01 | 0.7237956566119431 | 6 |
| board_grouped | random_resnet18_frequency_fusion | Frequency + Random ResNet-18 embedding | random_forest |  | 0.324002011104011 | W02 | 0.23159684595392274 | C01 | 0.4952118526828757 | 6 |
| board_grouped | imagenet_resnet18_embedding_only | ImageNet ResNet-18 embedding only | balanced_logistic_regression |  | 0.3793798576179885 | C01 | 0.28115926372148936 | W01 | 0.46568154148210644 | 6 |
| board_grouped | imagenet_resnet18_embedding_only | ImageNet ResNet-18 embedding only | random_forest |  | 0.2741808408972665 | W02 | 0.17652050919377651 | C03 | 0.32796964413906265 | 6 |
| board_grouped | imagenet_resnet18_frequency_fusion | Frequency + ImageNet ResNet-18 embedding | balanced_logistic_regression |  | 0.5250477912621878 | C01 | 0.42713127201725376 | C02 | 0.5870908760877936 | 6 |
| board_grouped | imagenet_resnet18_frequency_fusion | Frequency + ImageNet ResNet-18 embedding | random_forest |  | 0.42819368806174024 | W02 | 0.3076376619622567 | W01 | 0.504229037087876 | 6 |
| material_grouped | frequency_only | Frequency only | balanced_logistic_regression |  | 0.9408561364311898 | wood | 0.9375317788915524 | carbon | 0.9441804939708274 | 2 |
| material_grouped | frequency_only | Frequency only | random_forest |  | 0.9408561364311898 | wood | 0.9375317788915524 | carbon | 0.9441804939708274 | 2 |
| material_grouped | v6_1_embedding_only | v6.1 embedding only | balanced_logistic_regression |  | 0.6848743148193763 | carbon | 0.6526256877229313 | wood | 0.7171229419158214 | 2 |
| material_grouped | v6_1_embedding_only | v6.1 embedding only | random_forest |  | 0.8776757198710069 | wood | 0.8601791767154214 | carbon | 0.8951722630265924 | 2 |
| material_grouped | v6_1_frequency_fusion | Frequency + v6.1 embedding | balanced_logistic_regression |  | 0.7971296190574347 | carbon | 0.715106821884889 | wood | 0.8791524162299804 | 2 |
| material_grouped | v6_1_frequency_fusion | Frequency + v6.1 embedding | random_forest |  | 0.9019834473671781 | wood | 0.8897739710508523 | carbon | 0.9141929236835038 | 2 |
| material_grouped | v6_2_a_embedding_only | v6.2-A embedding only | balanced_logistic_regression |  | 0.9110766490505552 | wood | 0.8899964004860413 | carbon | 0.932156897615069 | 2 |
| material_grouped | v6_2_a_embedding_only | v6.2-A embedding only | random_forest |  | 0.9405498432710986 | wood | 0.9377956484963115 | carbon | 0.9433040380458856 | 2 |
| material_grouped | v6_2_a_frequency_fusion | Frequency + v6.2-A embedding | balanced_logistic_regression |  | 0.9402706029527824 | wood | 0.9312595914006636 | carbon | 0.9492816145049012 | 2 |
| material_grouped | v6_2_a_frequency_fusion | Frequency + v6.2-A embedding | random_forest |  | 0.9450017247513617 | wood | 0.9445749221125169 | carbon | 0.9454285273902064 | 2 |
| material_grouped | random_resnet18_embedding_only | Random ResNet-18 embedding only | balanced_logistic_regression |  | 0.2544191480306869 | carbon | 0.23668563042104723 | wood | 0.2721526656403265 | 2 |
| material_grouped | random_resnet18_embedding_only | Random ResNet-18 embedding only | random_forest |  | 0.18449083903647479 | wood | 0.17513498818618473 | carbon | 0.19384668988676484 | 2 |
| material_grouped | random_resnet18_frequency_fusion | Frequency + Random ResNet-18 embedding | balanced_logistic_regression |  | 0.38349269265029023 | carbon | 0.3733124628200365 | wood | 0.393672922480544 | 2 |
| material_grouped | random_resnet18_frequency_fusion | Frequency + Random ResNet-18 embedding | random_forest |  | 0.30368352788931763 | wood | 0.24218456819867132 | carbon | 0.36518248757996397 | 2 |
| material_grouped | imagenet_resnet18_embedding_only | ImageNet ResNet-18 embedding only | balanced_logistic_regression |  | 0.23015469928263663 | carbon | 0.16826248628727253 | wood | 0.2920469122780007 | 2 |
| material_grouped | imagenet_resnet18_embedding_only | ImageNet ResNet-18 embedding only | random_forest |  | 0.23708601442737104 | wood | 0.18575991849811518 | carbon | 0.2884121103566269 | 2 |
| material_grouped | imagenet_resnet18_frequency_fusion | Frequency + ImageNet ResNet-18 embedding | balanced_logistic_regression |  | 0.2903052932569818 | carbon | 0.24956617329793582 | wood | 0.3310444132160278 | 2 |
| material_grouped | imagenet_resnet18_frequency_fusion | Frequency + ImageNet ResNet-18 embedding | random_forest |  | 0.39313875102273405 | wood | 0.2967954746728057 | carbon | 0.4894820273726624 | 2 |

## Fusion Deltas

| protocol | encoder_name | frequency_only_method | frequency_only_macro_f1 | embedding_only_method | embedding_only_macro_f1 | fusion_method | fusion_macro_f1 | delta_fusion_minus_frequency_pp | delta_fusion_minus_embedding_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stratified_train_val | v6.1 | random_forest | 0.9866861288039631 | random_forest | 0.950836357647718 | balanced_logistic_regression | 0.9786365661023277 | -0.8049562701635371 | 2.780020845460973 |
| board_grouped | v6.1 | balanced_logistic_regression | 0.9748609579888732 | random_forest | 0.8908803730403659 | random_forest | 0.92790139185409 | -4.695956613478314 | 3.7021018813724105 |
| material_grouped | v6.1 | balanced_logistic_regression | 0.9408561364311898 | random_forest | 0.8776757198710069 | random_forest | 0.9019834473671781 | -3.8872689064011756 | 2.430772749617116 |
| stratified_train_val | v6.2-A | random_forest | 0.9866861288039631 | random_forest | 0.9348100151719869 | balanced_logistic_regression | 0.9674609129599487 | -1.9225215844014376 | 3.265089778796182 |
| board_grouped | v6.2-A | balanced_logistic_regression | 0.9748609579888732 | random_forest | 0.9428118238555557 | random_forest | 0.9478550859511922 | -2.700587203768101 | 0.5043262095636436 |
| material_grouped | v6.2-A | balanced_logistic_regression | 0.9408561364311898 | random_forest | 0.9405498432710986 | random_forest | 0.9450017247513617 | 0.4145588320171867 | 0.4451881480263098 |
| stratified_train_val | Random ResNet-18 | random_forest | 0.9866861288039631 | balanced_logistic_regression | 0.844113949749216 | balanced_logistic_regression | 0.9554643920733421 | -3.122173673062101 | 11.135044232412605 |
| board_grouped | Random ResNet-18 | balanced_logistic_regression | 0.9748609579888732 | balanced_logistic_regression | 0.3217028762048134 | balanced_logistic_regression | 0.5325131603198028 | -44.23477976690704 | 21.081028411498938 |
| material_grouped | Random ResNet-18 | balanced_logistic_regression | 0.9408561364311898 | balanced_logistic_regression | 0.2544191480306869 | balanced_logistic_regression | 0.38349269265029023 | -55.73634437808997 | 12.907354461960335 |
| stratified_train_val | ImageNet ResNet-18 | random_forest | 0.9866861288039631 | balanced_logistic_regression | 0.8125247157987667 | balanced_logistic_regression | 0.9028129533785243 | -8.387317542543881 | 9.028823757975758 |
| board_grouped | ImageNet ResNet-18 | balanced_logistic_regression | 0.9748609579888732 | balanced_logistic_regression | 0.3793798576179885 | balanced_logistic_regression | 0.5250477912621878 | -44.981316672668534 | 14.566793364419933 |
| material_grouped | ImageNet ResNet-18 | balanced_logistic_regression | 0.9408561364311898 | random_forest | 0.23708601442737104 | random_forest | 0.39313875102273405 | -54.77173854084558 | 15.605273659536302 |

## Board LOBO-Style Fold Scores

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
| board_grouped | v6_1_embedding_only | balanced_logistic_regression | C01 | 0.9462129527991219 | 0.7768232814995752 | 0.712211305287897 | 2733 | none |
| board_grouped | v6_1_embedding_only | balanced_logistic_regression | C02 | 0.9279862009486848 | 0.9492352896019829 | 0.8801881884930796 | 2319 | none |
| board_grouped | v6_1_embedding_only | balanced_logistic_regression | C03 | 0.8802687843616371 | 0.7599419819359815 | 0.6893785163522168 | 1637 | none |
| board_grouped | v6_1_embedding_only | balanced_logistic_regression | W01 | 0.9326167837949566 | 0.8907247704445004 | 0.8379249525685066 | 2419 | none |
| board_grouped | v6_1_embedding_only | balanced_logistic_regression | W02 | 0.9248945147679325 | 0.8632764776238109 | 0.8112352677730685 | 2370 | none |
| board_grouped | v6_1_embedding_only | balanced_logistic_regression | W03 | 0.9379263301500682 | 0.8785957730883525 | 0.8495234891989355 | 1466 | none |
| board_grouped | v6_1_embedding_only | random_forest | C01 | 0.9710940358580314 | 0.8593255145184135 | 0.8431928677724543 | 2733 | none |
| board_grouped | v6_1_embedding_only | random_forest | C02 | 0.9741267787839586 | 0.9296194975973646 | 0.9448686742033556 | 2319 | none |
| board_grouped | v6_1_embedding_only | random_forest | C03 | 0.9627367135003054 | 0.8842515699658557 | 0.929591686149853 | 1637 | none |
| board_grouped | v6_1_embedding_only | random_forest | W01 | 0.9599007854485324 | 0.882530875697792 | 0.9085693878949878 | 2419 | none |
| board_grouped | v6_1_embedding_only | random_forest | W02 | 0.9476793248945148 | 0.8254765158262556 | 0.8525086094674362 | 2370 | none |
| board_grouped | v6_1_embedding_only | random_forest | W03 | 0.9481582537517054 | 0.8686216370331232 | 0.8665510127541088 | 1466 | none |
| board_grouped | v6_1_frequency_fusion | balanced_logistic_regression | C01 | 0.9623124771313575 | 0.7974202082746974 | 0.7351389331459898 | 2733 | none |
| board_grouped | v6_1_frequency_fusion | balanced_logistic_regression | C02 | 0.9922380336351876 | 0.9840420938241821 | 0.9810971230344524 | 2319 | none |
| board_grouped | v6_1_frequency_fusion | balanced_logistic_regression | C03 | 0.9346365302382407 | 0.8146648925093422 | 0.7873302428070197 | 1637 | none |
| board_grouped | v6_1_frequency_fusion | balanced_logistic_regression | W01 | 0.9702356345597354 | 0.9201867559499586 | 0.919169738690014 | 2419 | none |
| board_grouped | v6_1_frequency_fusion | balanced_logistic_regression | W02 | 0.9628691983122363 | 0.9138690955727793 | 0.8900770283032087 | 2370 | none |
| board_grouped | v6_1_frequency_fusion | balanced_logistic_regression | W03 | 0.9645293315143247 | 0.9110374779136301 | 0.8958672798145411 | 1466 | none |
| board_grouped | v6_1_frequency_fusion | random_forest | C01 | 0.9846322722283205 | 0.9080542046572718 | 0.9090878778976108 | 2733 | none |
| board_grouped | v6_1_frequency_fusion | random_forest | C02 | 0.9879258300991807 | 0.9653428505381413 | 0.9777045443430762 | 2319 | none |
| board_grouped | v6_1_frequency_fusion | random_forest | C03 | 0.9682345754428833 | 0.900629486343772 | 0.9408217608513694 | 1637 | none |
| board_grouped | v6_1_frequency_fusion | random_forest | W01 | 0.9661016949152542 | 0.8987327224644857 | 0.9250634586495424 | 2419 | none |
| board_grouped | v6_1_frequency_fusion | random_forest | W02 | 0.9628691983122363 | 0.8754359561266867 | 0.8953282533997233 | 2370 | none |
| board_grouped | v6_1_frequency_fusion | random_forest | W03 | 0.970668485675307 | 0.9025658862177861 | 0.9194024559832183 | 1466 | none |
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
| board_grouped | random_resnet18_embedding_only | balanced_logistic_regression | C01 | 0.8309549945115258 | 0.41031267725013476 | 0.41340233538526217 | 2733 | none |
| board_grouped | random_resnet18_embedding_only | balanced_logistic_regression | C02 | 0.7723156532988357 | 0.32229511031950386 | 0.3252259294417491 | 2319 | none |
| board_grouped | random_resnet18_embedding_only | balanced_logistic_regression | C03 | 0.6273671350030544 | 0.3698473672902475 | 0.3104071202182591 | 1637 | none |
| board_grouped | random_resnet18_embedding_only | balanced_logistic_regression | W01 | 0.7044233154195949 | 0.3515491033724049 | 0.2807026019369698 | 2419 | none |
| board_grouped | random_resnet18_embedding_only | balanced_logistic_regression | W02 | 0.8177215189873418 | 0.41719353476932647 | 0.3676620218746874 | 2370 | none |
| board_grouped | random_resnet18_embedding_only | balanced_logistic_regression | W03 | 0.31036834924965895 | 0.3765006695304768 | 0.23281724837195306 | 1466 | none |
| board_grouped | random_resnet18_embedding_only | random_forest | C01 | 0.8466886205634834 | 0.2164705882352941 | 0.21361024635636036 | 2733 | none |
| board_grouped | random_resnet18_embedding_only | random_forest | C02 | 0.7326433807675722 | 0.2400249406889096 | 0.21857391799177858 | 2319 | none |
| board_grouped | random_resnet18_embedding_only | random_forest | C03 | 0.7562614538790471 | 0.2 | 0.17224347826086955 | 1637 | none |
| board_grouped | random_resnet18_embedding_only | random_forest | W01 | 0.7581645307978504 | 0.2 | 0.17257115972712303 | 2419 | none |
| board_grouped | random_resnet18_embedding_only | random_forest | W02 | 0.7902953586497891 | 0.20270270270270271 | 0.1818669314796425 | 2370 | none |
| board_grouped | random_resnet18_embedding_only | random_forest | W03 | 0.747612551159618 | 0.2 | 0.17111631537861044 | 1466 | none |
| board_grouped | random_resnet18_frequency_fusion | balanced_logistic_regression | C01 | 0.9473106476399561 | 0.74077087044512 | 0.7237956566119431 | 2733 | none |
| board_grouped | random_resnet18_frequency_fusion | balanced_logistic_regression | C02 | 0.8503665373005606 | 0.5035112874113705 | 0.49290989865516704 | 2319 | none |
| board_grouped | random_resnet18_frequency_fusion | balanced_logistic_regression | C03 | 0.8423946243127672 | 0.578626463782245 | 0.563767960380463 | 1637 | none |
| board_grouped | random_resnet18_frequency_fusion | balanced_logistic_regression | W01 | 0.8350558081852005 | 0.4926232387538816 | 0.4563741984535475 | 2419 | none |
| board_grouped | random_resnet18_frequency_fusion | balanced_logistic_regression | W02 | 0.8915611814345992 | 0.6480248367595083 | 0.5815010634819866 | 2370 | none |
| board_grouped | random_resnet18_frequency_fusion | balanced_logistic_regression | W03 | 0.616643929058663 | 0.46262175739546424 | 0.3767301843357092 | 1466 | none |
| board_grouped | random_resnet18_frequency_fusion | random_forest | C01 | 0.9096231247713136 | 0.46070223438212493 | 0.4952118526828757 | 2733 | none |
| board_grouped | random_resnet18_frequency_fusion | random_forest | C02 | 0.8076757222940922 | 0.35196014895342204 | 0.3699578891214911 | 2319 | none |
| board_grouped | random_resnet18_frequency_fusion | random_forest | C03 | 0.7849725106902871 | 0.2872235872235872 | 0.3010789538129289 | 1637 | none |
| board_grouped | random_resnet18_frequency_fusion | random_forest | W01 | 0.7767672591980157 | 0.27773122784402193 | 0.28803830446415996 | 2419 | none |
| board_grouped | random_resnet18_frequency_fusion | random_forest | W02 | 0.7949367088607595 | 0.23056685003587657 | 0.23159684595392274 | 2370 | none |
| board_grouped | random_resnet18_frequency_fusion | random_forest | W03 | 0.7639836289222374 | 0.25094850948509484 | 0.25812822058868756 | 1466 | none |
| board_grouped | imagenet_resnet18_embedding_only | balanced_logistic_regression | C01 | 0.5675082327113062 | 0.3389351802859208 | 0.28115926372148936 | 2733 | none |
| board_grouped | imagenet_resnet18_embedding_only | balanced_logistic_regression | C02 | 0.6218197498921949 | 0.43640996430212536 | 0.3735450796268472 | 2319 | none |
| board_grouped | imagenet_resnet18_embedding_only | balanced_logistic_regression | C03 | 0.7556505803298718 | 0.43458856995436734 | 0.4047304373966888 | 1637 | none |
| board_grouped | imagenet_resnet18_embedding_only | balanced_logistic_regression | W01 | 0.7560975609756098 | 0.5105998883458446 | 0.46568154148210644 | 2419 | none |
| board_grouped | imagenet_resnet18_embedding_only | balanced_logistic_regression | W02 | 0.7248945147679325 | 0.3370193948158502 | 0.3441807846118231 | 2370 | none |
| board_grouped | imagenet_resnet18_embedding_only | balanced_logistic_regression | W03 | 0.6875852660300137 | 0.44171862427359854 | 0.4069820388689756 | 1466 | none |
| board_grouped | imagenet_resnet18_embedding_only | random_forest | C01 | 0.8551042810098792 | 0.24951208390332877 | 0.26376762330077846 | 2733 | none |
| board_grouped | imagenet_resnet18_embedding_only | random_forest | C02 | 0.7645536869340233 | 0.35983855371032164 | 0.2866306326990717 | 2319 | none |
| board_grouped | imagenet_resnet18_embedding_only | random_forest | C03 | 0.7953573610262675 | 0.3189047881778092 | 0.32796964413906265 | 1637 | none |
| board_grouped | imagenet_resnet18_embedding_only | random_forest | W01 | 0.7796610169491526 | 0.2951659684176603 | 0.3147229292545769 | 2419 | none |
| board_grouped | imagenet_resnet18_embedding_only | random_forest | W02 | 0.789873417721519 | 0.2 | 0.17652050919377651 | 2370 | none |
| board_grouped | imagenet_resnet18_embedding_only | random_forest | W03 | 0.7646657571623465 | 0.26249587944431363 | 0.27547370679633276 | 1466 | none |
| board_grouped | imagenet_resnet18_frequency_fusion | balanced_logistic_regression | C01 | 0.7731430662275888 | 0.4718931055056359 | 0.42713127201725376 | 2733 | none |
| board_grouped | imagenet_resnet18_frequency_fusion | balanced_logistic_regression | C02 | 0.8094006037084951 | 0.6298574865096549 | 0.5870908760877936 | 2319 | none |
| board_grouped | imagenet_resnet18_frequency_fusion | balanced_logistic_regression | C03 | 0.8442272449602932 | 0.5552852446505805 | 0.5471536103925297 | 1637 | none |
| board_grouped | imagenet_resnet18_frequency_fusion | balanced_logistic_regression | W01 | 0.8272013228606863 | 0.6129065807257001 | 0.5438420996558987 | 2419 | none |
| board_grouped | imagenet_resnet18_frequency_fusion | balanced_logistic_regression | W02 | 0.8396624472573839 | 0.47407741562707095 | 0.4904699296337025 | 2370 | none |
| board_grouped | imagenet_resnet18_frequency_fusion | balanced_logistic_regression | W03 | 0.8274215552523875 | 0.5482070182810375 | 0.5545989597859483 | 1466 | none |
| board_grouped | imagenet_resnet18_frequency_fusion | random_forest | C01 | 0.9037687522868643 | 0.43440036479708166 | 0.46116453185287315 | 2733 | none |
| board_grouped | imagenet_resnet18_frequency_fusion | random_forest | C02 | 0.8257869771453212 | 0.4130597014925373 | 0.41117653443540714 | 2319 | none |
| board_grouped | imagenet_resnet18_frequency_fusion | random_forest | C03 | 0.8197923029932804 | 0.39174447174447175 | 0.3854961930465163 | 1637 | none |
| board_grouped | imagenet_resnet18_frequency_fusion | random_forest | W01 | 0.8230673832162051 | 0.4811311633902674 | 0.504229037087876 | 2419 | none |
| board_grouped | imagenet_resnet18_frequency_fusion | random_forest | W02 | 0.8042194092827004 | 0.2797656063142789 | 0.3076376619622567 | 2370 | none |
| board_grouped | imagenet_resnet18_frequency_fusion | random_forest | W03 | 0.8342428376534788 | 0.4789964157706093 | 0.49945816998551207 | 1466 | none |

## Material LOMO-Style Fold Scores

| protocol | condition | method | held_out_group | accuracy | macro_recall | macro_f1 | n_test | stress_test_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| material_grouped | frequency_only | balanced_logistic_regression | carbon | 0.987143070713111 | 0.9418918918918919 | 0.9441804939708274 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | frequency_only | balanced_logistic_regression | wood | 0.9792166266986411 | 0.937799043062201 | 0.9375317788915524 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | frequency_only | random_forest | carbon | 0.987143070713111 | 0.9418918918918919 | 0.9441804939708274 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | frequency_only | random_forest | wood | 0.9792166266986411 | 0.937799043062201 | 0.9375317788915524 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | v6_1_embedding_only | balanced_logistic_regression | carbon | 0.8771116758857826 | 0.7219237381143053 | 0.6526256877229313 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | v6_1_embedding_only | balanced_logistic_regression | wood | 0.8298960831334932 | 0.8475125378293352 | 0.7171229419158214 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | v6_1_embedding_only | random_forest | carbon | 0.9606817162505606 | 0.8560546040065209 | 0.8951722630265924 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | v6_1_embedding_only | random_forest | wood | 0.9440447641886491 | 0.8386712940934405 | 0.8601791767154214 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | v6_1_frequency_fusion | balanced_logistic_regression | carbon | 0.9219614292121393 | 0.7582534153881946 | 0.715106821884889 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | v6_1_frequency_fusion | balanced_logistic_regression | wood | 0.9501199040767386 | 0.9010941012708317 | 0.8791524162299804 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | v6_1_frequency_fusion | random_forest | carbon | 0.9680071759605322 | 0.8762312538346517 | 0.9141929236835038 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | v6_1_frequency_fusion | random_forest | wood | 0.9568345323741008 | 0.8707057511182086 | 0.8897739710508523 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | v6_2_a_embedding_only | balanced_logistic_regression | carbon | 0.9693526685603229 | 0.9304735644669779 | 0.932156897615069 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | v6_2_a_embedding_only | balanced_logistic_regression | wood | 0.9443645083932853 | 0.9283298973657704 | 0.8899964004860413 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | v6_2_a_embedding_only | random_forest | carbon | 0.9753326356705038 | 0.9172216823288618 | 0.9433040380458856 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | v6_2_a_embedding_only | random_forest | wood | 0.9705835331734612 | 0.9265896541546661 | 0.9377956484963115 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | v6_2_a_frequency_fusion | balanced_logistic_regression | carbon | 0.9790701151143669 | 0.9406055022388872 | 0.9492816145049012 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | v6_2_a_frequency_fusion | balanced_logistic_regression | wood | 0.9702637889688249 | 0.9502589096916652 | 0.9312595914006636 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | v6_2_a_frequency_fusion | random_forest | carbon | 0.97652862909254 | 0.9187266272100745 | 0.9454285273902064 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | v6_2_a_frequency_fusion | random_forest | wood | 0.9750599520383693 | 0.9296869696509112 | 0.9445749221125169 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | random_resnet18_embedding_only | balanced_logistic_regression | carbon | 0.7409179249514127 | 0.2335588276922355 | 0.23668563042104723 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | random_resnet18_embedding_only | balanced_logistic_regression | wood | 0.6258992805755396 | 0.3311481230589539 | 0.2721526656403265 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | random_resnet18_embedding_only | random_forest | carbon | 0.7871131708775602 | 0.2086505912513632 | 0.19384668988676484 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | random_resnet18_embedding_only | random_forest | wood | 0.7673860911270983 | 0.2005893377759267 | 0.17513498818618473 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | random_resnet18_frequency_fusion | balanced_logistic_regression | carbon | 0.815667513828674 | 0.3761890863187667 | 0.3733124628200365 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | random_resnet18_frequency_fusion | balanced_logistic_regression | wood | 0.7544364508393285 | 0.46696379521457965 | 0.393672922480544 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | random_resnet18_frequency_fusion | random_forest | carbon | 0.8300194348931081 | 0.33157819660843113 | 0.36518248757996397 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | random_resnet18_frequency_fusion | random_forest | wood | 0.7769784172661871 | 0.23933419521654814 | 0.24218456819867132 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | imagenet_resnet18_embedding_only | balanced_logistic_regression | carbon | 0.3263567050381223 | 0.2621257798676594 | 0.16826248628727253 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | imagenet_resnet18_embedding_only | balanced_logistic_regression | wood | 0.7283772981614708 | 0.3231782767726136 | 0.2920469122780007 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | imagenet_resnet18_embedding_only | random_forest | carbon | 0.8044550754970847 | 0.2835717381096023 | 0.2884121103566269 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | imagenet_resnet18_embedding_only | random_forest | wood | 0.769144684252598 | 0.2060520310360566 | 0.18575991849811518 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | imagenet_resnet18_frequency_fusion | balanced_logistic_regression | carbon | 0.498280759455823 | 0.33717015435226616 | 0.24956617329793582 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | imagenet_resnet18_frequency_fusion | balanced_logistic_regression | wood | 0.75171862509992 | 0.3998857519205258 | 0.3310444132160278 | 6255 | descriptive stress-test only; two material groups available |
| material_grouped | imagenet_resnet18_frequency_fusion | random_forest | carbon | 0.869487217820302 | 0.4739419329518487 | 0.4894820273726624 | 6689 | descriptive stress-test only; two material groups available |
| material_grouped | imagenet_resnet18_frequency_fusion | random_forest | wood | 0.7851318944844125 | 0.2740917389319945 | 0.2967954746728057 | 6255 | descriptive stress-test only; two material groups available |

## Existing Encoder-Only Reference Metrics

| encoder | stratified_knn_macro_f1 | board_lobo_macro_f1 | material_lomo_macro_f1 |
| --- | --- | --- | --- |
| Random ResNet-18 | 0.8439863653774754 | 0.319913304491168 | 0.2543697950088857 |
| ImageNet ResNet-18 | 0.8715130509203878 | 0.3930272194094586 | 0.229527070800846 |
| v6.1 | 0.9506911900286517 | 0.8709697195504678 | 0.8632932432336362 |
| v6.2-A | 0.927755752028969 | 0.9356408210770706 | 0.9348717330088938 |

## v6.2-A Information-Budget Interpretation

- `stratified_train_val`: frequency-only 98.67%, v6.2-A embedding-only 93.48%, frequency + v6.2-A 96.75%; fusion minus frequency -1.92 pp, fusion minus embedding +3.27 pp.
- `board_grouped`: frequency-only 97.49%, v6.2-A embedding-only 94.28%, frequency + v6.2-A 94.79%; fusion minus frequency -2.70 pp, fusion minus embedding +0.50 pp.
- `material_grouped`: frequency-only 94.09%, v6.2-A embedding-only 94.05%, frequency + v6.2-A 94.50%; fusion minus frequency +0.41 pp, fusion minus embedding +0.45 pp.
- v6.1 benefits more from frequency fusion than v6.2-A on average across the evaluated protocols.
- Frequency + v6.2-A improves over frequency-only in at least one protocol, indicating that the frozen ESPI embedding can add complementary image-derived information under that setting.
- Material LOMO remains a descriptive stress test because only two material groups are available.
- The result refines, rather than expands, the manuscript claim: the main claim is image-derived frozen encoder evidence, not acoustic-response prediction.

## Manuscript Patch Recommendation

- Abstract: state that frequency metadata is a strong metadata-only predictor and that v6.2-A is the strongest image-derived frozen ESPI encoder baseline, not the strongest predictor overall.
- Results: add a frequency-only and frequency-plus-embedding information-budget paragraph before the final encoder decision paragraph.
- Discussion: explain that the five-class modal protocol is strongly frequency-structured; frozen ESPI embeddings should be interpreted as image-derived representation evidence rather than as proof that frequency metadata is insufficient.