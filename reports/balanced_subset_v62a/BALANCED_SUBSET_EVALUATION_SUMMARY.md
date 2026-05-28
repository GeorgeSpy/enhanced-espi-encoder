# Balanced Subset Evaluation

This evaluation tests whether the v6.2-A grouped advantage persists under class/material/frequency balancing.

- v6.1 features: `outputs\encoder_features_normalized_v001\features_v61.normalized.npz`
- v6.2-A features: `outputs\encoder_features_normalized_v001\features_v62a_epoch25.normalized.npz`
- kNN: fixed k=10
- Frequency-bin width for matched subset: `25.0` Hz
- Min cell for frequency-matched subset: `3`

## Subset Composition

| subset                     | label | n   | freq_min | freq_max  | freq_mean | wood_n | carbon_n |
| -------------------------- | ----- | --- | -------- | --------- | --------- | ------ | -------- |
| class_balanced             | 0     | 653 | 150.0000 | 190.0000  | 171.1868  | 280    | 373      |
| class_balanced             | 1     | 653 | 320.0000 | 360.0000  | 340.2220  | 302    | 351      |
| class_balanced             | 2     | 653 | 500.0000 | 556.0000  | 528.4717  | 389    | 264      |
| class_balanced             | 3     | 653 | 550.0000 | 605.0000  | 579.2665  | 352    | 301      |
| class_balanced             | 4     | 653 | 700.0000 | 1550.0000 | 1026.4380 | 298    | 355      |
| class_material_balanced    | 0     | 560 | 150.0000 | 190.0000  | 169.8214  | 280    | 280      |
| class_material_balanced    | 1     | 560 | 320.0000 | 360.0000  | 339.5179  | 280    | 280      |
| class_material_balanced    | 2     | 560 | 500.0000 | 556.0000  | 530.4803  | 280    | 280      |
| class_material_balanced    | 3     | 560 | 550.0000 | 605.0000  | 580.1304  | 280    | 280      |
| class_material_balanced    | 4     | 560 | 700.0000 | 1550.0000 | 1010.9714 | 280    | 280      |
| class_material_freqmatched | 0     | 184 | 150.0000 | 170.0000  | 165.0000  | 92     | 92       |
| class_material_freqmatched | 1     | 184 | 325.0000 | 345.0000  | 337.5000  | 89     | 95       |
| class_material_freqmatched | 2     | 184 | 525.0000 | 549.0000  | 535.0217  | 90     | 94       |
| class_material_freqmatched | 3     | 184 | 550.0000 | 595.0000  | 580.0109  | 98     | 86       |
| class_material_freqmatched | 4     | 184 | 700.0000 | 1530.0000 | 981.8967  | 86     | 98       |

## Summary Metrics

| subset                     | protocol      | method                 | n_folds | macro_f1_mean | macro_f1_std | balanced_accuracy_mean | accuracy_mean | weighted_f1_mean | worst_group | worst_group_macro_f1 | best_group | best_group_macro_f1 |
| -------------------------- | ------------- | ---------------------- | ------- | ------------- | ------------ | ---------------------- | ------------- | ---------------- | ----------- | -------------------- | ---------- | ------------------- |
| class_balanced             | board_lobo    | frequency_only_lr      | 6       | 0.9798        | 0.0102       | 0.9783                 | 0.9794        | 0.9792           | W03         | 0.9673               | C01        | 1.0000              |
| class_balanced             | board_lobo    | frequency_plus_v62a_lr | 6       | 0.9518        | 0.0082       | 0.9496                 | 0.9540        | 0.9542           | W01         | 0.9380               | W02        | 0.9633              |
| class_balanced             | board_lobo    | v61_knn                | 6       | 0.9023        | 0.0286       | 0.9037                 | 0.9087        | 0.9088           | W02         | 0.8634               | C03        | 0.9496              |
| class_balanced             | board_lobo    | v62a_knn               | 6       | 0.9442        | 0.0103       | 0.9412                 | 0.9470        | 0.9470           | W01         | 0.9256               | C03        | 0.9548              |
| class_balanced             | material_lomo | frequency_only_lr      | 2       | 0.9405        | 0.0035       | 0.9393                 | 0.9423        | 0.9407           | wood        | 0.9370               | carbon     | 0.9440              |
| class_balanced             | material_lomo | frequency_plus_v62a_lr | 2       | 0.9494        | 0.0038       | 0.9467                 | 0.9479        | 0.9482           | carbon      | 0.9456               | wood       | 0.9531              |
| class_balanced             | material_lomo | v61_knn                | 2       | 0.8864        | 0.0144       | 0.8880                 | 0.8853        | 0.8847           | wood        | 0.8720               | carbon     | 0.9007              |
| class_balanced             | material_lomo | v62a_knn               | 2       | 0.9364        | 0.0057       | 0.9334                 | 0.9356        | 0.9359           | wood        | 0.9308               | carbon     | 0.9421              |
| class_balanced             | stratified    | frequency_only_lr      | 1       | 0.9712        | 0.0000       | 0.9712                 | 0.9707        | 0.9707           | validation  | 0.9712               | validation | 0.9712              |
| class_balanced             | stratified    | frequency_plus_v62a_lr | 1       | 0.9557        | 0.0000       | 0.9558                 | 0.9553        | 0.9553           | validation  | 0.9557               | validation | 0.9557              |
| class_balanced             | stratified    | v61_knn                | 1       | 0.9508        | 0.0000       | 0.9509                 | 0.9507        | 0.9506           | validation  | 0.9508               | validation | 0.9508              |
| class_balanced             | stratified    | v62a_knn               | 1       | 0.9416        | 0.0000       | 0.9418                 | 0.9414        | 0.9415           | validation  | 0.9416               | validation | 0.9416              |
| class_material_balanced    | board_lobo    | frequency_only_lr      | 6       | 0.9741        | 0.0133       | 0.9738                 | 0.9736        | 0.9735           | W02         | 0.9600               | C01        | 1.0000              |
| class_material_balanced    | board_lobo    | frequency_plus_v62a_lr | 6       | 0.9513        | 0.0047       | 0.9487                 | 0.9539        | 0.9540           | C03         | 0.9457               | W02        | 0.9581              |
| class_material_balanced    | board_lobo    | v61_knn                | 6       | 0.8989        | 0.0298       | 0.8985                 | 0.9095        | 0.9096           | C01         | 0.8520               | C03        | 0.9435              |
| class_material_balanced    | board_lobo    | v62a_knn               | 6       | 0.9386        | 0.0095       | 0.9366                 | 0.9424        | 0.9423           | W01         | 0.9287               | W02        | 0.9552              |
| class_material_balanced    | material_lomo | frequency_only_lr      | 2       | 0.9424        | 0.0015       | 0.9436                 | 0.9436        | 0.9424           | carbon      | 0.9409               | wood       | 0.9439              |
| class_material_balanced    | material_lomo | frequency_plus_v62a_lr | 2       | 0.9479        | 0.0034       | 0.9475                 | 0.9475        | 0.9479           | carbon      | 0.9445               | wood       | 0.9513              |
| class_material_balanced    | material_lomo | v61_knn                | 2       | 0.8824        | 0.0132       | 0.8836                 | 0.8836        | 0.8824           | wood        | 0.8692               | carbon     | 0.8956              |
| class_material_balanced    | material_lomo | v62a_knn               | 2       | 0.9373        | 0.0016       | 0.9368                 | 0.9368        | 0.9373           | wood        | 0.9357               | carbon     | 0.9389              |
| class_material_balanced    | stratified    | frequency_only_lr      | 1       | 0.9760        | 0.0000       | 0.9760                 | 0.9768        | 0.9768           | validation  | 0.9760               | validation | 0.9760              |
| class_material_balanced    | stratified    | frequency_plus_v62a_lr | 1       | 0.9547        | 0.0000       | 0.9547                 | 0.9554        | 0.9555           | validation  | 0.9547               | validation | 0.9547              |
| class_material_balanced    | stratified    | v61_knn                | 1       | 0.9291        | 0.0000       | 0.9291                 | 0.9305        | 0.9304           | validation  | 0.9291               | validation | 0.9291              |
| class_material_balanced    | stratified    | v62a_knn               | 1       | 0.9319        | 0.0000       | 0.9320                 | 0.9323        | 0.9324           | validation  | 0.9319               | validation | 0.9319              |
| class_material_freqmatched | board_lobo    | frequency_only_lr      | 6       | 0.9894        | 0.0132       | 0.9893                 | 0.9907        | 0.9906           | W03         | 0.9638               | C01        | 1.0000              |
| class_material_freqmatched | board_lobo    | frequency_plus_v62a_lr | 6       | 0.9466        | 0.0137       | 0.9451                 | 0.9483        | 0.9489           | W03         | 0.9228               | W02        | 0.9621              |
| class_material_freqmatched | board_lobo    | v61_knn                | 6       | 0.8643        | 0.0531       | 0.8699                 | 0.8807        | 0.8788           | C01         | 0.7912               | C03        | 0.9392              |
| class_material_freqmatched | board_lobo    | v62a_knn               | 6       | 0.9483        | 0.0126       | 0.9478                 | 0.9505        | 0.9508           | W03         | 0.9310               | C02        | 0.9709              |
| class_material_freqmatched | material_lomo | frequency_only_lr      | 2       | 0.9915        | 0.0085       | 0.9918                 | 0.9912        | 0.9912           | wood        | 0.9830               | carbon     | 1.0000              |
| class_material_freqmatched | material_lomo | frequency_plus_v62a_lr | 2       | 0.9339        | 0.0150       | 0.9334                 | 0.9335        | 0.9333           | wood        | 0.9190               | carbon     | 0.9489              |
| class_material_freqmatched | material_lomo | v61_knn                | 2       | 0.8181        | 0.0414       | 0.8221                 | 0.8256        | 0.8186           | wood        | 0.7767               | carbon     | 0.8595              |
| class_material_freqmatched | material_lomo | v62a_knn               | 2       | 0.9473        | 0.0123       | 0.9465                 | 0.9466        | 0.9470           | wood        | 0.9350               | carbon     | 0.9596              |
| class_material_freqmatched | stratified    | frequency_only_lr      | 1       | 0.9945        | 0.0000       | 0.9952                 | 0.9949        | 0.9950           | validation  | 0.9945               | validation | 0.9945              |
| class_material_freqmatched | stratified    | frequency_plus_v62a_lr | 1       | 0.9572        | 0.0000       | 0.9534                 | 0.9596        | 0.9593           | validation  | 0.9572               | validation | 0.9572              |
| class_material_freqmatched | stratified    | v61_knn                | 1       | 0.9309        | 0.0000       | 0.9316                 | 0.9343        | 0.9338           | validation  | 0.9309               | validation | 0.9309              |
| class_material_freqmatched | stratified    | v62a_knn               | 1       | 0.9420        | 0.0000       | 0.9413                 | 0.9495        | 0.9483           | validation  | 0.9420               | validation | 0.9420              |

## Fold Metrics

| accuracy | balanced_accuracy | macro_f1 | weighted_f1 | subset                     | protocol      | method                 | heldout_group | n_train | n_test |
| -------- | ----------------- | -------- | ----------- | -------------------------- | ------------- | ---------------------- | ------------- | ------- | ------ |
| 1.0000   | 1.0000            | 1.0000   | 1.0000      | class_balanced             | board_lobo    | frequency_only_lr      | C01           | 2708    | 557    |
| 0.9725   | 0.9721            | 0.9729   | 0.9723      | class_balanced             | board_lobo    | frequency_only_lr      | C02           | 2647    | 618    |
| 0.9787   | 0.9783            | 0.9792   | 0.9786      | class_balanced             | board_lobo    | frequency_only_lr      | C03           | 2796    | 469    |
| 0.9781   | 0.9817            | 0.9818   | 0.9781      | class_balanced             | board_lobo    | frequency_only_lr      | W01           | 2625    | 640    |
| 0.9735   | 0.9795            | 0.9777   | 0.9735      | class_balanced             | board_lobo    | frequency_only_lr      | W02           | 2699    | 566    |
| 0.9735   | 0.9585            | 0.9673   | 0.9726      | class_balanced             | board_lobo    | frequency_only_lr      | W03           | 2850    | 415    |
| 0.9749   | 0.9569            | 0.9576   | 0.9748      | class_balanced             | board_lobo    | frequency_plus_v62a_lr | C01           | 2708    | 557    |
| 0.9450   | 0.9425            | 0.9457   | 0.9458      | class_balanced             | board_lobo    | frequency_plus_v62a_lr | C02           | 2647    | 618    |
| 0.9510   | 0.9527            | 0.9514   | 0.9517      | class_balanced             | board_lobo    | frequency_plus_v62a_lr | C03           | 2796    | 469    |
| 0.9313   | 0.9358            | 0.9380   | 0.9316      | class_balanced             | board_lobo    | frequency_plus_v62a_lr | W01           | 2625    | 640    |
| 0.9629   | 0.9605            | 0.9633   | 0.9629      | class_balanced             | board_lobo    | frequency_plus_v62a_lr | W02           | 2699    | 566    |
| 0.9590   | 0.9493            | 0.9548   | 0.9587      | class_balanced             | board_lobo    | frequency_plus_v62a_lr | W03           | 2850    | 415    |
| 0.9138   | 0.8901            | 0.8751   | 0.9171      | class_balanced             | board_lobo    | v61_knn                | C01           | 2708    | 557    |
| 0.9239   | 0.9224            | 0.9219   | 0.9228      | class_balanced             | board_lobo    | v61_knn                | C02           | 2647    | 618    |
| 0.9510   | 0.9486            | 0.9496   | 0.9507      | class_balanced             | board_lobo    | v61_knn                | C03           | 2796    | 469    |
| 0.8969   | 0.8995            | 0.9047   | 0.8980      | class_balanced             | board_lobo    | v61_knn                | W01           | 2625    | 640    |
| 0.8534   | 0.8683            | 0.8634   | 0.8524      | class_balanced             | board_lobo    | v61_knn                | W02           | 2699    | 566    |
| 0.9133   | 0.8930            | 0.8989   | 0.9115      | class_balanced             | board_lobo    | v61_knn                | W03           | 2850    | 415    |
| 0.9587   | 0.9359            | 0.9392   | 0.9585      | class_balanced             | board_lobo    | v62a_knn               | C01           | 2708    | 557    |
| 0.9515   | 0.9514            | 0.9523   | 0.9516      | class_balanced             | board_lobo    | v62a_knn               | C02           | 2647    | 618    |
| 0.9531   | 0.9544            | 0.9548   | 0.9536      | class_balanced             | board_lobo    | v62a_knn               | C03           | 2796    | 469    |
| 0.9219   | 0.9243            | 0.9256   | 0.9221      | class_balanced             | board_lobo    | v62a_knn               | W01           | 2625    | 640    |
| 0.9523   | 0.9482            | 0.9526   | 0.9522      | class_balanced             | board_lobo    | v62a_knn               | W02           | 2699    | 566    |
| 0.9446   | 0.9331            | 0.9406   | 0.9444      | class_balanced             | board_lobo    | v62a_knn               | W03           | 2850    | 415    |
| 0.9538   | 0.9424            | 0.9440   | 0.9525      | class_balanced             | material_lomo | frequency_only_lr      | carbon        | 1621    | 1644   |
| 0.9309   | 0.9362            | 0.9370   | 0.9288      | class_balanced             | material_lomo | frequency_only_lr      | wood          | 1644    | 1621   |
| 0.9465   | 0.9425            | 0.9456   | 0.9465      | class_balanced             | material_lomo | frequency_plus_v62a_lr | carbon        | 1621    | 1644   |
| 0.9494   | 0.9509            | 0.9531   | 0.9498      | class_balanced             | material_lomo | frequency_plus_v62a_lr | wood          | 1644    | 1621   |
| 0.9063   | 0.9022            | 0.9007   | 0.9053      | class_balanced             | material_lomo | v61_knn                | carbon        | 1621    | 1644   |
| 0.8643   | 0.8737            | 0.8720   | 0.8642      | class_balanced             | material_lomo | v61_knn                | wood          | 1644    | 1621   |
| 0.9428   | 0.9380            | 0.9421   | 0.9429      | class_balanced             | material_lomo | v62a_knn               | carbon        | 1621    | 1644   |
| 0.9284   | 0.9287            | 0.9308   | 0.9288      | class_balanced             | material_lomo | v62a_knn               | wood          | 1644    | 1621   |
| 0.9707   | 0.9712            | 0.9712   | 0.9707      | class_balanced             | stratified    | frequency_only_lr      | validation    | 2616    | 649    |
| 0.9553   | 0.9558            | 0.9557   | 0.9553      | class_balanced             | stratified    | frequency_plus_v62a_lr | validation    | 2616    | 649    |
| 0.9507   | 0.9509            | 0.9508   | 0.9506      | class_balanced             | stratified    | v61_knn                | validation    | 2616    | 649    |
| 0.9414   | 0.9418            | 0.9416   | 0.9415      | class_balanced             | stratified    | v62a_knn               | validation    | 2616    | 649    |
| 1.0000   | 1.0000            | 1.0000   | 1.0000      | class_material_balanced    | board_lobo    | frequency_only_lr      | C01           | 2327    | 473    |
| 0.9683   | 0.9744            | 0.9734   | 0.9682      | class_material_balanced    | board_lobo    | frequency_only_lr      | C02           | 2264    | 536    |
| 0.9770   | 0.9802            | 0.9804   | 0.9769      | class_material_balanced    | board_lobo    | frequency_only_lr      | C03           | 2409    | 391    |
| 0.9620   | 0.9672            | 0.9643   | 0.9620      | class_material_balanced    | board_lobo    | frequency_only_lr      | W01           | 2248    | 552    |
| 0.9565   | 0.9632            | 0.9600   | 0.9565      | class_material_balanced    | board_lobo    | frequency_only_lr      | W02           | 2317    | 483    |
| 0.9781   | 0.9579            | 0.9666   | 0.9773      | class_material_balanced    | board_lobo    | frequency_only_lr      | W03           | 2435    | 365    |
| 0.9662   | 0.9525            | 0.9564   | 0.9656      | class_material_balanced    | board_lobo    | frequency_plus_v62a_lr | C01           | 2327    | 473    |
| 0.9440   | 0.9400            | 0.9462   | 0.9444      | class_material_balanced    | board_lobo    | frequency_plus_v62a_lr | C02           | 2264    | 536    |
| 0.9437   | 0.9512            | 0.9457   | 0.9449      | class_material_balanced    | board_lobo    | frequency_plus_v62a_lr | C03           | 2409    | 391    |
| 0.9493   | 0.9491            | 0.9506   | 0.9495      | class_material_balanced    | board_lobo    | frequency_plus_v62a_lr | W01           | 2248    | 552    |
| 0.9586   | 0.9560            | 0.9581   | 0.9586      | class_material_balanced    | board_lobo    | frequency_plus_v62a_lr | W02           | 2317    | 483    |
| 0.9616   | 0.9432            | 0.9508   | 0.9609      | class_material_balanced    | board_lobo    | frequency_plus_v62a_lr | W03           | 2435    | 365    |
| 0.8943   | 0.8630            | 0.8520   | 0.8955      | class_material_balanced    | board_lobo    | v61_knn                | C01           | 2327    | 473    |
| 0.9328   | 0.9283            | 0.9272   | 0.9326      | class_material_balanced    | board_lobo    | v61_knn                | C02           | 2264    | 536    |
| 0.9437   | 0.9436            | 0.9435   | 0.9433      | class_material_balanced    | board_lobo    | v61_knn                | C03           | 2409    | 391    |
| 0.8913   | 0.8912            | 0.8950   | 0.8930      | class_material_balanced    | board_lobo    | v61_knn                | W01           | 2248    | 552    |
| 0.8882   | 0.8956            | 0.8938   | 0.8880      | class_material_balanced    | board_lobo    | v61_knn                | W02           | 2317    | 483    |
| 0.9068   | 0.8692            | 0.8820   | 0.9050      | class_material_balanced    | board_lobo    | v61_knn                | W03           | 2435    | 365    |
| 0.9408   | 0.9257            | 0.9310   | 0.9400      | class_material_balanced    | board_lobo    | v62a_knn               | C01           | 2327    | 473    |
| 0.9459   | 0.9431            | 0.9456   | 0.9460      | class_material_balanced    | board_lobo    | v62a_knn               | C02           | 2264    | 536    |
| 0.9412   | 0.9441            | 0.9406   | 0.9415      | class_material_balanced    | board_lobo    | v62a_knn               | C03           | 2409    | 391    |
| 0.9275   | 0.9293            | 0.9287   | 0.9277      | class_material_balanced    | board_lobo    | v62a_knn               | W01           | 2248    | 552    |
| 0.9565   | 0.9524            | 0.9552   | 0.9563      | class_material_balanced    | board_lobo    | v62a_knn               | W02           | 2317    | 483    |
| 0.9425   | 0.9253            | 0.9308   | 0.9421      | class_material_balanced    | board_lobo    | v62a_knn               | W03           | 2435    | 365    |
| 0.9421   | 0.9421            | 0.9409   | 0.9409      | class_material_balanced    | material_lomo | frequency_only_lr      | carbon        | 1400    | 1400   |
| 0.9450   | 0.9450            | 0.9439   | 0.9439      | class_material_balanced    | material_lomo | frequency_only_lr      | wood          | 1400    | 1400   |
| 0.9443   | 0.9443            | 0.9445   | 0.9445      | class_material_balanced    | material_lomo | frequency_plus_v62a_lr | carbon        | 1400    | 1400   |
| 0.9507   | 0.9507            | 0.9513   | 0.9513      | class_material_balanced    | material_lomo | frequency_plus_v62a_lr | wood          | 1400    | 1400   |
| 0.8971   | 0.8971            | 0.8956   | 0.8956      | class_material_balanced    | material_lomo | v61_knn                | carbon        | 1400    | 1400   |
| 0.8700   | 0.8700            | 0.8692   | 0.8692      | class_material_balanced    | material_lomo | v61_knn                | wood          | 1400    | 1400   |
| 0.9386   | 0.9386            | 0.9389   | 0.9389      | class_material_balanced    | material_lomo | v62a_knn               | carbon        | 1400    | 1400   |
| 0.9350   | 0.9350            | 0.9357   | 0.9357      | class_material_balanced    | material_lomo | v62a_knn               | wood          | 1400    | 1400   |
| 0.9768   | 0.9760            | 0.9760   | 0.9768      | class_material_balanced    | stratified    | frequency_only_lr      | validation    | 2239    | 561    |
| 0.9554   | 0.9547            | 0.9547   | 0.9555      | class_material_balanced    | stratified    | frequency_plus_v62a_lr | validation    | 2239    | 561    |
| 0.9305   | 0.9291            | 0.9291   | 0.9304      | class_material_balanced    | stratified    | v61_knn                | validation    | 2239    | 561    |
| 0.9323   | 0.9320            | 0.9319   | 0.9324      | class_material_balanced    | stratified    | v62a_knn               | validation    | 2239    | 561    |
| 1.0000   | 1.0000            | 1.0000   | 1.0000      | class_material_freqmatched | board_lobo    | frequency_only_lr      | C01           | 750     | 170    |
| 1.0000   | 1.0000            | 1.0000   | 1.0000      | class_material_freqmatched | board_lobo    | frequency_only_lr      | C02           | 750     | 170    |
| 1.0000   | 1.0000            | 1.0000   | 1.0000      | class_material_freqmatched | board_lobo    | frequency_only_lr      | C03           | 795     | 125    |
| 0.9887   | 0.9902            | 0.9903   | 0.9887      | class_material_freqmatched | board_lobo    | frequency_only_lr      | W01           | 743     | 177    |
| 0.9818   | 0.9857            | 0.9824   | 0.9819      | class_material_freqmatched | board_lobo    | frequency_only_lr      | W02           | 755     | 165    |
| 0.9735   | 0.9600            | 0.9638   | 0.9729      | class_material_freqmatched | board_lobo    | frequency_only_lr      | W03           | 807     | 113    |
| 0.9765   | 0.9682            | 0.9620   | 0.9771      | class_material_freqmatched | board_lobo    | frequency_plus_v62a_lr | C01           | 750     | 170    |
| 0.9412   | 0.9429            | 0.9434   | 0.9417      | class_material_freqmatched | board_lobo    | frequency_plus_v62a_lr | C02           | 750     | 170    |
| 0.9360   | 0.9382            | 0.9390   | 0.9387      | class_material_freqmatched | board_lobo    | frequency_plus_v62a_lr | C03           | 795     | 125    |
| 0.9435   | 0.9480            | 0.9501   | 0.9441      | class_material_freqmatched | board_lobo    | frequency_plus_v62a_lr | W01           | 743     | 177    |
| 0.9636   | 0.9606            | 0.9621   | 0.9636      | class_material_freqmatched | board_lobo    | frequency_plus_v62a_lr | W02           | 755     | 165    |
| 0.9292   | 0.9129            | 0.9228   | 0.9280      | class_material_freqmatched | board_lobo    | frequency_plus_v62a_lr | W03           | 807     | 113    |
| 0.8765   | 0.8331            | 0.7912   | 0.8681      | class_material_freqmatched | board_lobo    | v61_knn                | C01           | 750     | 170    |
| 0.9294   | 0.9271            | 0.9284   | 0.9281      | class_material_freqmatched | board_lobo    | v61_knn                | C02           | 750     | 170    |
| 0.9360   | 0.9391            | 0.9392   | 0.9369      | class_material_freqmatched | board_lobo    | v61_knn                | C03           | 795     | 125    |
| 0.8418   | 0.8534            | 0.8586   | 0.8440      | class_material_freqmatched | board_lobo    | v61_knn                | W01           | 743     | 177    |
| 0.8242   | 0.8337            | 0.8333   | 0.8305      | class_material_freqmatched | board_lobo    | v61_knn                | W02           | 755     | 165    |
| 0.8761   | 0.8331            | 0.8353   | 0.8653      | class_material_freqmatched | board_lobo    | v61_knn                | W03           | 807     | 113    |
| 0.9647   | 0.9525            | 0.9486   | 0.9650      | class_material_freqmatched | board_lobo    | v62a_knn               | C01           | 750     | 170    |
| 0.9706   | 0.9727            | 0.9709   | 0.9711      | class_material_freqmatched | board_lobo    | v62a_knn               | C02           | 750     | 170    |
| 0.9520   | 0.9570            | 0.9551   | 0.9521      | class_material_freqmatched | board_lobo    | v62a_knn               | C03           | 795     | 125    |
| 0.9322   | 0.9372            | 0.9391   | 0.9331      | class_material_freqmatched | board_lobo    | v62a_knn               | W01           | 743     | 177    |
| 0.9455   | 0.9409            | 0.9448   | 0.9453      | class_material_freqmatched | board_lobo    | v62a_knn               | W02           | 755     | 165    |
| 0.9381   | 0.9264            | 0.9310   | 0.9383      | class_material_freqmatched | board_lobo    | v62a_knn               | W03           | 807     | 113    |
| 1.0000   | 1.0000            | 1.0000   | 1.0000      | class_material_freqmatched | material_lomo | frequency_only_lr      | carbon        | 455     | 465    |
| 0.9824   | 0.9837            | 0.9830   | 0.9824      | class_material_freqmatched | material_lomo | frequency_only_lr      | wood          | 465     | 455    |
| 0.9484   | 0.9487            | 0.9489   | 0.9487      | class_material_freqmatched | material_lomo | frequency_plus_v62a_lr | carbon        | 455     | 465    |
| 0.9187   | 0.9182            | 0.9190   | 0.9180      | class_material_freqmatched | material_lomo | frequency_plus_v62a_lr | wood          | 465     | 455    |
| 0.8667   | 0.8612            | 0.8595   | 0.8628      | class_material_freqmatched | material_lomo | v61_knn                | carbon        | 455     | 465    |
| 0.7846   | 0.7830            | 0.7767   | 0.7744      | class_material_freqmatched | material_lomo | v61_knn                | wood          | 465     | 455    |
| 0.9591   | 0.9588            | 0.9596   | 0.9593      | class_material_freqmatched | material_lomo | v62a_knn               | carbon        | 455     | 465    |
| 0.9341   | 0.9341            | 0.9350   | 0.9346      | class_material_freqmatched | material_lomo | v62a_knn               | wood          | 465     | 455    |
| 0.9949   | 0.9952            | 0.9945   | 0.9950      | class_material_freqmatched | stratified    | frequency_only_lr      | validation    | 722     | 198    |
| 0.9596   | 0.9534            | 0.9572   | 0.9593      | class_material_freqmatched | stratified    | frequency_plus_v62a_lr | validation    | 722     | 198    |
| 0.9343   | 0.9316            | 0.9309   | 0.9338      | class_material_freqmatched | stratified    | v61_knn                | validation    | 722     | 198    |
| 0.9495   | 0.9413            | 0.9420   | 0.9483      | class_material_freqmatched | stratified    | v62a_knn               | validation    | 722     | 198    |

## Output Files

- `reports\balanced_subset_v62a\balanced_subset_summary.csv`
- `reports\balanced_subset_v62a\balanced_subset_fold_metrics.csv`
- `reports\balanced_subset_v62a\balanced_subset_composition.csv`
- `reports\balanced_subset_v62a\balanced_subset_key_numbers.json`
