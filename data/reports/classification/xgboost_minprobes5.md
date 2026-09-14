# xgboost_minprobes5

Script: `scripts/gbm_classifiers.py`
Params: `{'objective': 'multi:softprob', 'num_class': 3, 'device': 'cuda', 'tree_method': 'hist', 'learning_rate': 0.1, 'max_depth': 10, 'min_child_weight': 100, 'subsample': 0.7, 'colsample_bytree': 0.8, 'reg_lambda': 1.0, 'max_bin': 256, 'eval_metric': 'mlogloss', 'seed': 42}`
Max rounds 2000, early stopping 50 on one held-out weekday of the training set (mlogloss), balanced sample weights
Validation: leave-one-day-out; training-set features computed out-of-fold (same as logistic regression baseline)
Features (30): segment_mean_speed, segment_std_speed, segment_min_speed, segment_q10_speed, segment_congestion_rate, segment_slow_rate, seg_vs_global_hour, seg_vs_global_dow, seg_hour_vs_city_ratio, hist_seg_hour_std, seg_hour_cong_rate, seg_hour_min, seg_hour_q25, speed_limit, frc, distance, road_capacity, is_peak, time_slot, time_sin, time_cos, day_of_week_num, is_weekend, hist_seg_dow, hist_seg_weekend, hist_seg_hour, global_hour_mean, global_dow_hour_mean, hist_seg_hour_probes, hist_seg_hour_light_rate

Loaded 15,356,640 rows, class distribution: free_flow 11,229,546 (73.1%), slow 3,395,051 (22.1%), congested 732,043 (4.8%)

```
============================================================
XGBOOST_MINPROBES5 RESULTS
============================================================
Overall accuracy:    0.7607 (76.07%)
Weighted F1:         0.7727
Macro F1:            0.6329

============================================================
CLASSIFICATION REPORT
============================================================
              precision    recall  f1-score   support

   free_flow     0.9029    0.8166    0.8576  11229546
        slow     0.5223    0.5881    0.5532   3395051
   congested     0.3736    0.7028    0.4878    732043

    accuracy                         0.7607  15356640
   macro avg     0.5996    0.7025    0.6329  15356640
weighted avg     0.7935    0.7607    0.7727  15356640

============================================================
CONFUSION MATRIX
Rows = actual
Columns = predicted
============================================================
           free_flow     slow  congested
free_flow    9170283  1673114     386149
slow          921931  1996508     476612
congested      64279   153258     514506

============================================================
NORMALIZED CONFUSION MATRIX
Rows sum to 100%
============================================================
           free_flow  slow  congested
free_flow       81.7  14.9        3.4
slow            27.2  58.8       14.0
congested        8.8  20.9       70.3

============================================================
PER-DAY RESULTS
============================================================
       day  accuracy  weighted_f1  macro_f1  best_iteration
2024-08-25  0.762843     0.788548  0.536068            1999
2024-08-26  0.757166     0.767884  0.632111              70
2024-08-27  0.778216     0.783711  0.670470            1999
2024-08-28  0.776893     0.782261  0.674104            1999
2024-08-29  0.774593     0.780325  0.674889            1999
2024-08-30  0.769506     0.777032  0.662045            1999
2024-08-31  0.705491     0.741140  0.523993            1998

============================================================
FEATURE IMPORTANCE (gain, mean over folds, normalised)
============================================================
            seg_hour_q25: 0.2098 #####################
      seg_hour_cong_rate: 0.1692 #################
            seg_hour_min: 0.1002 ##########
    hist_seg_hour_probes: 0.0698 #######
           hist_seg_hour: 0.0457 #####
        hist_seg_weekend: 0.0453 #####
  seg_hour_vs_city_ratio: 0.0426 ####
       segment_slow_rate: 0.0336 ###
       hist_seg_hour_std: 0.0306 ###
       seg_vs_global_dow: 0.0281 ###
 segment_congestion_rate: 0.0275 ###
       segment_std_speed: 0.0215 ##
       segment_q10_speed: 0.0181 ##
        global_hour_mean: 0.0174 ##
       segment_min_speed: 0.0169 ##
               time_slot: 0.0166 ##
                distance: 0.0163 ##
hist_seg_hour_light_rate: 0.0160 ##
      segment_mean_speed: 0.0156 ##
                time_cos: 0.0120 #
                time_sin: 0.0117 #
      seg_vs_global_hour: 0.0093 #
           road_capacity: 0.0067 #
         day_of_week_num: 0.0064 #
                     frc: 0.0049 
                 is_peak: 0.0041 
             speed_limit: 0.0028 
            hist_seg_dow: 0.0009 
              is_weekend: 0.0004 
    global_dow_hour_mean: 0.0001 

Total time: 35.1 min
```
