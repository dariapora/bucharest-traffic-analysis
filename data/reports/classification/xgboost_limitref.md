# xgboost_limitref

Script: `scripts/gbm_classifiers.py`
Params: `{'objective': 'multi:softprob', 'num_class': 3, 'device': 'cuda', 'tree_method': 'hist', 'learning_rate': 0.1, 'max_depth': 10, 'min_child_weight': 100, 'subsample': 0.7, 'colsample_bytree': 0.8, 'reg_lambda': 1.0, 'max_bin': 256, 'eval_metric': 'mlogloss', 'seed': 42}`
Max rounds 2000, early stopping 50 on one held-out weekday of the training set (mlogloss), balanced sample weights
Validation: leave-one-day-out; training-set features computed out-of-fold (same as logistic regression baseline)
Features (30): segment_mean_speed, segment_std_speed, segment_min_speed, segment_q10_speed, segment_congestion_rate, segment_slow_rate, seg_vs_global_hour, seg_vs_global_dow, seg_hour_vs_city_ratio, hist_seg_hour_std, seg_hour_cong_rate, seg_hour_min, seg_hour_q25, speed_limit, frc, distance, road_capacity, is_peak, time_slot, time_sin, time_cos, day_of_week_num, is_weekend, hist_seg_dow, hist_seg_weekend, hist_seg_hour, global_hour_mean, global_dow_hour_mean, hist_seg_hour_probes, hist_seg_hour_light_rate

Loaded 15,356,640 rows, class distribution: free_flow 8,897,521 (57.9%), slow 4,151,590 (27.0%), congested 2,307,529 (15.0%)

```
============================================================
XGBOOST_LIMITREF RESULTS
============================================================
Overall accuracy:    0.7925 (79.25%)
Weighted F1:         0.7953
Macro F1:            0.7518

============================================================
CLASSIFICATION REPORT
============================================================
              precision    recall  f1-score   support

   free_flow     0.8948    0.8472    0.8704   8897521
        slow     0.6752    0.7053    0.6899   4151590
   congested     0.6566    0.7386    0.6952   2307529

    accuracy                         0.7925  15356640
   macro avg     0.7422    0.7637    0.7518  15356640
weighted avg     0.7996    0.7925    0.7953  15356640

============================================================
CONFUSION MATRIX
Rows = actual
Columns = predicted
============================================================
           free_flow     slow  congested
free_flow    7538179   991526     367816
slow          700125  2928040     523425
congested     185997   417182    1704350

============================================================
NORMALIZED CONFUSION MATRIX
Rows sum to 100%
============================================================
           free_flow  slow  congested
free_flow       84.7  11.1        4.1
slow            16.9  70.5       12.6
congested        8.1  18.1       73.9

============================================================
PER-DAY RESULTS
============================================================
       day  accuracy  weighted_f1  macro_f1  best_iteration
2024-08-25  0.763573     0.774226  0.702915            1994
2024-08-26  0.804066     0.805017  0.764128            1999
2024-08-27  0.803342     0.803878  0.765087            1997
2024-08-28  0.805487     0.805434  0.769429            1998
2024-08-29  0.805000     0.805017  0.770611            1999
2024-08-30  0.809064     0.809272  0.774473            1994
2024-08-31  0.757627     0.767857  0.710967            1998

============================================================
FEATURE IMPORTANCE (gain, mean over folds, normalised)
============================================================
            seg_hour_q25: 0.2559 ##########################
      seg_hour_cong_rate: 0.2055 #####################
       segment_slow_rate: 0.1259 #############
           hist_seg_hour: 0.0755 ########
    hist_seg_hour_probes: 0.0732 #######
            seg_hour_min: 0.0335 ###
 segment_congestion_rate: 0.0249 ##
        hist_seg_weekend: 0.0242 ##
       seg_vs_global_dow: 0.0241 ##
hist_seg_hour_light_rate: 0.0197 ##
      segment_mean_speed: 0.0194 ##
               time_slot: 0.0126 #
       hist_seg_hour_std: 0.0124 #
        global_hour_mean: 0.0121 #
       segment_std_speed: 0.0107 #
                distance: 0.0096 #
                time_cos: 0.0090 #
       segment_min_speed: 0.0082 #
       segment_q10_speed: 0.0078 #
                time_sin: 0.0075 #
      seg_vs_global_hour: 0.0056 #
  seg_hour_vs_city_ratio: 0.0054 #
         day_of_week_num: 0.0050 
           road_capacity: 0.0039 
                     frc: 0.0027 
                 is_peak: 0.0025 
             speed_limit: 0.0024 
            hist_seg_dow: 0.0006 
              is_weekend: 0.0004 
    global_dow_hour_mean: 0.0001 

Total time: 40.2 min
```
