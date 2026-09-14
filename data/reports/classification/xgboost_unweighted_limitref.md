# xgboost_unweighted_limitref

Script: `scripts/gbm_classifiers.py`
Params: `{'objective': 'multi:softprob', 'num_class': 3, 'device': 'cuda', 'tree_method': 'hist', 'learning_rate': 0.1, 'max_depth': 10, 'min_child_weight': 100, 'subsample': 0.7, 'colsample_bytree': 0.8, 'reg_lambda': 1.0, 'max_bin': 256, 'eval_metric': 'mlogloss', 'seed': 42}`
Max rounds 2000, early stopping 50 on one held-out weekday of the training set (mlogloss), unweighted samples
Validation: leave-one-day-out; training-set features computed out-of-fold (same as logistic regression baseline)
Features (30): segment_mean_speed, segment_std_speed, segment_min_speed, segment_q10_speed, segment_congestion_rate, segment_slow_rate, seg_vs_global_hour, seg_vs_global_dow, seg_hour_vs_city_ratio, hist_seg_hour_std, seg_hour_cong_rate, seg_hour_min, seg_hour_q25, speed_limit, frc, distance, road_capacity, is_peak, time_slot, time_sin, time_cos, day_of_week_num, is_weekend, hist_seg_dow, hist_seg_weekend, hist_seg_hour, global_hour_mean, global_dow_hour_mean, hist_seg_hour_probes, hist_seg_hour_light_rate

Loaded 15,356,640 rows, class distribution: free_flow 8,897,521 (57.9%), slow 4,151,590 (27.0%), congested 2,307,529 (15.0%)

```
============================================================
XGBOOST_UNWEIGHTED_LIMITREF RESULTS
============================================================
Overall accuracy:    0.7951 (79.51%)
Weighted F1:         0.7897
Macro F1:            0.7385

============================================================
CLASSIFICATION REPORT
============================================================
              precision    recall  f1-score   support

   free_flow     0.8410    0.9161    0.8770   8897521
        slow     0.7000    0.6432    0.6704   4151590
   congested     0.7508    0.6019    0.6681   2307529

    accuracy                         0.7951  15356640
   macro avg     0.7639    0.7204    0.7385  15356640
weighted avg     0.7893    0.7951    0.7897  15356640

============================================================
CONFUSION MATRIX
Rows = actual
Columns = predicted
============================================================
           free_flow     slow  congested
free_flow    8151322   590612     155587
slow         1176057  2670139     305394
congested     364816   553845    1388868

============================================================
NORMALIZED CONFUSION MATRIX
Rows sum to 100%
============================================================
           free_flow  slow  congested
free_flow       91.6   6.6        1.7
slow            28.3  64.3        7.4
congested       15.8  24.0       60.2

============================================================
PER-DAY RESULTS
============================================================
       day  accuracy  weighted_f1  macro_f1  best_iteration
2024-08-25  0.783438     0.788049  0.715497             243
2024-08-26  0.802009     0.794614  0.746425             296
2024-08-27  0.795272     0.786926  0.738316             781
2024-08-28  0.794505     0.785748  0.740031             544
2024-08-29  0.793502     0.784804  0.741274             343
2024-08-30  0.800048     0.791719  0.749134             939
2024-08-31  0.797171     0.797882  0.737010             532

============================================================
FEATURE IMPORTANCE (gain, mean over folds, normalised)
============================================================
            seg_hour_q25: 0.2667 ###########################
      seg_hour_cong_rate: 0.1857 ###################
       segment_slow_rate: 0.1573 ################
    hist_seg_hour_probes: 0.0922 #########
           hist_seg_hour: 0.0746 #######
            seg_hour_min: 0.0296 ###
hist_seg_hour_light_rate: 0.0271 ###
       seg_vs_global_dow: 0.0262 ###
        hist_seg_weekend: 0.0206 ##
 segment_congestion_rate: 0.0198 ##
               time_slot: 0.0128 #
      segment_mean_speed: 0.0117 #
        global_hour_mean: 0.0110 #
       hist_seg_hour_std: 0.0088 #
  seg_hour_vs_city_ratio: 0.0081 #
                time_cos: 0.0076 #
                time_sin: 0.0062 #
       segment_std_speed: 0.0058 #
         day_of_week_num: 0.0047 
       segment_min_speed: 0.0038 
                distance: 0.0037 
       segment_q10_speed: 0.0036 
      seg_vs_global_hour: 0.0032 
           road_capacity: 0.0023 
                 is_peak: 0.0022 
             speed_limit: 0.0021 
                     frc: 0.0018 
            hist_seg_dow: 0.0004 
              is_weekend: 0.0004 
    global_dow_hour_mean: 0.0001 

Total time: 40.3 min
```
