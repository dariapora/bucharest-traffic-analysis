# xgboost_unweighted_thr80_50

Script: `scripts/gbm_classifiers.py`
Params: `{'objective': 'multi:softprob', 'num_class': 3, 'device': 'cuda', 'tree_method': 'hist', 'learning_rate': 0.1, 'max_depth': 10, 'min_child_weight': 100, 'subsample': 0.7, 'colsample_bytree': 0.8, 'reg_lambda': 1.0, 'max_bin': 256, 'eval_metric': 'mlogloss', 'seed': 42}`
Max rounds 2000, early stopping 50 on one held-out weekday of the training set (mlogloss), unweighted samples
Validation: leave-one-day-out; training-set features computed out-of-fold (same as logistic regression baseline)
Features (30): segment_mean_speed, segment_std_speed, segment_min_speed, segment_q10_speed, segment_congestion_rate, segment_slow_rate, seg_vs_global_hour, seg_vs_global_dow, seg_hour_vs_city_ratio, hist_seg_hour_std, seg_hour_cong_rate, seg_hour_min, seg_hour_q25, speed_limit, frc, distance, road_capacity, is_peak, time_slot, time_sin, time_cos, day_of_week_num, is_weekend, hist_seg_dow, hist_seg_weekend, hist_seg_hour, global_hour_mean, global_dow_hour_mean, hist_seg_hour_probes, hist_seg_hour_light_rate

Loaded 15,356,640 rows, class distribution: free_flow 12,027,990 (78.3%), slow 2,679,985 (17.5%), congested 648,665 (4.2%)

```
============================================================
XGBOOST_UNWEIGHTED_THR80_50 RESULTS
============================================================
Overall accuracy:    0.8091 (80.91%)
Weighted F1:         0.7725
Macro F1:            0.5175

============================================================
CLASSIFICATION REPORT
============================================================
              precision    recall  f1-score   support

   free_flow     0.8334    0.9696    0.8963  12027990
        slow     0.5516    0.2283    0.3229   2679985
   congested     0.5937    0.2315    0.3332    648665

    accuracy                         0.8091  15356640
   macro avg     0.6595    0.4765    0.5175  15356640
weighted avg     0.7741    0.8091    0.7725  15356640

============================================================
CONFUSION MATRIX
Rows = actual
Columns = predicted
============================================================
           free_flow    slow  congested
free_flow   11662533  314865      50592
slow         2015994  611776      52215
congested     315992  182477     150196

============================================================
NORMALIZED CONFUSION MATRIX
Rows sum to 100%
============================================================
           free_flow  slow  congested
free_flow       97.0   2.6        0.4
slow            75.2  22.8        1.9
congested       48.7  28.1       23.2

============================================================
PER-DAY RESULTS
============================================================
       day  accuracy  weighted_f1  macro_f1  best_iteration
2024-08-25  0.846667     0.841034  0.506228             122
2024-08-26  0.809971     0.768769  0.536075             264
2024-08-27  0.796183     0.747108  0.511957             489
2024-08-28  0.793133     0.743957  0.518035             453
2024-08-29  0.790352     0.740897  0.515445             185
2024-08-30  0.798003     0.751075  0.516207             860
2024-08-31  0.828122     0.815116  0.500384             185

============================================================
FEATURE IMPORTANCE (gain, mean over folds, normalised)
============================================================
            seg_hour_q25: 0.3166 ################################
      seg_hour_cong_rate: 0.1225 ############
    hist_seg_hour_probes: 0.0857 #########
           hist_seg_hour: 0.0785 ########
            seg_hour_min: 0.0573 ######
        hist_seg_weekend: 0.0514 #####
       segment_slow_rate: 0.0509 #####
       seg_vs_global_dow: 0.0506 #####
       hist_seg_hour_std: 0.0194 ##
        global_hour_mean: 0.0181 ##
 segment_congestion_rate: 0.0174 ##
hist_seg_hour_light_rate: 0.0168 ##
               time_slot: 0.0156 ##
       segment_std_speed: 0.0140 #
      segment_mean_speed: 0.0130 #
                time_cos: 0.0103 #
                time_sin: 0.0086 #
       segment_q10_speed: 0.0085 #
  seg_hour_vs_city_ratio: 0.0072 #
       segment_min_speed: 0.0072 #
                distance: 0.0065 #
      seg_vs_global_hour: 0.0050 
         day_of_week_num: 0.0048 
                 is_peak: 0.0042 
           road_capacity: 0.0035 
                     frc: 0.0034 
             speed_limit: 0.0016 
            hist_seg_dow: 0.0007 
              is_weekend: 0.0004 
    global_dow_hour_mean: 0.0001 

Total time: 36.2 min
```
