# xgboost_unweighted_minprobes5

Script: `scripts/gbm_classifiers.py`
Params: `{'objective': 'multi:softprob', 'num_class': 3, 'device': 'cuda', 'tree_method': 'hist', 'learning_rate': 0.1, 'max_depth': 10, 'min_child_weight': 100, 'subsample': 0.7, 'colsample_bytree': 0.8, 'reg_lambda': 1.0, 'max_bin': 256, 'eval_metric': 'mlogloss', 'seed': 42}`
Max rounds 2000, early stopping 50 on one held-out weekday of the training set (mlogloss), unweighted samples
Validation: leave-one-day-out; training-set features computed out-of-fold (same as logistic regression baseline)
Features (30): segment_mean_speed, segment_std_speed, segment_min_speed, segment_q10_speed, segment_congestion_rate, segment_slow_rate, seg_vs_global_hour, seg_vs_global_dow, seg_hour_vs_city_ratio, hist_seg_hour_std, seg_hour_cong_rate, seg_hour_min, seg_hour_q25, speed_limit, frc, distance, road_capacity, is_peak, time_slot, time_sin, time_cos, day_of_week_num, is_weekend, hist_seg_dow, hist_seg_weekend, hist_seg_hour, global_hour_mean, global_dow_hour_mean, hist_seg_hour_probes, hist_seg_hour_light_rate

Loaded 15,356,640 rows, class distribution: free_flow 11,229,546 (73.1%), slow 3,395,051 (22.1%), congested 732,043 (4.8%)

```
============================================================
XGBOOST_UNWEIGHTED_MINPROBES5 RESULTS
============================================================
Overall accuracy:    0.7835 (78.35%)
Weighted F1:         0.7559
Macro F1:            0.5646

============================================================
CLASSIFICATION REPORT
============================================================
              precision    recall  f1-score   support

   free_flow     0.8152    0.9507    0.8777  11229546
        slow     0.5981    0.3397    0.4333   3395051
   congested     0.6129    0.2783    0.3828    732043

    accuracy                         0.7835  15356640
   macro avg     0.6754    0.5229    0.5646  15356640
weighted avg     0.7575    0.7835    0.7559  15356640

============================================================
CONFUSION MATRIX
Rows = actual
Columns = predicted
============================================================
           free_flow     slow  congested
free_flow   10675751   497751      56044
slow         2169236  1153150      72665
congested     251293   276995     203755

============================================================
NORMALIZED CONFUSION MATRIX
Rows sum to 100%
============================================================
           free_flow  slow  congested
free_flow       95.1   4.4        0.5
slow            63.9  34.0        2.1
congested       34.3  37.8       27.8

============================================================
PER-DAY RESULTS
============================================================
       day  accuracy  weighted_f1  macro_f1  best_iteration
2024-08-25  0.819196     0.817702  0.532447             201
2024-08-26  0.788664     0.755966  0.589339             518
2024-08-27  0.773693     0.733750  0.561543             685
2024-08-28  0.769519     0.729058  0.561546             407
2024-08-29  0.768277     0.728595  0.569568             336
2024-08-30  0.774498     0.736726  0.568260             603
2024-08-31  0.789909     0.790067  0.536909             454

============================================================
FEATURE IMPORTANCE (gain, mean over folds, normalised)
============================================================
            seg_hour_q25: 0.2937 #############################
    hist_seg_hour_probes: 0.1177 ############
      seg_hour_cong_rate: 0.1145 ###########
            seg_hour_min: 0.1145 ###########
       seg_vs_global_dow: 0.0508 #####
        hist_seg_weekend: 0.0475 #####
           hist_seg_hour: 0.0452 #####
       segment_slow_rate: 0.0318 ###
hist_seg_hour_light_rate: 0.0264 ###
        global_hour_mean: 0.0185 ##
               time_slot: 0.0158 ##
       hist_seg_hour_std: 0.0153 ##
 segment_congestion_rate: 0.0146 #
                time_cos: 0.0115 #
       segment_std_speed: 0.0113 #
      segment_mean_speed: 0.0105 #
                time_sin: 0.0104 #
       segment_min_speed: 0.0066 #
       segment_q10_speed: 0.0066 #
  seg_hour_vs_city_ratio: 0.0064 #
                distance: 0.0062 #
      seg_vs_global_hour: 0.0052 #
         day_of_week_num: 0.0052 #
                 is_peak: 0.0038 
                     frc: 0.0036 
           road_capacity: 0.0035 
             speed_limit: 0.0017 
            hist_seg_dow: 0.0007 
              is_weekend: 0.0004 
    global_dow_hour_mean: 0.0001 

Total time: 35.3 min
```
