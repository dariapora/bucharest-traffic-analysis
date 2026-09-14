# xgboost_thr80_50

Script: `scripts/gbm_classifiers.py`
Params: `{'objective': 'multi:softprob', 'num_class': 3, 'device': 'cuda', 'tree_method': 'hist', 'learning_rate': 0.1, 'max_depth': 10, 'min_child_weight': 100, 'subsample': 0.7, 'colsample_bytree': 0.8, 'reg_lambda': 1.0, 'max_bin': 256, 'eval_metric': 'mlogloss', 'seed': 42}`
Max rounds 2000, early stopping 50 on one held-out weekday of the training set (mlogloss), balanced sample weights
Validation: leave-one-day-out; training-set features computed out-of-fold (same as logistic regression baseline)
Features (30): segment_mean_speed, segment_std_speed, segment_min_speed, segment_q10_speed, segment_congestion_rate, segment_slow_rate, seg_vs_global_hour, seg_vs_global_dow, seg_hour_vs_city_ratio, hist_seg_hour_std, seg_hour_cong_rate, seg_hour_min, seg_hour_q25, speed_limit, frc, distance, road_capacity, is_peak, time_slot, time_sin, time_cos, day_of_week_num, is_weekend, hist_seg_dow, hist_seg_weekend, hist_seg_hour, global_hour_mean, global_dow_hour_mean, hist_seg_hour_probes, hist_seg_hour_light_rate

Loaded 15,356,640 rows, class distribution: free_flow 12,027,990 (78.3%), slow 2,679,985 (17.5%), congested 648,665 (4.2%)

```
============================================================
XGBOOST_THR80_50 RESULTS
============================================================
Overall accuracy:    0.7599 (75.99%)
Weighted F1:         0.7778
Macro F1:            0.5971

============================================================
CLASSIFICATION REPORT
============================================================
              precision    recall  f1-score   support

   free_flow     0.9170    0.8102    0.8603  12027990
        slow     0.4314    0.5628    0.4884   2679985
   congested     0.3378    0.6417    0.4426    648665

    accuracy                         0.7599  15356640
   macro avg     0.5621    0.6716    0.5971  15356640
weighted avg     0.8078    0.7599    0.7778  15356640

============================================================
CONFUSION MATRIX
Rows = actual
Columns = predicted
============================================================
           free_flow     slow  congested
free_flow    9745658  1838328     444004
slow          799759  1508285     371941
congested      82785   149606     416274

============================================================
NORMALIZED CONFUSION MATRIX
Rows sum to 100%
============================================================
           free_flow  slow  congested
free_flow       81.0  15.3        3.7
slow            29.8  56.3       13.9
congested       12.8  23.1       64.2

============================================================
PER-DAY RESULTS
============================================================
       day  accuracy  weighted_f1  macro_f1  best_iteration
2024-08-25  0.772611     0.803515  0.514940            1999
2024-08-26  0.759560     0.775473  0.601356             123
2024-08-27  0.777326     0.787201  0.632443            1999
2024-08-28  0.777202     0.786680  0.637136            1999
2024-08-29  0.774821     0.784985  0.639771            1998
2024-08-30  0.770495     0.782746  0.624567            1999
2024-08-31  0.687283     0.735690  0.486712            1999

============================================================
FEATURE IMPORTANCE (gain, mean over folds, normalised)
============================================================
            seg_hour_q25: 0.2318 #######################
      seg_hour_cong_rate: 0.1607 ################
            seg_hour_min: 0.0903 #########
    hist_seg_hour_probes: 0.0578 ######
           hist_seg_hour: 0.0470 #####
        hist_seg_weekend: 0.0439 ####
       hist_seg_hour_std: 0.0371 ####
       segment_slow_rate: 0.0361 ####
 segment_congestion_rate: 0.0347 ###
       segment_std_speed: 0.0259 ###
       seg_vs_global_dow: 0.0252 ###
       segment_q10_speed: 0.0227 ##
        global_hour_mean: 0.0205 ##
       segment_min_speed: 0.0203 ##
                distance: 0.0201 ##
      segment_mean_speed: 0.0188 ##
               time_slot: 0.0178 ##
                time_sin: 0.0128 #
                time_cos: 0.0128 #
      seg_vs_global_hour: 0.0123 #
hist_seg_hour_light_rate: 0.0113 #
  seg_hour_vs_city_ratio: 0.0112 #
           road_capacity: 0.0080 #
         day_of_week_num: 0.0065 #
                     frc: 0.0054 #
                 is_peak: 0.0041 
             speed_limit: 0.0033 
            hist_seg_dow: 0.0010 
              is_weekend: 0.0004 
    global_dow_hour_mean: 0.0002 

Total time: 36.1 min
```
