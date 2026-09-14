# lightgbm

Script: `scripts/gbm_classifiers.py`
Params: `{'objective': 'multiclass', 'num_class': 3, 'learning_rate': 0.1, 'num_leaves': 127, 'min_data_in_leaf': 500, 'feature_fraction': 0.8, 'bagging_fraction': 0.7, 'bagging_freq': 1, 'lambda_l2': 1.0, 'max_bin': 255, 'num_threads': 32, 'seed': 42, 'verbose': -1}`
Max rounds 2000, early stopping 50 on one held-out weekday of the training set (mlogloss), balanced sample weights
Validation: leave-one-day-out; training-set features computed out-of-fold (same as logistic regression baseline)
Features (30): segment_mean_speed, segment_std_speed, segment_min_speed, segment_q10_speed, segment_congestion_rate, segment_slow_rate, seg_vs_global_hour, seg_vs_global_dow, seg_hour_vs_city_ratio, hist_seg_hour_std, seg_hour_cong_rate, seg_hour_min, seg_hour_q25, speed_limit, frc, distance, road_capacity, is_peak, time_slot, time_sin, time_cos, day_of_week_num, is_weekend, hist_seg_dow, hist_seg_weekend, hist_seg_hour, global_hour_mean, global_dow_hour_mean, hist_seg_hour_probes, hist_seg_hour_light_rate

Loaded 15,356,640 rows, class distribution: free_flow 10,745,245 (70.0%), slow 3,778,562 (24.6%), congested 832,833 (5.4%)

```
============================================================
LIGHTGBM RESULTS
============================================================
Overall accuracy:    0.6924 (69.24%)
Weighted F1:         0.7146
Macro F1:            0.6012

============================================================
CLASSIFICATION REPORT
============================================================
              precision    recall  f1-score   support

   free_flow     0.9090    0.7057    0.7946  10745245
        slow     0.4688    0.6420    0.5419   3778562
   congested     0.3393    0.7495    0.4672    832833

    accuracy                         0.6924  15356640
   macro avg     0.5724    0.6991    0.6012  15356640
weighted avg     0.7698    0.6924    0.7146  15356640

============================================================
CONFUSION MATRIX
Rows = actual
Columns = predicted
============================================================
           free_flow     slow  congested
free_flow    7583054  2594543     567648
slow          704994  2425894     647674
congested      54214   154380     624239

============================================================
NORMALIZED CONFUSION MATRIX
Rows sum to 100%
============================================================
           free_flow  slow  congested
free_flow       70.6  24.1        5.3
slow            18.7  64.2       17.1
congested        6.5  18.5       75.0

============================================================
PER-DAY RESULTS
============================================================
       day  accuracy  weighted_f1  macro_f1  best_iteration
2024-08-25  0.716701     0.745075  0.523228            2000
2024-08-26  0.699678     0.719028  0.615168            2000
2024-08-27  0.720215     0.736112  0.646433            1999
2024-08-28  0.724517     0.738886  0.652469            1999
2024-08-29  0.723500     0.738035  0.654273            1998
2024-08-30  0.714391     0.730321  0.637691            2000
2024-08-31  0.547079     0.602729  0.439269            1998

============================================================
FEATURE IMPORTANCE (gain, mean over folds, normalised)
============================================================
      seg_hour_cong_rate: 0.2061 #####################
            seg_hour_q25: 0.1715 #################
            seg_hour_min: 0.0945 #########
    hist_seg_hour_probes: 0.0797 ########
           hist_seg_hour: 0.0584 ######
        hist_seg_weekend: 0.0438 ####
 segment_congestion_rate: 0.0332 ###
       segment_slow_rate: 0.0329 ###
  seg_hour_vs_city_ratio: 0.0296 ###
       seg_vs_global_dow: 0.0279 ###
       hist_seg_hour_std: 0.0275 ###
       segment_std_speed: 0.0215 ##
       segment_q10_speed: 0.0204 ##
        global_hour_mean: 0.0190 ##
      segment_mean_speed: 0.0166 ##
                distance: 0.0162 ##
               time_slot: 0.0161 ##
       segment_min_speed: 0.0158 ##
                time_cos: 0.0113 #
                time_sin: 0.0109 #
hist_seg_hour_light_rate: 0.0090 #
      seg_vs_global_hour: 0.0088 #
         day_of_week_num: 0.0072 #
           road_capacity: 0.0066 #
                     frc: 0.0049 
                 is_peak: 0.0043 
             speed_limit: 0.0028 
            hist_seg_dow: 0.0022 
              is_weekend: 0.0012 
    global_dow_hour_mean: 0.0001 

Total time: 91.9 min
```
