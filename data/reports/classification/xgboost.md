# xgboost

Script: `scripts/gbm_classifiers.py`
Params: `{'objective': 'multi:softprob', 'num_class': 3, 'device': 'cuda', 'tree_method': 'hist', 'learning_rate': 0.1, 'max_depth': 10, 'min_child_weight': 100, 'subsample': 0.7, 'colsample_bytree': 0.8, 'reg_lambda': 1.0, 'max_bin': 256, 'eval_metric': 'mlogloss', 'seed': 42}`
Max rounds 2000, early stopping 50 on one held-out weekday of the training set (mlogloss), balanced sample weights
Validation: leave-one-day-out; training-set features computed out-of-fold (same as logistic regression baseline)
Features (30): segment_mean_speed, segment_std_speed, segment_min_speed, segment_q10_speed, segment_congestion_rate, segment_slow_rate, seg_vs_global_hour, seg_vs_global_dow, seg_hour_vs_city_ratio, hist_seg_hour_std, seg_hour_cong_rate, seg_hour_min, seg_hour_q25, speed_limit, frc, distance, road_capacity, is_peak, time_slot, time_sin, time_cos, day_of_week_num, is_weekend, hist_seg_dow, hist_seg_weekend, hist_seg_hour, global_hour_mean, global_dow_hour_mean, hist_seg_hour_probes, hist_seg_hour_light_rate

Loaded 15,356,640 rows, class distribution: free_flow 10,745,245 (70.0%), slow 3,778,562 (24.6%), congested 832,833 (5.4%)

```
============================================================
XGBOOST RESULTS
============================================================
Overall accuracy:    0.7217 (72.17%)
Weighted F1:         0.7338
Macro F1:            0.6113

============================================================
CLASSIFICATION REPORT
============================================================
              precision    recall  f1-score   support

   free_flow     0.8728    0.7803    0.8240  10745245
        slow     0.5058    0.5656    0.5340   3778562
   congested     0.3680    0.6738    0.4760    832833

    accuracy                         0.7217  15356640
   macro avg     0.5822    0.6732    0.6113  15356640
weighted avg     0.7551    0.7217    0.7338  15356640

============================================================
CONFUSION MATRIX
Rows = actual
Columns = predicted
============================================================
           free_flow     slow  congested
free_flow    8384956  1912381     447908
slow         1125779  2136990     515793
congested      95699   175984     561150

============================================================
NORMALIZED CONFUSION MATRIX
Rows sum to 100%
============================================================
           free_flow  slow  congested
free_flow       78.0  17.8        4.2
slow            29.8  56.6       13.7
congested       11.5  21.1       67.4

============================================================
PER-DAY RESULTS
============================================================
       day  accuracy  weighted_f1  macro_f1  best_iteration
2024-08-25  0.722517     0.748972  0.525637            1999
2024-08-26  0.735463     0.743161  0.627026            1999
2024-08-27  0.745586     0.749799  0.647735            1999
2024-08-28  0.742164     0.746819  0.648852            1998
2024-08-29  0.742771     0.747714  0.652925            1999
2024-08-30  0.738277     0.744913  0.640981            1999
2024-08-31  0.624834     0.665455  0.493192            1999

============================================================
FEATURE IMPORTANCE (gain, mean over folds, normalised)
============================================================
            seg_hour_q25: 0.1891 ###################
      seg_hour_cong_rate: 0.1725 #################
            seg_hour_min: 0.0763 ########
    hist_seg_hour_probes: 0.0737 #######
           hist_seg_hour: 0.0603 ######
        hist_seg_weekend: 0.0463 #####
       segment_slow_rate: 0.0356 ####
       hist_seg_hour_std: 0.0349 ###
 segment_congestion_rate: 0.0309 ###
  seg_hour_vs_city_ratio: 0.0306 ###
       seg_vs_global_dow: 0.0282 ###
       segment_std_speed: 0.0238 ##
       segment_q10_speed: 0.0222 ##
        global_hour_mean: 0.0196 ##
                distance: 0.0195 ##
       segment_min_speed: 0.0193 ##
      segment_mean_speed: 0.0184 ##
               time_slot: 0.0180 ##
hist_seg_hour_light_rate: 0.0159 ##
                time_cos: 0.0129 #
                time_sin: 0.0129 #
      seg_vs_global_hour: 0.0110 #
           road_capacity: 0.0078 #
         day_of_week_num: 0.0063 #
                     frc: 0.0055 #
                 is_peak: 0.0039 
             speed_limit: 0.0033 
            hist_seg_dow: 0.0010 
              is_weekend: 0.0004 
    global_dow_hour_mean: 0.0001 

Total time: 92.1 min
```
