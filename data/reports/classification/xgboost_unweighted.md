# xgboost_unweighted

Script: `scripts/gbm_classifiers.py`
Params: `{'objective': 'multi:softprob', 'num_class': 3, 'device': 'cuda', 'tree_method': 'hist', 'learning_rate': 0.1, 'max_depth': 10, 'min_child_weight': 100, 'subsample': 0.7, 'colsample_bytree': 0.8, 'reg_lambda': 1.0, 'max_bin': 256, 'eval_metric': 'mlogloss', 'seed': 42}`
Max rounds 2000, early stopping 50 on one held-out weekday of the training set (mlogloss), unweighted samples
Validation: leave-one-day-out; training-set features computed out-of-fold (same as logistic regression baseline)
Features (30): segment_mean_speed, segment_std_speed, segment_min_speed, segment_q10_speed, segment_congestion_rate, segment_slow_rate, seg_vs_global_hour, seg_vs_global_dow, seg_hour_vs_city_ratio, hist_seg_hour_std, seg_hour_cong_rate, seg_hour_min, seg_hour_q25, speed_limit, frc, distance, road_capacity, is_peak, time_slot, time_sin, time_cos, day_of_week_num, is_weekend, hist_seg_dow, hist_seg_weekend, hist_seg_hour, global_hour_mean, global_dow_hour_mean, hist_seg_hour_probes, hist_seg_hour_light_rate

Loaded 15,356,640 rows, class distribution: free_flow 10,745,245 (70.0%), slow 3,778,562 (24.6%), congested 832,833 (5.4%)

```
============================================================
XGBOOST_UNWEIGHTED RESULTS
============================================================
Overall accuracy:    0.7505 (75.05%)
Weighted F1:         0.7189
Macro F1:            0.5480

============================================================
CLASSIFICATION REPORT
============================================================
              precision    recall  f1-score   support

   free_flow     0.7824    0.9394    0.8537  10745245
        slow     0.5787    0.3175    0.4100   3778562
   congested     0.6057    0.2773    0.3804    832833

    accuracy                         0.7505  15356640
   macro avg     0.6556    0.5114    0.5480  15356640
weighted avg     0.7227    0.7505    0.7189  15356640

============================================================
CONFUSION MATRIX
Rows = actual
Columns = predicted
============================================================
           free_flow     slow  congested
free_flow   10094133   584049      67063
slow         2495610  1199685      83267
congested     312538   289387     230908

============================================================
NORMALIZED CONFUSION MATRIX
Rows sum to 100%
============================================================
           free_flow  slow  congested
free_flow       93.9   5.4        0.6
slow            66.0  31.7        2.2
congested       37.5  34.7       27.7

============================================================
PER-DAY RESULTS
============================================================
       day  accuracy  weighted_f1  macro_f1  best_iteration
2024-08-25  0.788335     0.782585  0.521950             203
2024-08-26  0.757442     0.718105  0.565961             603
2024-08-27  0.742505     0.696223  0.545830             453
2024-08-28  0.738279     0.691890  0.545320             522
2024-08-29  0.737644     0.692317  0.549743             407
2024-08-30  0.744843     0.701959  0.552950             811
2024-08-31  0.743071     0.746554  0.526892             289

============================================================
FEATURE IMPORTANCE (gain, mean over folds, normalised)
============================================================
            seg_hour_q25: 0.2751 ############################
      seg_hour_cong_rate: 0.1277 #############
    hist_seg_hour_probes: 0.1267 #############
           hist_seg_hour: 0.0701 #######
            seg_hour_min: 0.0642 ######
        hist_seg_weekend: 0.0529 #####
       seg_vs_global_dow: 0.0501 #####
       segment_slow_rate: 0.0335 ###
hist_seg_hour_light_rate: 0.0251 ###
        global_hour_mean: 0.0192 ##
       hist_seg_hour_std: 0.0178 ##
 segment_congestion_rate: 0.0168 ##
               time_slot: 0.0168 ##
       segment_std_speed: 0.0132 #
      segment_mean_speed: 0.0123 #
                time_cos: 0.0113 #
                time_sin: 0.0102 #
       segment_q10_speed: 0.0083 #
  seg_hour_vs_city_ratio: 0.0077 #
       segment_min_speed: 0.0076 #
                distance: 0.0070 #
      seg_vs_global_hour: 0.0057 #
         day_of_week_num: 0.0053 #
                 is_peak: 0.0039 
                     frc: 0.0039 
           road_capacity: 0.0039 
             speed_limit: 0.0019 
            hist_seg_dow: 0.0008 
              is_weekend: 0.0005 
    global_dow_hour_mean: 0.0001 

Total time: 16.7 min
```
