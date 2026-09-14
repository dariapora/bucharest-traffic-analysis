# xgboost

Script: `scripts/gbm_classifiers.py`
Params: `{'objective': 'multi:softprob', 'num_class': 3, 'device': 'cuda', 'tree_method': 'hist', 'learning_rate': 0.1, 'max_depth': 10, 'min_child_weight': 100, 'subsample': 0.7, 'colsample_bytree': 0.8, 'reg_lambda': 1.0, 'max_bin': 256, 'eval_metric': 'mlogloss', 'seed': 42}`
Max rounds 2000, early stopping 50 on one held-out weekday of the training set (mlogloss), balanced sample weights
Validation: leave-one-day-out; training-set features computed out-of-fold (same as logistic regression baseline)
Features (31): segment_mean_speed, segment_std_speed, segment_min_speed, segment_q10_speed, segment_congestion_rate, segment_slow_rate, seg_vs_global_hour, seg_vs_global_dow, seg_hour_vs_city_ratio, hist_seg_hour_std, seg_hour_cong_rate, seg_hour_min, seg_hour_q25, speed_limit, frc, distance, road_capacity, is_peak, time_slot, time_sin, time_cos, day_of_week_num, is_weekend, hist_seg_dow, hist_seg_weekend, hist_seg_hour, global_hour_mean, global_dow_hour_mean, hist_seg_hour_probes, hist_seg_hour_light_rate, sample_size

Loaded 15,356,640 rows, class distribution: free_flow 10,745,245 (70.0%), slow 3,778,562 (24.6%), congested 832,833 (5.4%)

```
============================================================
XGBOOST RESULTS
============================================================
Overall accuracy:    0.7327 (73.27%)
Weighted F1:         0.7483
Macro F1:            0.6357

============================================================
CLASSIFICATION REPORT
============================================================
              precision    recall  f1-score   support

   free_flow     0.9070    0.7627    0.8286  10745245
        slow     0.5083    0.6573    0.5733   3778562
   congested     0.3992    0.6882    0.5053    832833

    accuracy                         0.7327  15356640
   macro avg     0.6049    0.7027    0.6357  15356640
weighted avg     0.7814    0.7327    0.7483  15356640

============================================================
CONFUSION MATRIX
Rows = actual
Columns = predicted
============================================================
           free_flow     slow  congested
free_flow    8195194  2186720     363331
slow          795646  2483731     499185
congested      44159   215550     573124

============================================================
NORMALIZED CONFUSION MATRIX
Rows sum to 100%
============================================================
           free_flow  slow  congested
free_flow       76.3  20.4        3.4
slow            21.1  65.7       13.2
congested        5.3  25.9       68.8

============================================================
PER-DAY RESULTS
============================================================
       day  accuracy  weighted_f1  macro_f1  best_iteration
2024-08-25  0.735459     0.763781  0.551366            1999
2024-08-26  0.740921     0.753577  0.649145            1999
2024-08-27  0.756391     0.765033  0.672075            1999
2024-08-28  0.752768     0.761780  0.673956            1999
2024-08-29  0.752187     0.761356  0.675960            1998
2024-08-30  0.745430     0.756437  0.663248            1998
2024-08-31  0.645564     0.685568  0.520613            1999

============================================================
FEATURE IMPORTANCE (gain, mean over folds, normalised)
============================================================
            seg_hour_q25: 0.2169 ######################
             sample_size: 0.1667 #################
      seg_hour_cong_rate: 0.1297 #############
            seg_hour_min: 0.0630 ######
           hist_seg_hour: 0.0500 #####
        hist_seg_weekend: 0.0435 ####
    hist_seg_hour_probes: 0.0320 ###
       hist_seg_hour_std: 0.0287 ###
       segment_slow_rate: 0.0255 ###
 segment_congestion_rate: 0.0246 ##
       seg_vs_global_dow: 0.0212 ##
       segment_std_speed: 0.0210 ##
       segment_q10_speed: 0.0200 ##
  seg_hour_vs_city_ratio: 0.0199 ##
       segment_min_speed: 0.0164 ##
                distance: 0.0162 ##
      segment_mean_speed: 0.0159 ##
        global_hour_mean: 0.0158 ##
               time_slot: 0.0131 #
                time_cos: 0.0110 #
                time_sin: 0.0103 #
      seg_vs_global_hour: 0.0097 #
           road_capacity: 0.0066 #
         day_of_week_num: 0.0053 #
hist_seg_hour_light_rate: 0.0047 
                     frc: 0.0045 
                 is_peak: 0.0035 
             speed_limit: 0.0029 
            hist_seg_dow: 0.0010 
              is_weekend: 0.0004 
    global_dow_hour_mean: 0.0002 

Total time: 32.7 min
```
