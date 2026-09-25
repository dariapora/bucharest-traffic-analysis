Loaded 10,424,846 rows, 23,506 segments, 7 days

Traffic state distribution:
     free_flow:  6,099,760 (58.5%)
          slow:  2,844,652 (27.3%)
     congested:  1,480,434 (14.2%)

Class weights:
     free_flow: 0.570
          slow: 1.222
     congested: 2.347

Days in dataset: ['2024-08-25', '2024-08-26', '2024-08-27', '2024-08-28', '2024-08-29', '2024-08-30', '2024-08-31']
Running leave-one-day-out cross-validation...

  2024-08-25 (Sunday, weekend): accuracy=0.6775  F1-weighted=0.6879  F1-macro=0.5723
  2024-08-26 (Monday, weekday): accuracy=0.5902  F1-weighted=0.6086  F1-macro=0.5482
  2024-08-27 (Tuesday, weekday): accuracy=0.6025  F1-weighted=0.6181  F1-macro=0.5676
  2024-08-28 (Wednesday, weekday): accuracy=0.5943  F1-weighted=0.6114  F1-macro=0.5652
  2024-08-29 (Thursday, weekday): accuracy=0.5964  F1-weighted=0.6132  F1-macro=0.5669
  2024-08-30 (Friday, weekday): accuracy=0.5880  F1-weighted=0.6067  F1-macro=0.5563
  2024-08-31 (Saturday, weekend): accuracy=0.5997  F1-weighted=0.6266  F1-macro=0.5339

============================================================
OVERALL ACCURACY:    0.6068 (60.7%)
OVERALL F1 WEIGHTED: 0.6251
OVERALL F1 MACRO:    0.5612
============================================================

Confusion matrix (rows = actual, columns = predicted):
           free_flow     slow  congested
free_flow    3887672  1194691    1017397
slow          560853  1435575     848224
congested     193748   283781    1002905

Normalized confusion matrix (% of each actual class):
           free_flow  slow  congested
free_flow       63.7  19.6       16.7
slow            19.7  50.5       29.8
congested       13.1  19.2       67.7

Classification report:
              precision    recall  f1-score   support

   free_flow     0.8375    0.6373    0.7238   6099760
        slow     0.4926    0.5047    0.4986   2844652
   congested     0.3496    0.6774    0.4612   1480434

    accuracy                         0.6068  10424846
   macro avg     0.5599    0.6065    0.5612  10424846
weighted avg     0.6741    0.6068    0.6251  10424846

Feature importance (retrained on full dataset):
           speed_limit: 0.2406 ████████████
                   frc: 0.1523 ███████
              distance: 0.1291 ██████
            is_weekend: 0.1163 █████
        segment_id_enc: 0.1129 █████
              time_cos: 0.1079 █████
              time_sin: 0.0804 ████
       day_of_week_num: 0.0604 ███

Weekday vs weekend:
  Weekday: accuracy=0.5942 (59.4%)  F1-macro=0.5609
  Weekend: accuracy=0.6388 (63.9%)  F1-macro=0.5521

Accuracy by time of day (1-hour):
  00:00 - 01:00: 0.7531 (75.3%)
  01:00 - 02:00: 0.7765 (77.6%)
  02:00 - 03:00: 0.7879 (78.8%)
  03:00 - 04:00: 0.8006 (80.1%)
  04:00 - 05:00: 0.8128 (81.3%)
  05:00 - 06:00: 0.7992 (79.9%)
  06:00 - 07:00: 0.7143 (71.4%)
  07:00 - 08:00: 0.6153 (61.5%)
  08:00 - 09:00: 0.5424 (54.2%)
  09:00 - 10:00: 0.5325 (53.2%)
  10:00 - 11:00: 0.5431 (54.3%)
  11:00 - 12:00: 0.5398 (54.0%)
  12:00 - 13:00: 0.5355 (53.6%)
  13:00 - 14:00: 0.5330 (53.3%)
  14:00 - 15:00: 0.5266 (52.7%)
  15:00 - 16:00: 0.5192 (51.9%)
  16:00 - 17:00: 0.4888 (48.9%)
  17:00 - 18:00: 0.4735 (47.4%)
  18:00 - 19:00: 0.5061 (50.6%)
  19:00 - 20:00: 0.5825 (58.2%)
  20:00 - 21:00: 0.6170 (61.7%)
  21:00 - 22:00: 0.6618 (66.2%)
  22:00 - 23:00: 0.6990 (69.9%)
  23:00 - 24:00: 0.7357 (73.6%)

Per-segment accuracy saved to data/reports/dataframe_filtered_segment_accuracy_v2.csv
  (24,013 segments)
  Most predictable:  1.0000
  Least predictable: 0.0000
  Median accuracy:   0.5975