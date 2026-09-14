# Logistic regression baseline

Script: `scripts/logistic_regression_baseline.py`
Model: multinomial logistic regression (PyTorch `nn.Linear`, full-batch L-BFGS on GPU), StandardScaler, balanced class weights
Validation: leave-one-day-out; training-set features computed out-of-fold (each training day's aggregates come from the other training days)
Labels: speed_ratio = median_speed / segment free-flow speed (85th percentile of the segment's median speed over intervals with >= 5 probes);
        >= 0.85 free_flow, >= 0.55 slow, else congested; intervals with < 3 probes are labelled free_flow (light traffic) and excluded from speed aggregates
Features: 30 historical / static / time-of-day features (no contemporaneous measurement of the predicted interval)

```
Using device: cuda
Loaded 15,356,640 rows
24,672 segments
7 days

Traffic-state distribution:
   free_flow: 10,745,245 (70.0%)
        slow:  3,778,562 (24.6%)
   congested:    832,833 (5.4%)
Testing day: 2024-08-25
  Accuracy:    0.6333
  Weighted F1: 0.6838
  Macro F1:    0.4600
Testing day: 2024-08-26
  Accuracy:    0.7126
  Weighted F1: 0.7271
  Macro F1:    0.6133
Testing day: 2024-08-27
  Accuracy:    0.7088
  Weighted F1: 0.7250
  Macro F1:    0.6288
Testing day: 2024-08-28
  Accuracy:    0.7137
  Weighted F1: 0.7283
  Macro F1:    0.6354
Testing day: 2024-08-29
  Accuracy:    0.7185
  Weighted F1: 0.7322
  Macro F1:    0.6405
Testing day: 2024-08-30
  Accuracy:    0.7172
  Weighted F1: 0.7315
  Macro F1:    0.6299
Testing day: 2024-08-31
  Accuracy:    0.7144
  Weighted F1: 0.7370
  Macro F1:    0.5353


============================================================
LOGISTIC REGRESSION RESULTS
============================================================
Overall accuracy:    0.7024 (70.24%)
Weighted F1:         0.7217
Macro F1:            0.5984


============================================================
CLASSIFICATION REPORT
============================================================
              precision    recall  f1-score   support

   free_flow     0.8914    0.7385    0.8078  10745245
        slow     0.4918    0.5904    0.5366   3778562
   congested     0.3234    0.7447    0.4509    832833

    accuracy                         0.7024  15356640
   macro avg     0.5689    0.6912    0.5984  15356640
weighted avg     0.7623    0.7024    0.7217  15356640



============================================================
CONFUSION MATRIX
Rows = actual
Columns = predicted
============================================================
           free_flow     slow  congested
free_flow    7935760  2154850     654635
slow          904567  2230784     643211
congested      62052   150558     620223


============================================================
NORMALIZED CONFUSION MATRIX
Rows sum to 100%
============================================================
           free_flow  slow  congested
free_flow       73.9  20.1        6.1
slow            23.9  59.0       17.0
congested        7.5  18.1       74.5


============================================================
PER-DAY RESULTS
============================================================
       day  accuracy  weighted_f1  macro_f1
2024-08-25  0.633273     0.683789  0.459951
2024-08-26  0.712632     0.727144  0.613256
2024-08-27  0.708771     0.725047  0.628816
2024-08-28  0.713737     0.728270  0.635357
2024-08-29  0.718505     0.732244  0.640479
2024-08-30  0.717157     0.731490  0.629937
2024-08-31  0.714401     0.736951  0.535339
```
