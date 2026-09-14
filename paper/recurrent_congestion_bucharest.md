# How Recurrent Is Urban Congestion? Predicting the Traffic State of Bucharest Road Segments from Their Own History

Authors: [AUTHORS]
Affiliation: Bucharest University of Economic Studies, 2026

---

## Abstract

Urban congestion has a recurrent component, driven by daily routines, and a non-recurrent component, driven by incidents, events and weather. How large the recurrent share is determines what can be achieved by day-ahead prediction and by planning interventions. This study quantifies that share for Bucharest using a week of TomTom Traffic Stats probe data (25–31 August 2024; 15,356,640 segment-intervals at 15-minute resolution across 24,672 road segments). Each interval was labelled free-flow, slow or congested relative to the segment's own free-flow speed, and three classifiers — multinomial logistic regression, LightGBM and XGBoost — were trained to predict the label of an unseen day using only the segment's historical profile, static road attributes and time of day, with no measurement from the predicted interval itself. Under leave-one-day-out validation the models reach 70–73% accuracy and a macro-F1 of 0.60–0.61; a purely deterministic rule that replays each segment's most frequent state at the same time on the other days reaches 75.4% accuracy and a macro-F1 of 0.59. Trained without class weights, XGBoost matches the rule exactly (75.1% accuracy, macro-F1 0.55). The machine-learning models therefore add little discriminative information beyond the historical profile; their main effect, through class weighting, is to change the operating point, raising congestion recall from 36% to 67–75% at the cost of precision. Adding the number of probe vehicles observed in the predicted interval — a real-time signal — improves macro-F1 by only 0.02, and the comparison is unchanged under three alternative label definitions. Predictability is highest at night and on motorways and local streets, and lowest on mid-hierarchy arterials and on the segments that are congested most often. Weekend days, of which the week contains one of each, are markedly less predictable than weekdays. The results indicate that in the observed week roughly three quarters of segment-level traffic states in Bucharest are reproducible from history alone, and that the remaining quarter — concentrated in the peak hours, on collector roads and in the "slow" transition band — is where real-time data are needed. The analysis also documents two data-quality pitfalls of probe-based labels: intervals without probe vehicles and speed ratios computed against the posted limit both create spurious night-time "congestion" that inflates model accuracy.

**Keywords:** traffic state classification, recurrent congestion, probe vehicle data, gradient boosting, logistic regression, leave-one-day-out validation, feature engineering, urban mobility, Bucharest

---

## 1. Introduction

Bucharest is among the most congested capitals in Europe. Congestion is costly in time, fuel and emissions, and its mitigation is a stated priority of the city administration. Most interventions available to a city — signal timing, public-transport priority, delivery windows, road works scheduling, information to drivers — are planned in advance. Their effectiveness therefore depends on how much of tomorrow's congestion can be anticipated today. If traffic states are largely the product of routine (commuting schedules, school runs, shopping hours), they can be predicted from history and planned around. If they are largely the product of disruptions, only real-time detection and response can help.

This paper asks that question for Bucharest in a directly measurable form: **given a road segment, a day of the week and a time of day, how well can the segment's traffic state be predicted from its own past behaviour alone?** We answer it with a week of TomTom Traffic Stats probe data covering the whole city at 15-minute resolution. Each segment-interval is assigned one of three states — free-flow, slow, congested — and we predict the state on a held-out day using only information that would be available the day before: the segment's historical speed and volume profile, static road attributes, and the time of the interval.

Three model families are compared in the same protocol: a multinomial logistic regression, which can only combine features linearly; and two gradient-boosted tree ensembles, LightGBM and XGBoost, which can learn interactions between segment, hour and day type. The three are evaluated against two deterministic reference rules: always predicting free-flow, and replaying each segment's most frequent state at the same time of day on the other days. The gap between the reference rules and the models measures what machine learning contributes over a simple historical profile; the gap between the models and perfect prediction measures the non-recurrent share of congestion.

Beyond the headline numbers the study contributes three things. First, a careful label definition for probe-based traffic states, showing that two natural choices — treating intervals with no probe vehicles as observations, and normalising speed by the posted limit — produce artefacts that dominate model accuracy if left uncorrected. Second, a leakage-free protocol for target-encoded historical features, in which every training row's features are computed from days other than its own. Third, a decomposition of predictability by hour of day, road class, weekday/weekend and segment, which locates the non-recurrent share of congestion in space and time.

The study is deliberately limited to one week in late August. August is the main holiday month in Romania and traffic volumes in Bucharest are lower than in the rest of the year; the week also contains a single Saturday and a single Sunday. These limitations are discussed in Section 6 and shape how the results should be read: as a measurement of the recurrent share of congestion in a light-traffic week, not as an all-year forecast model.

---

## 2. Literature review

> **Note to authors.** No references are cited in this draft; the literature review is to be assembled from the sources you retrieve. Each paragraph below states what the paragraph must establish, followed by the Scite.ai prompts that should surface suitable, citable sources. Replace the prompts with the retrieved references and adapt the text.

**2.1 Recurrent versus non-recurrent congestion.** The paragraph should introduce the distinction between recurrent congestion (predictable from time-of-day/day-of-week patterns) and non-recurrent congestion (incidents, weather, events), cite estimates of the share of delay attributable to each, and note that this share is location-dependent.

- Scite prompt: *"What proportion of urban traffic delay is attributable to recurrent versus non-recurrent congestion, and how is the distinction defined?"*
- Scite prompt: *"Empirical studies decomposing travel-time variability into recurrent and incident-related components in European cities"*

**2.2 Probe-vehicle (floating car) data for traffic-state estimation.** The paragraph should establish that GPS probe data from navigation providers are now a standard source for network-wide speed monitoring, summarise known limitations (penetration rate, low sample counts on minor roads and at night, bias toward certain vehicle fleets) and how minimum-sample thresholds are typically handled.

- Scite prompt: *"Reliability of floating car data speed estimates as a function of the number of probe vehicles per road segment and time interval"*
- Scite prompt: *"TomTom or HERE probe data used for city-wide traffic state estimation: coverage, penetration rate and validation against loop detectors"*

**2.3 Definitions of traffic state from speed.** The paragraph should review how speed-based congestion indices are defined — speed relative to free-flow speed, speed performance index, travel-time index — and justify the use of a segment-specific free-flow reference (e.g. an upper percentile of observed speeds) rather than the posted speed limit.

- Scite prompt: *"Speed performance index and free-flow speed estimation from probe data for road-segment congestion classification"*
- Scite prompt: *"Why posted speed limits are a poor proxy for free-flow speed on urban streets"*

**2.4 Machine learning for short-term and day-ahead traffic-state prediction.** The paragraph should summarise the use of gradient-boosted trees (XGBoost, LightGBM) and linear models for tabular traffic prediction, including target-encoded historical features, and note the recurring finding that historical averages are a strong baseline that sophisticated models only modestly improve upon.

- Scite prompt: *"Gradient boosting (XGBoost, LightGBM) for road-segment traffic congestion classification using historical aggregate features"*
- Scite prompt: *"Historical average as a baseline for traffic prediction: how much do machine learning models improve over it?"*
- Scite prompt: *"Target encoding leakage in time-series cross-validation for traffic prediction"*

**2.5 Bucharest and Romanian context.** The paragraph should cite any prior quantitative studies of Bucharest traffic, the TomTom Traffic Index ranking of Bucharest, and evidence on seasonal (summer-holiday) variation in Romanian urban traffic.

- Scite prompt: *"Traffic congestion in Bucharest: empirical studies, TomTom Traffic Index, seasonal variation"*
- Scite prompt: *"Seasonal reduction of urban traffic volumes during August holidays in Southern and Eastern Europe"*

---

## 3. Methodology

### 3.1 Data

Traffic data were obtained from the TomTom Traffic Stats API for the administrative area of Bucharest, delimited by a GeoJSON outline, for the seven consecutive days from Sunday 25 August to Saturday 31 August 2024. The API returns, for every monitored road segment and every 15-minute interval, the median, average and harmonic-average speed of the probe vehicles that traversed the segment, the standard deviation of speed, the number of probe vehicles (*sample size*), and travel-time statistics, together with segment metadata: identifier, street name, functional road class (FRC 0 = motorway to FRC 6 = local street), posted speed limit, segment length and a representative coordinate. The exports were consolidated into a single table with one row per segment and interval.

The table contains 15,356,640 rows: 24,672 segments × 7 days × 96 intervals, a complete grid. Segment lengths are short (median 38 m, mean 51 m), so a street is represented by many consecutive segments. Speed limits are predominantly 50 km/h (74% of rows) and 30 km/h (14%). Functional road classes 2–6 hold 99% of the segments.

Probe counts per interval are highly skewed (median 11, mean 20.2, maximum 341) and vary strongly by hour and road class. 11.7% of all rows have **zero** probe vehicles; for these the API reports zero for every speed and travel-time field. A further 12.2% have one or two probes. Zero-probe intervals concentrate at night (29% of segments at 04:00, 6% at 17:00) and on local streets (34% of FRC 6 rows against 2% of FRC 2). Table 1 summarises the dataset.

**Table 1. Dataset summary.**

| Property | Value |
|---|---|
| Collection period | Sunday 25 – Saturday 31 August 2024 |
| Temporal resolution | 15 minutes (96 intervals per day) |
| Segment-intervals | 15,356,640 |
| Road segments | 24,672 (median length 38 m) |
| Probe vehicles per interval | median 11, mean 20.2 |
| Intervals with 0 probes | 1,804,225 (11.7%) |
| Intervals with 1–2 probes | 1,859,448 (12.1%) |
| Intervals with ≥ 5 probes | 10,424,846 (67.9%) |

> **[FIGURE 1 — Probe coverage.]** Two panels from `paper/figure_data/probes_by_hour.csv` and `frc_summary.csv`: (a) mean probes per interval and share of zero-probe intervals by hour of day; (b) share of zero-probe intervals by functional road class. Purpose: show that data sparsity is structured (night, minor roads), which motivates the labelling rules in 3.2.

### 3.2 Traffic-state label

The target is a three-state ordinal label derived from the segment's speed ratio. Two choices in its construction turned out to matter more than any modelling decision, and are therefore stated explicitly.

**Free-flow reference.** The speed ratio is defined as the interval's median speed divided by the segment's *own* free-flow speed, estimated as the 85th percentile of its median speeds over all intervals of the week with at least five probes (fallback: over any measured interval; then the posted limit — needed for 8 segments). The posted limit was rejected as a reference because it measures road design rather than traffic: 24% of segments never reach 70% of their posted limit even when empty — short segments at signalised intersections, streets with speed bumps, tight curves — and would be labelled slow or congested at every hour under a limit-based ratio. The median free-flow speed is 43 km/h; relative to the posted limit it ranges from a median of 1.23 on motorways to 0.75 on local streets (Table 2).

**Table 2. Segment free-flow speed (85th percentile of median speed) relative to the posted limit, by functional road class.**

| FRC | segments | median free-flow speed (km/h) | median ratio to posted limit |
|---|---|---|---|
| 0 (motorway) | 15 | 90.0 | 1.23 |
| 1 | 413 | 80.0 | 1.14 |
| 2 | 4,685 | 51.0 | 0.98 |
| 3 | 4,220 | 50.0 | 1.00 |
| 4 | 7,157 | 45.0 | 0.92 |
| 5 | 2,214 | 39.0 | 0.84 |
| 6 (local) | 5,968 | 32.1 | 0.75 |

**Thresholds.** Free-flow: ratio ≥ 0.85; slow: 0.55 ≤ ratio < 0.85; congested: ratio < 0.55.

**Light-traffic rule.** An interval with fewer than three probe vehicles is labelled free-flow regardless of speed, and its speed is excluded from all historical aggregates. Zero-probe intervals are not measurements at all (all fields are zero) and, physically, a segment traversed by no probe vehicle in 15 minutes is a near-empty road, the opposite of a congested one. Intervals with one or two probes are measurements of one or two vehicles: their "congested" share is 12% at 04:00 and rises only 1.6-fold by the evening peak, whereas for intervals with 20 or more probes it is 1.2% at 04:00 and rises 13-fold (Figure 2). Below three probes the speed reading is dominated by individual driver behaviour (parking, turning, stopping) rather than by the traffic state. The rule affects 23.9% of rows.

> **[FIGURE 2 — Label reliability versus probe count.]** From `paper/figure_data/congested_share_by_probe_bucket_and_hour.csv`: line chart, x = hour of day, y = share of intervals with speed ratio < 0.55, one line per probe-count bucket (0, 1–2, 3–4, 5–9, 10–19, 20+). Purpose: show that "congestion" in low-probe intervals is flat across the day (noise) while in well-sampled intervals it follows the commuting profile (signal).

The resulting class distribution is 70.0% free-flow (10,745,245), 24.6% slow (3,778,562) and 5.4% congested (832,833). Its daily profile is that of a functioning city: congested intervals are 1.6% of segments at 04:00, 7.3% at 08:00 and 11.9% at 17:00; slow intervals rise from 6% to 35% over the same period (Figure 3). Under the two rejected choices — posted-limit reference and zero-probe intervals taken at face value — the congested share was 38% at 04:00, higher than at any daytime hour, and a fifth of the "congestion" in the week was missing data.

> **[FIGURE 3 — Traffic-state profile.]** From `paper/figure_data/class_share_by_hour_daytype.csv`: stacked area (or three lines) of free-flow / slow / congested share by hour, two panels: weekdays and weekend. Purpose: establish the daily rhythm the models must reproduce, and the visibly different weekend profile.

### 3.3 Features

Thirty features describe each segment-interval (Table 3). None is a measurement of the interval being predicted; all are either static, a function of the clock, or an aggregate over *other* days. They fall into five groups.

*Segment profile* (six features): the mean, standard deviation, minimum and 10th percentile of the segment's speed ratio, and the fraction of its intervals in the congested and slow states.

*Segment × time profile* (nine features): the mean speed ratio of the segment by hour of day, by day of week and by weekend/weekday; the standard deviation, minimum, 25th percentile and congested fraction of the segment at that hour; and the mean probe count and the fraction of light-traffic intervals (< 3 probes) of the segment at that hour. The last two are the historical volume proxies that replace the interval's own probe count.

*City-wide context and deviations* (five features): the city-wide mean ratio at that hour and at that day-of-week × hour, and the segment's deviation from and ratio to those means.

*Static road attributes* (four features): posted speed limit, functional road class, segment length and their product as a capacity proxy.

*Time* (six features): the 15-minute slot, sine and cosine of the time of day, day-of-week index, weekend flag and a peak-hour flag (07:00–10:00, 16:00–19:00).

**Table 3. Feature set.**

| Group | Feature | Description |
|---|---|---|
| Segment profile | segment_mean_speed | mean speed ratio of the segment |
| | segment_std_speed | standard deviation of the ratio |
| | segment_min_speed | minimum ratio |
| | segment_q10_speed | 10th percentile of the ratio |
| | segment_congestion_rate | share of intervals congested |
| | segment_slow_rate | share of intervals slow |
| Segment × time | hist_seg_hour | mean ratio at this hour |
| | hist_seg_dow | mean ratio on this day of week |
| | hist_seg_weekend | mean ratio on this day type |
| | hist_seg_hour_std | std of ratio at this hour |
| | seg_hour_min | minimum ratio at this hour |
| | seg_hour_q25 | 25th percentile of ratio at this hour |
| | seg_hour_cong_rate | share congested at this hour |
| | hist_seg_hour_probes | mean probe count at this hour |
| | hist_seg_hour_light_rate | share of intervals with < 3 probes at this hour |
| City context | global_hour_mean | city-wide mean ratio at this hour |
| | global_dow_hour_mean | city-wide mean ratio at this day-of-week and hour |
| | seg_vs_global_hour | hist_seg_hour − global_hour_mean |
| | seg_vs_global_dow | hist_seg_dow − global_dow_hour_mean |
| | seg_hour_vs_city_ratio | hist_seg_hour / global_hour_mean |
| Static | speed_limit, frc, distance | posted limit, functional class, length |
| | road_capacity | frc × speed_limit |
| Time | time_slot, time_sin, time_cos | 15-min slot and cyclical encoding |
| | day_of_week_num, is_weekend, is_peak | calendar flags |

**Out-of-fold construction.** Historical aggregates are target encodings: they are computed from the label variable (the speed ratio). If a training row's aggregates include the row itself, features such as *seg_hour_min* or *seg_hour_q25* partially encode the row's own label — a segment-hour minimum is bounded above by every interval in it — and the model learns a relationship that does not exist at prediction time. To prevent this, every training day's features are computed from the other training days only (leave-one-day-out *within* the training set), and the test day's features from all training days. In a preliminary experiment with in-sample aggregates the logistic regression reached 80% accuracy on its training data and 54% on the held-out day; with out-of-fold aggregates the gap closed to 78% / 74% on the same labels. All results reported below use out-of-fold features.

### 3.4 Models

Three classifiers were trained on identical features and folds.

*Multinomial logistic regression.* Features standardised; softmax over the three states; parameters fitted by full-batch L-BFGS (strong-Wolfe line search, up to 500 iterations) on a GPU. The problem is convex, so the solution is the global optimum; with 13 million training rows the L2 penalty of a standard implementation is negligible and was omitted. This model can only weight features additively and serves as the linear reference.

*LightGBM.* Leaf-wise gradient boosting, multiclass objective, 127 leaves, minimum 500 observations per leaf, learning rate 0.1, 80% feature and 70% row subsampling per iteration, L2 regularisation 1.0, up to 2,000 rounds.

*XGBoost.* Depth-wise gradient boosting on GPU (histogram method, 256 bins), multiclass softmax objective, maximum depth 10, minimum child weight 100, learning rate 0.1, 80% column and 70% row subsampling, L2 regularisation 1.0, up to 2,000 rounds.

For both boosted models the number of rounds was selected by early stopping (patience 50) on the multiclass log-loss of one held-out weekday from the training set; in every fold the models were still improving marginally at the 2,000-round cap.

All three models were trained with **balanced class weights** (each class weighted inversely to its frequency), so that the rare congested class (5.4%) is not ignored. This choice moves the models' operating point toward congestion recall and away from raw accuracy; to make its effect visible, XGBoost was additionally trained without class weights, and a further XGBoost variant was trained with the interval's own probe count (*sample_size*) added as a 31st feature, to quantify the value of one real-time signal.

**Table 4. Model configurations.**

| Model | Key settings |
|---|---|
| Logistic regression | standardised inputs, softmax, L-BFGS (≤ 500 it.), balanced class weights |
| LightGBM | 127 leaves, min 500 obs/leaf, η = 0.1, feature 0.8 / bagging 0.7, λ₂ = 1, ≤ 2000 rounds, early stopping 50 |
| XGBoost | depth 10, min child weight 100, η = 0.1, colsample 0.8 / subsample 0.7, λ₂ = 1, 256 bins, ≤ 2000 rounds, early stopping 50, GPU |
| XGBoost (unweighted) | as above, no class weights |
| XGBoost + probe count | as above, plus the interval's *sample_size* |

### 3.5 Validation protocol and reference rules

Performance was estimated by leave-one-day-out cross-validation: seven folds, each training on six days and predicting the seventh. This mirrors day-ahead use and guarantees that no information from the predicted day enters the model, either directly or through the historical aggregates (Section 3.3). Predictions from the seven folds are pooled to compute overall accuracy, weighted F1, macro F1, per-class precision and recall, and confusion matrices; per-day, per-hour and per-segment accuracy are computed from the same pooled predictions.

Two deterministic rules provide the reference points against which the models are read:

- **Always free-flow** — the majority class; its accuracy (70.0%) is the floor any useful model must exceed.
- **Historical mode** — for each segment and 15-minute slot, the most frequent state on the six other days. This rule contains no learning and uses only the segment's own past; it is the operational definition of "recurrent" used in this study. Its accuracy measures how much of the traffic state is reproducible from the profile alone.

### 3.6 Visualisation platform

> **[Optional section — include if the interactive map is part of the submission.]** Describe the web application used to inspect the network state (map of segments coloured by state, time slider, per-segment profile), the backend (SQLite + REST) and the front-end libraries. The per-segment accuracy file produced in this study (`data/reports/classification/segment_accuracy.csv`, with coordinates) can be loaded into it to map predictability.

---

## 4. Results

### 4.1 Traffic states in the observed week

The week's traffic-state mix differs sharply by day type (Table 5). On weekdays 65–68% of segment-intervals are free-flowing, 26–28% slow and 5.7–6.8% congested; on Sunday 80% are free-flowing and 2.7% congested, and Saturday lies between (77% / 3.4%) with a midday hump — congested share 6.2% at 12:00 against 3.8% on Sunday — consistent with shopping and leisure traffic. Weekday congestion has a bimodal profile with a morning shoulder (7.3% at 08:00) and a broader evening peak (11.9% at 17:00); the slow state, at 35% of segments through the afternoon, is the dominant non-free condition.

**Table 5. Traffic-state mix by day.**

| Day | free-flow | slow | congested | mean probes / interval |
|---|---|---|---|---|
| Sun 25 Aug | 80.5% | 16.8% | 2.7% | 18.3 |
| Mon 26 Aug | 68.0% | 26.2% | 5.8% | 19.9 |
| Tue 27 Aug | 66.6% | 27.0% | 6.4% | 19.8 |
| Wed 28 Aug | 65.8% | 27.7% | 6.6% | 20.5 |
| Thu 29 Aug | 65.4% | 27.8% | 6.8% | 21.2 |
| Fri 30 Aug | 66.3% | 27.2% | 6.4% | 21.8 |
| Sat 31 Aug | 77.0% | 19.7% | 3.4% | 19.9 |

> **[FIGURE 4 — Network snapshots.]** Two map screenshots from the visualisation platform at 08:00 on a weekday and at 08:00 on Sunday, segments coloured by state (free-flow / slow / congested), with the share of each state in the caption. Purpose: qualitative view of the spatial structure of the weekday peak.

### 4.2 Model comparison

Table 6 reports pooled results over the seven folds. The always-free-flow rule sets the floor at 70.0% accuracy. The historical-mode rule — no learning, just each segment's usual state at that time — reaches **75.4% accuracy, weighted F1 0.738 and macro F1 0.586**, while detecting 36% of congested intervals.

**Table 6. Pooled leave-one-day-out performance (15,356,640 predictions).**

| Model | Accuracy | Weighted F1 | Macro F1 | Recall free-flow | Recall slow | Recall congested | Precision congested |
|---|---|---|---|---|---|---|---|
| Always free-flow | 0.700 | 0.576 | 0.274 | 1.00 | 0.00 | 0.00 | – |
| Historical mode | **0.754** | 0.738 | 0.586 | 0.91 | 0.41 | 0.36 | 0.52 |
| Logistic regression | 0.702 | 0.722 | 0.598 | 0.74 | 0.59 | 0.74 | 0.32 |
| LightGBM | 0.692 | 0.715 | 0.601 | 0.71 | 0.64 | 0.75 | 0.34 |
| XGBoost | 0.722 | 0.734 | 0.611 | 0.78 | 0.57 | 0.67 | 0.37 |
| XGBoost, unweighted | 0.751 | 0.719 | 0.548 | 0.94 | 0.32 | 0.28 | 0.61 |
| XGBoost + probe count | 0.733 | 0.748 | **0.636** | 0.76 | 0.66 | 0.69 | 0.40 |

The three learned models, trained with balanced class weights, reach 69–72% accuracy and a macro F1 of 0.60–0.61 — below the historical-mode rule in accuracy and weighted F1, and only 0.01–0.03 above it in macro F1. What they change is the *operating point*: congestion recall rises from 36% to 67–75%, and slow recall from 41% to 57–64%, while free-flow recall falls from 91% to 71–78%. The confusion matrices (Table 7) show where the trade is made: with the historical-mode rule most errors are congested or slow intervals predicted as free-flow; with the weighted models most errors are free-flow intervals predicted as slow (18–24% of them) and slow intervals predicted as congested (14–17%). Between the three learned models the differences are small: XGBoost is the most accurate (72.2%), LightGBM the most congestion-sensitive (75% recall), and the linear model sits in between on every metric. Model non-linearity is not what limits performance here.

**Table 7. Row-normalised confusion matrices (% of actual class).**

| | → free-flow | → slow | → congested |
|---|---|---|---|
| **Historical mode** | | | |
| free-flow | 90.7 | 8.2 | 1.1 |
| slow | 55.1 | 40.6 | 4.3 |
| congested | 31.3 | 32.2 | 36.5 |
| **XGBoost, unweighted** | | | |
| free-flow | 93.9 | 5.4 | 0.6 |
| slow | 66.0 | 31.7 | 2.2 |
| congested | 37.5 | 34.7 | 27.7 |
| **Logistic regression** | | | |
| free-flow | 73.9 | 20.1 | 6.1 |
| slow | 23.9 | 59.0 | 17.0 |
| congested | 7.5 | 18.1 | 74.5 |
| **LightGBM** | | | |
| free-flow | 70.6 | 24.1 | 5.3 |
| slow | 18.7 | 64.2 | 17.1 |
| congested | 6.5 | 18.5 | 75.0 |
| **XGBoost** | | | |
| free-flow | 78.0 | 17.8 | 4.2 |
| slow | 29.8 | 56.6 | 13.7 |
| congested | 11.5 | 21.1 | 67.4 |

> **[FIGURE 5 — Confusion matrices.]** Five row-normalised heatmaps (historical mode, logistic regression, LightGBM, XGBoost, XGBoost unweighted) from Table 7. Purpose: show the operating-point shift and that free-flow ↔ congested confusion is rare (4–12%) — errors are between adjacent states.

Removing the class weights from XGBoost makes the comparison with the lookup rule direct, because both are then tuned to the same objective — being right as often as possible. The unweighted model reaches 75.1% accuracy, the historical-mode rule 75.4%; its macro F1 (0.548) is *below* the rule's (0.586), and its confusion matrix (Table 7) is that of the rule with the free-flow bias slightly amplified: 94% of free-flow intervals correct, but only 32% of slow and 28% of congested intervals detected. At the accuracy-maximising operating point, a 2,000-round gradient-boosted ensemble with thirty features reproduces the segment's usual state and nothing more.

Adding the interval's own probe count — a real-time volume signal — to XGBoost raises accuracy by 1.1 points and macro F1 by 0.025, and becomes the second most important feature (17% of gain). This is a modest gain for a signal that is unavailable in day-ahead use, and indicates that most of what volume tells about the state is already carried by the historical volume profile.

### 4.3 Per-day results and the weekend problem

Accuracy on the five weekdays is stable for every model (Table 8): 71–72% for logistic regression, 70–72% for LightGBM, 74–75% for XGBoost, with macro F1 between 0.61 and 0.65. The two weekend days are different. Sunday is the worst day for the linear model (63%, macro F1 0.46); Saturday is the worst day for both tree models (55% and 62%, macro F1 0.44 and 0.49), whereas logistic regression handles Saturday as well as a weekday (71%).

**Table 8. Accuracy (macro F1) by held-out day.**

| Day | Always free-flow | Logistic regression | LightGBM | XGBoost |
|---|---|---|---|---|
| Sun 25 Aug | 0.805 | 0.633 (0.460) | 0.717 (0.523) | 0.723 (0.526) |
| Mon 26 Aug | 0.680 | 0.713 (0.613) | 0.700 (0.615) | 0.735 (0.627) |
| Tue 27 Aug | 0.666 | 0.709 (0.629) | 0.720 (0.646) | 0.746 (0.648) |
| Wed 28 Aug | 0.658 | 0.714 (0.635) | 0.725 (0.652) | 0.742 (0.649) |
| Thu 29 Aug | 0.654 | 0.719 (0.640) | 0.724 (0.654) | 0.743 (0.653) |
| Fri 30 Aug | 0.663 | 0.717 (0.630) | 0.714 (0.638) | 0.738 (0.641) |
| Sat 31 Aug | 0.770 | 0.714 (0.535) | 0.547 (0.439) | 0.625 (0.493) |
| Weekdays pooled | 0.664 | 0.714 (0.630) | 0.716 (0.641) | 0.741 (0.644) |
| Weekend pooled | 0.788 | 0.673 (0.496) | 0.633 (0.480) | 0.674 (0.511) |

> **[FIGURE 6 — Per-day accuracy.]** Grouped bars from Table 8 (and `data/reports/classification/*.md` per-day tables): one group per day, bars for always-free-flow, historical mode, logistic regression, LightGBM, XGBoost. Purpose: weekday stability versus weekend collapse; Saturday vs Sunday asymmetry.

The weekend results are a direct consequence of the data window. Each weekday fold has four other weekdays in training; each weekend fold has exactly one other weekend day — of the *other* kind. A model asked to predict Saturday has learned "weekend" from Sunday, which lacks Saturday's midday hump (Section 4.1); the tree models, which split sharply on the weekend flag and on the segment's weekend profile, reproduce Sunday's quiet profile and under-predict Saturday's slow and congested intervals, while the linear model, which cannot isolate the weekend as strongly, degrades more gracefully. The historical-mode rule shows the same pattern (74.7% on weekends against 75.7% on weekdays, macro F1 0.51 against 0.61). With one example of each weekend day, weekend recurrence is not identifiable from these data; the weekday results are the reliable part of the study.

### 4.4 Predictability by hour of day

Figure 7 plots accuracy and macro F1 by hour of day, pooled over all folds. Two regimes are visible.

At night (23:00–06:00) the network is almost entirely free-flowing, and accuracy is high for every rule and model — 86–93% for always-free-flow, 85–92% for the historical mode, 78–89% for the class-weighted models, which lose a few points by occasionally predicting a slow state. Predictability here is trivial: there is nothing to predict.

During the day (07:00–21:00) the always-free-flow rule drops to 53–72%, the historical mode holds 64–74%, and the class-weighted models 58–72% in accuracy — but their macro F1 rises to its daily maximum in the evening peak (XGBoost 0.63 at 17:00, against 0.48 at 04:00), because the slow and congested classes are populated and the models discriminate them. XGBoost's accuracy from 08:00 to 19:00 is a remarkably flat 66–67%, and its margin over always-free-flow is largest exactly in the evening peak (66% vs 53% at 17:00). The non-recurrent share of congestion is therefore not concentrated in one hour: across the whole working day roughly one segment-interval in three is not where its history says it should be, and this fraction barely moves between 09:00 and 19:00.

**Table 9. Accuracy by hour of day (selected hours).**

| Hour | Always free-flow | Historical mode | Logistic regression | LightGBM | XGBoost | congested share | slow share |
|---|---|---|---|---|---|---|---|
| 04 | 0.926 | 0.922 | 0.886 | 0.889 | 0.881 | 1.6% | 5.8% |
| 08 | 0.636 | 0.677 | 0.640 | 0.598 | 0.664 | 7.3% | 29.1% |
| 12 | 0.550 | 0.683 | 0.622 | 0.622 | 0.665 | 8.0% | 37.0% |
| 17 | 0.533 | 0.641 | 0.590 | 0.606 | 0.663 | 11.9% | 34.8% |
| 21 | 0.702 | 0.738 | 0.673 | 0.634 | 0.649 | 2.9% | 26.8% |

> **[FIGURE 7 — Predictability by hour.]** From `data/reports/classification/accuracy_by_hour.csv`: (a) accuracy by hour, one line per rule/model, with the congested + slow share as a shaded background; (b) macro F1 by hour (values in `analysis.md`). Purpose: night vs day regimes; flat daytime accuracy; macro F1 peaking in the evening peak.

### 4.5 Predictability by segment

Pooling the seven folds gives an accuracy for each of the 24,672 segments over its 672 intervals (Figure 8). For XGBoost the median segment accuracy is 71%; 53% of segments are predicted at 70% or better, 8.5% below 50%. Two structural gradients explain most of the spread.

*Road hierarchy.* Segments at the top (FRC 0–1: motorways and major arterials, median 85–93%) and bottom (FRC 6: local streets, median 83%) of the hierarchy are the most predictable — the former because they carry regular, high-volume commuting flows, the latter because they are almost always free-flowing. The middle of the hierarchy — FRC 3–5, the secondary and collector roads that connect neighbourhoods to arterials — is the least predictable (median 66–70%).

*Congestion frequency.* Predictability falls monotonically with how often a segment is congested: segments that are never congested are predicted at 89% (median), those congested 2–5% of the time at 67%, and those congested more than 10% of the time at 57–58%. The segments that matter most for congestion management are precisely the ones whose state is least reproducible from history — congestion on them is intermittent, not scheduled.

**Table 10. XGBoost per-segment accuracy by functional road class and by the segment's congestion frequency.**

| Functional road class | segments | median accuracy |
|---|---|---|
| 0 (motorway) | 15 | 0.927 |
| 1 | 413 | 0.850 |
| 2 | 4,685 | 0.726 |
| 3 | 4,220 | 0.698 |
| 4 | 7,157 | 0.667 |
| 5 | 2,214 | 0.660 |
| 6 (local) | 5,968 | 0.829 |

| Share of intervals congested | segments | median accuracy |
|---|---|---|
| 0% | 4,399 | 0.888 |
| 0–2% | 10,555 | 0.743 |
| 2–5% | 3,440 | 0.668 |
| 5–10% | 2,193 | 0.637 |
| 10–20% | 2,003 | 0.575 |
| > 20% | 2,082 | 0.575 |

> **[FIGURE 8 — Per-segment predictability.]** From `data/reports/classification/segment_accuracy.csv` (includes latitude/longitude, FRC and congestion share per segment): (a) histogram and cumulative distribution of XGBoost segment accuracy; (b) a map of Bucharest with segments coloured by accuracy (e.g. quintiles), ideally rendered in the visualisation platform. Purpose: locate the unpredictable part of the network.

### 4.6 Feature importance

Gain-based importance (mean over folds, normalised) is concentrated in the segment × hour profile for both boosted models (Table 11). The share of intervals congested at this hour (*seg_hour_cong_rate*) and the lower quartile of the ratio at this hour (*seg_hour_q25*) together account for 36–38% of the gain, followed by the hourly minimum (*seg_hour_min*, 8–9%), the historical probe count at this hour (*hist_seg_hour_probes*, 7–8%) and the hourly mean (*hist_seg_hour*, 6%). Static attributes — posted limit, road class, length — contribute under 2% each: once the segment's own profile is known, what kind of road it is adds almost nothing. The historical volume proxy ranks fourth, confirming that the amount of traffic a segment normally carries at that hour is informative about its state even without the current count.

**Table 11. Top features by normalised gain.**

| Feature | LightGBM | XGBoost |
|---|---|---|
| seg_hour_cong_rate | 0.206 | 0.173 |
| seg_hour_q25 | 0.172 | 0.189 |
| seg_hour_min | 0.095 | 0.076 |
| hist_seg_hour_probes | 0.080 | 0.074 |
| hist_seg_hour | 0.058 | 0.060 |
| hist_seg_weekend | 0.044 | 0.046 |
| segment_congestion_rate | 0.033 | 0.031 |
| segment_slow_rate | 0.033 | 0.036 |
| seg_hour_vs_city_ratio | 0.030 | 0.031 |
| hist_seg_hour_std | – | 0.035 |
| seg_vs_global_dow | 0.028 | – |

> **[FIGURE 9 — Feature importance.]** Horizontal bars for the top 12 features, LightGBM and XGBoost side by side, from `data/reports/classification/lightgbm.md` and `xgboost.md`. Purpose: dominance of the segment × hour profile; irrelevance of static attributes.

### 4.7 Sensitivity to the label definition

The label involves three choices — the free-flow reference, the class thresholds and the light-traffic probe rule — none of which has a canonical value. To check that the conclusions do not depend on them, XGBoost was retrained (with and without class weights) and the historical-mode rule recomputed under three alternative definitions: a stricter light-traffic rule (intervals with fewer than five probes labelled free-flow), wider bands (free-flow ≥ 0.80, congested < 0.50), and the posted speed limit as reference instead of the segment's own free-flow speed. Table 12 reports the pooled results.

**Table 12. Sensitivity of the main comparison to the label definition (pooled leave-one-day-out results).**

| Label definition | Class mix free / slow / congested | Historical mode: accuracy / macro F1 | XGBoost unweighted: accuracy / macro F1 | XGBoost balanced: accuracy / macro F1 | Balanced: congested recall |
|---|---|---|---|---|---|
| Main (segment p85 reference, 0.85 / 0.55, < 3 probes → free-flow) | 70.0 / 24.6 / 5.4 | 0.754 / 0.586 | 0.751 / 0.548 | 0.722 / 0.611 | 0.67 |
| Stricter light-traffic rule (< 5 probes → free-flow) | 73.1 / 22.1 / 4.8 | 0.786 / 0.605 | 0.784 / 0.565 | 0.761 / 0.633 | 0.70 |
| Wider bands (0.80 / 0.50) | 78.3 / 17.5 / 4.2 | 0.810 / 0.574 | 0.809 / 0.518 | 0.760 / 0.597 | 0.64 |
| Posted speed limit as reference | 57.9 / 27.0 / 15.0 | 0.795 / 0.740 | 0.795 / 0.739 | 0.793 / 0.752 | 0.74 |

The absolute figures move with the definition, as they must: labelling more intervals free-flow raises every accuracy, and the class mix changes the majority-rule floor from 58% to 78%. The comparison does not move. Under all four definitions the unweighted gradient-boosted model matches the historical-mode rule to within 0.3 accuracy points and does not exceed its macro F1; the class-weighted model adds 0.01–0.03 macro F1 over the rule and raises congestion recall to 64–74%. The finding that the segment's usual state carries essentially all of the recoverable signal is therefore not an artefact of where the class boundaries were drawn or of how sparse intervals were handled.

The posted-limit row also illustrates why that reference was rejected. Under it, everything appears more predictable — the historical-mode rule reaches a macro F1 of 0.74, against 0.57–0.61 for the segment-relative definitions — because a quarter of segments never approach their posted limit and are labelled slow or congested at every hour of every day. Such "congestion" is perfectly recurrent, and a model that reproduces it reports a spurious recurrent share. A label that encodes road geometry makes traffic look more habitual than it is.

---

## 5. Discussion

**Most of the predictable structure is the profile itself.** The central result is that a rule with no parameters — each segment's usual state at that time of day — reproduces 75% of segment-intervals, and that three classifiers of very different capacity add little discriminative power on top of it: macro F1 moves from 0.59 to 0.60–0.61 with class weighting, and without it the best model reproduces the rule's accuracy to within half a point with a lower macro F1. The tree models, given interactions the linear model cannot form, do not separate from it. This pattern (strong historical baseline, small model gains) is a recurring one in traffic prediction and is worth stating plainly, because the accuracy of a machine-learning model is easily read as the accuracy of machine learning. In this study, the models' value lies elsewhere: through class weighting they convert the same information into a detector that catches two thirds to three quarters of congested intervals instead of one third, at a precision of 32–40%. Which operating point is preferable depends on the use: for an information service, false congestion alarms erode trust; for planning interventions, missed congestion is the costlier error.

**The non-recurrent quarter has a location.** Whatever the model, about one segment-interval in four is not where its history places it, and during working hours it is one in three. This residual is not spread evenly. It is largest on mid-hierarchy collector roads and on the segments that are congested most often; it is small on motorways, on local streets and at night. On the segments with the highest congestion frequency, accuracy is 57% — history explains their state barely better than the majority class. Congestion on these segments is intermittent: it depends on demand fluctuations, incidents and signal behaviour that a weekly profile cannot capture. This is the part of the network where real-time detection, rather than historical prediction, would pay off — and the probe-count experiment suggests that even one real-time signal moves macro F1 by only 0.02, so it is the *state* of neighbouring segments and the *recent* trajectory, not volume, that would have to be added.

**Errors are ordinal.** Free-flow and congested are almost never confused with each other (4–12% of either class); the errors are between adjacent states, and the slow band absorbs them from both sides. This partly reflects the label: "slow" is a 30-percentage-point band of a continuous ratio, and intervals near its edges are inherently ambiguous. It also suggests that reporting the models as regressors of the speed ratio, with the class thresholds applied afterwards, would give users a sense of how far into the band a prediction lies.

**Two data-quality pitfalls dominate everything else.** Two choices that look like preprocessing details changed the results more than any modelling choice. Treating zero-probe intervals as observations (the API reports them with zero speed) and using the posted limit as free-flow reference produced a label in which 38% of segments were "congested" at 04:00 and a fifth of the week's congestion was missing data; models trained on that label reached 81% accuracy largely by learning *no data ⇒ congested* and *short intersection segment ⇒ congested*. Likewise, computing historical aggregates in-sample let every training row see its own label through its segment-hour minimum, producing a model that was excellent on training data and poor on held-out days. Probe-based traffic studies should report the treatment of low-sample intervals, the free-flow reference and the encoding protocol as first-class methodological choices.

**Weekends are not measured, only glimpsed.** With one Saturday and one Sunday, the study cannot say how recurrent weekend traffic is; it can only say that Saturday and Sunday differ enough (a midday hump on Saturday) that one does not predict the other. The weekday results — five days with four-day training profiles — are the reliable core of the study.

## 6. Limitations

*One week, in August.* The data cover seven consecutive days, 25–31 August 2024. August is the principal holiday month in Romania: schools are closed, many residents are away, and traffic volumes in Bucharest are appreciably lower than in the autumn and winter months. Two consequences follow. First, the absolute levels reported here — 5.4% of segment-intervals congested, 12% at the evening peak — should be read as a lower bound for a typical working week; the class balance, and with it the accuracy of the majority rule and the difficulty of the congested class, would be different in October. Second, the recurrent share itself may differ: a holiday week has fewer of the fixed schedules (school runs, full office occupancy) that generate recurrence, but also fewer of the volume-driven breakdowns that generate non-recurrent congestion; the direction of the net effect is not known from these data. The results are a measurement of the recurrent share in a light-traffic week, not a model of Bucharest traffic in general.

*One example of each weekend day.* As discussed, weekend predictability is not identifiable; the week also contains no public holiday, no major event and no weather extreme that would test the models under disruption.

*No seasonal, holiday or trend information.* Features are computed from the same week; there is no notion of a segment's behaviour in a different month, or of long-term change.

*Probe data represent probe vehicles.* The sample is the fleet of vehicles reporting to the data provider, whose composition (navigation-app users, fleet vehicles) is not that of all traffic. Penetration is lowest exactly where the label is most uncertain — minor roads, night — and the light-traffic rule labels those intervals free-flow by assumption, not by measurement; 24% of rows carry this assumed label. A different threshold changes the class balance and the absolute accuracies but, as Section 4.7 shows, not the relative findings; exclusion of sparse intervals rather than assumed free-flow was not tested.

*Label construction from the same week.* The segment free-flow reference (85th percentile) is estimated over the whole week, including the held-out day. It is a segment property rather than a feature of the predicted interval, but strictly it carries one week of information about the test day's speed distribution into the label definition.

*Spatial independence is not exploited.* Segments are treated as independent; the state of neighbouring segments, upstream and downstream, is a known predictor of congestion propagation and is not used. This is the most promising direction for capturing the non-recurrent share.

*Hyperparameters were not tuned.* Both boosted models were still improving slowly at the round cap; a tuned configuration would likely gain one or two points, without altering the comparison with the historical-mode rule.

---

## 7. Conclusions

Using a complete week of probe-vehicle data for the Bucharest road network, this study measured how much of a road segment's traffic state can be predicted from its own history. The answer, for the observed week, is about three quarters: a deterministic replay of each segment's usual state at that time of day reproduces 75% of segment-intervals, and three classifiers — logistic regression, LightGBM and XGBoost — trained on thirty historical, static and time-of-day features do not improve on that figure in accuracy (XGBoost without class weights: 75.1%) and improve macro F1 by only 0.01–0.03 when class-weighted. What the classifiers provide is a different operating point: with balanced class weights they detect 67–75% of congested intervals instead of 36%, at a precision of about a third. Adding the number of probe vehicles in the predicted interval, a real-time signal, raises macro F1 by a further 0.02.

The unpredictable quarter has a clear geography and timing. It is concentrated in the working day, where a third of segment-intervals deviate from their profile at every hour from 09:00 to 19:00; on secondary and collector roads rather than on motorways or local streets; and on the segments that are congested most often, where history explains the state barely better than the majority class. These are the places and times where real-time detection, and information from neighbouring segments, would be needed.

The study also shows that the answer depends on decisions made before any model is trained. Intervals without probe vehicles must not be read as congested; the free-flow reference must be the segment's own speed distribution rather than the posted limit; and historical aggregates used as features must be computed out-of-fold. Each of these, if ignored, produces a more accurate-looking model of a quantity that is not traffic.

Finally, the results are bounded by the data window: a single week in August, the lightest traffic month of the year in Bucharest, with one Saturday and one Sunday. Extending the collection to several months across seasons — and in particular to a full autumn working period — is the necessary next step to establish whether the recurrent share measured here is representative, to make weekend recurrence identifiable, and to test the models under the incidents, weather and events that make up the non-recurrent remainder.

---

## References

> To be populated from the Scite.ai searches listed in Section 2. No references are asserted in this draft.

---

## Appendix A. Reproducibility

- Feature engineering and label definition: `scripts/features.py`
- Logistic regression: `scripts/logistic_regression_baseline.py`
- LightGBM / XGBoost (`--with-sample-size`, `--no-class-weights` variants): `scripts/gbm_classifiers.py`
- Reference rules and per-hour / per-segment analysis: `scripts/analyze_predictions.py`
- Reports: `data/reports/classification/{logistic_regression,lightgbm,xgboost,xgboost_with_sample_size,xgboost_unweighted,analysis}.md`; sensitivity variants carry the suffixes `_minprobes5`, `_thr80_50`, `_limitref` (label definition set through the environment variables `TRAFFIC_MIN_PROBES`, `TRAFFIC_THRESHOLDS`, `TRAFFIC_REFERENCE`)
- Figure data: `paper/figure_data/*.csv`, `data/reports/classification/{accuracy_by_hour,segment_accuracy}.csv`
- Hardware: NVIDIA RTX PRO 6000 (XGBoost, logistic regression), 32 CPU threads (LightGBM). Feature construction ≈ 75 s per fold; training per fold ≈ 20 s (logistic regression), ≈ 3 min (XGBoost), ≈ 8 min (LightGBM).
