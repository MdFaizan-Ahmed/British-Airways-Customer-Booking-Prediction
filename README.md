# British Airways Customer Booking Prediction

Predicting whether a customer completes a flight booking, to support conversion-targeting campaigns. Part of the Forage British Airways virtual experience.

## Problem

Only ~15% of bookings are completed. A model that identifies high-probability customers lets marketing target the right segment, rather than discounting indiscriminately. The question isn't "can we predict completion" — it's "does this feature set carry enough signal to be useful, and if not, what's missing?"

## Dataset

50,000 records, 14 features: `purchase_lead`, `length_of_stay`, `flight_hour`, `flight_day`, `route`, `booking_origin`, add-on preferences (`wants_extra_baggage`, `wants_preferred_seat`, `wants_in_flight_meals`), `flight_duration`. Target: `booking_complete` (binary, 15% positive). No missing values.

## Method

- **Split first, encode second.** 80/20 stratified train/test split before any feature engineering. Frequency encoding for `route` and `booking_origin` was fit on **train only**, with unseen categories falling back to frequency 1 — the same category of mistake as look-ahead bias, just in feature engineering. This is the thing most tutorial notebooks get wrong.
- **Class imbalance handled consistently.** Random Forest used `class_weight='balanced'`; XGBoost used `scale_pos_weight` (ratio of negatives to positives in the training set). Both models were addressing the same problem, not one balanced and one not.
- **Stratified 5-fold CV** on the training set, for the model whose test score is reported (XGBoost), not a different model.
- **Evaluated with ROC-AUC, precision, recall, F1** — not accuracy. At 85% negative class, accuracy rewards a model that always predicts "no."

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Random Forest (`class_weight='balanced'`) | 0.853 | 0.54 | 0.09 | 0.16 | 0.773 |
| XGBoost (`scale_pos_weight`) | 0.737 | 0.32 | 0.66 | 0.43 | 0.777 |

XGBoost 5-fold stratified CV ROC-AUC: **0.765 ± 0.016**.

Confusion matrix (XGBoost, test set): 991 true positives, 505 false negatives, 2,129 false positives, 6,375 true negatives.

## Key Findings

**ROC-AUC plateaus around 0.77.** This is threshold-independent, so it reflects the real ceiling on predictive signal in this feature set — not the precision/recall tradeoff. The features carry moderate signal, and no amount of model tuning will push past it.

**The precision/recall tradeoff is a business decision, not a modelling one.** XGBoost catches ~2 in 3 completed bookings (recall 0.66) but is right less than a third of the time when it predicts "yes" (precision 0.32). Whether that trade is worth it depends on the relative cost of missing a completed booking vs. chasing a false lead — which the dataset doesn't specify.

**Top predictors: customer origin country, sales channel (mobile vs. internet), add-on purchases.** These are proxies for intent, not direct measures of it. A customer who wants extra baggage and books on mobile is more committed; a customer from a high-frequency origin country is more likely to complete. The model is learning intent from behaviour, not from the thing itself.

## Limitations & Next Steps

- **No loyalty, price sensitivity, or competitor pricing data.** The ceiling is feature-driven, not model-driven. Adding these is the only way to move ROC-AUC meaningfully.
- **Threshold should be tuned to the business cost** of false positives vs. false negatives. The default 0.5 is arbitrary.
- **A calibrated model** would make the threshold decision interpretable, rather than a probability that isn't a probability.
- **The Random Forest's low recall (0.09) despite `class_weight='balanced'`** is worth investigating — it suggests the balanced weighting isn't enough for RF on this feature set, or that the RF is overfitting the majority class despite the weights. XGBoost's `scale_pos_weight` handled it better.

## Tech Stack

Python: pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn.
