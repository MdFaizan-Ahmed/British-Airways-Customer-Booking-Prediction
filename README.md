# British Airways Customer Booking Prediction

Predicting flight booking completion using customer booking data.

## Dataset

The dataset `customer_booking.csv` contains 50,000 customer booking records with features including:
- `purchase_lead` – days between booking and travel
- `length_of_stay` – days spent at destination
- `flight_hour` / `flight_day` – departure time
- `wants_extra_baggage` / `wants_preferred_seat` / `wants_in_flight_meals` – add-on preferences
- `route` and `booking_origin` – route and customer country
- `booking_complete` – target variable (1 = booking completed)

## Results

- **XGBoost ROC-AUC:** 0.786
- **Top predictors:** Customer origin country, sales channel (mobile vs internet), add-on purchases

## Dependencies
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
