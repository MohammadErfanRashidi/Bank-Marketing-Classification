# Bank Term Deposit Subscription Prediction

A machine learning project to predict whether a client will subscribe to a term deposit based on data from a Portuguese bank's direct marketing campaign.

## Dataset

**Source**: Direct marketing campaign of a Portuguese banking institution.  
**Size**: 45,211 entries, 17 original attributes.  
**Target variable**: `y` (binary: “yes” if subscribed, “no” otherwise).

### Original attributes

| Attribute   | Type        | Description |
|-------------|-------------|-------------|
| age         | numeric     | Client age |
| job         | categorical | Type of job (admin., blue‑collar, entrepreneur, etc.) |
| marital     | categorical | Marital status (married, single, divorced) |
| education   | categorical | Education level (primary, secondary, tertiary, unknown) |
| default     | categorical | Has credit in default? (yes, no) |
| balance     | numeric     | Average yearly balance (euros) |
| housing     | categorical | Has housing loan? (yes, no) |
| loan        | categorical | Has personal loan? (yes, no) |
| contact     | categorical | Communication type (unknown, telephone, cellular) |
| day         | numeric     | Last contact day of month (1–31) |
| month       | categorical | Last contact month (jan, feb, …, dec) |
| duration    | numeric     | Last contact duration (seconds) |
| campaign    | numeric     | Number of contacts during this campaign |
| pdays       | numeric     | Days since last contact from a previous campaign (-1 if never contacted) |
| previous    | numeric     | Number of contacts before this campaign |
| poutcome    | categorical | Outcome of previous campaign (unknown, other, failure, success) |
| y           | binary      | Subscribed term deposit? (yes, no) |

## Cleaning

- No completely empty columns were present.
- “unknown” values were converted to proper nulls.
- 3 rows with `duration` equal to 0 were removed.
- No duplicate rows were found.
- Class imbalance: only 5,289 “yes” subscriptions (≈11.6%).
- A flag `prev_contacted` (0 = never contacted, 1 = contacted before) was derived.
- Missing values in `job` and `education` were dropped; missing values in `contact` and `poutcome` were replaced with the string “missing”.
- Categorical columns were converted to `category` dtype.

## Exploratory Data Analysis (EDA)

### Categorical features (subscription rate “yes” highlights)

- **job**: retired and student show highest subscription rates.
- **marital**: singles subscribe most.
- **education**: tertiary education yields the highest yes rate.
- **default**: clients without default subscribe more.
- **housing**: those without a housing loan subscribe more.
- **loan**: those without a personal loan subscribe more.
- **contact**: missing contact type has the lowest yes rate.
- **month**: December and March have the highest subscription rates.
- **poutcome**: previous success strongly predicts subscription.

### Numerical features (comparing yes vs. no)

- **age**: higher maximum and standard deviation for yes; median similar.
- **balance**: larger IQR, higher median, higher maximum for yes.
- **day**: distribution nearly identical across target classes.
- **duration**: much larger IQR, higher median, higher maximum for yes.
- **campaign**: distributions similar.
- **pdays**: larger values and IQR for yes (clients previously contacted longer ago).
- **previous**: larger values and IQR for yes.
- **prev_contacted**: larger values, higher median, higher max for yes.

### Correlation heatmap

- `duration` had the strongest linear correlation with the target (+0.4).
- Increase in duration tended to increase subscription likelihood regardless of age.
- `prev_contacted` correlated very highly with `pdays` (+0.87).

## Feature Engineering & Preprocessing

All categorical features were transformed into binary indicators based on the EDA insights. The original categorical columns were dropped after encoding.

1. **Job**: flag for `student` or `retired` → `job_student_or_retired_feat`.  
2. **Marital**: flag for `single` → `marital_single_feat` (later dropped, as it did not pass final selection).  
3. **Education**: flag for `tertiary` → `education_tertiary_feat` (dropped later).  
4. **Contact**: flag for `cellular` or `telephone` → `contact_cellular_telephone_feat`.  
5. **Month**: flag for months `dec`, `oct`, `sep`, `mar` → `month_dec_oct_sep_mar_feat`.  
6. **Poutcome**: flag for `success` → `poutcome_success_feat`.  
7. **Housing, Loan, Default**: binary encoding (yes=1, no=0) → `housing_feat`, `loan_feat`, `default_feat` (only `housing_feat` kept in final selection).  

Numerical features were binned and then turned into binary indicators.

- **Duration**: binned into quartiles; flag for Q3/Q4 → `duration_bin_Q3_Q4_feat` (dropped later in favour of raw duration).  
- **Age**: binned and inspected but not retained.  
- **Previous**: flag for any previous contact (>0) → `previous_any_feat` (not retained).  
- **Day**, **Campaign**: binned but distributions showed negligible effect, so they were dropped.  
- **Pdays**: replaced -1 with 999 to create `pdays_replaced_feat`.

Log-transformed versions of `balance` and `duration` were created but not used in the final model.

### Feature selection

Correlation with target (`y`) was computed for all engineered features. Features with absolute correlation > 0.1 were considered candidates. Among those, pairs with inter‑feature correlation > 0.7 were examined; the feature with lower target correlation in each pair was removed to reduce multicollinearity.

**Final selected features**:
- `duration`
- `poutcome_success_feat`
- `month_dec_oct_sep_mar_feat`
- `pdays_replaced_feat`
- `contact_cellular_telephone_feat`
- `housing_feat`
- `job_student_or_retired_feat`

The final dataset was saved as `bank_final.csv`.

### Scaling

Continuous features (`duration`, `pdays_replaced_feat`) were scaled using `RobustScaler`. The scaler object was saved as `scaler.pkl`.

## Modeling

Data split: 85% training, 15% test (stratified). A dummy classifier served as baseline.

| Model               | Accuracy | Class 1 Recall | ROC AUC  |
|---------------------|----------|----------------|----------|
| Dummy (baseline)    | 0.50     | 0.50           | 0.50     |
| Logistic Regression | 0.83     | 0.82           | 0.896    |
| Decision Tree       | 0.72     | 0.91           | 0.851    |
| Random Forest       | 0.81     | 0.86           | 0.906    |
| XGBoost             | 0.79     | 0.88           | 0.895    |

Random Forest was selected for deployment due to its highest ROC AUC and strong recall on the minority class.

All trained models were saved for later use.

## Deployment (Streamlit App)

A Streamlit application was built to serve the model with two modes:

### Single Prediction
- User provides: `poutcome`, `month`, `pdays`, `contact`, `housing`, `job`, `duration`.
- Feature engineering steps are applied:
  - `poutcome_success_feat` = 1 if poutcome is “success” else 0.
  - `month_dec_oct_sep_mar_feat` = 1 if month is in [dec, oct, sep, mar] else 0.
  - `pdays_replaced_feat` = 999 if pdays == -1 else the actual value.
  - `contact_cellular_telephone_feat` = 1 if contact is cellular or telephone else 0.
  - `housing_feat` = 1 if housing is “yes” else 0.
  - `job_student_or_retired_feat` = 1 if job is student or retired else 0.
- The seven features are assembled into an array, then `duration` and `pdays_replaced_feat` are scaled using the loaded `scaler.pkl`.
- The Random Forest model predicts the subscription outcome, which is displayed to the user.

### Batch Prediction
- User uploads a CSV file containing the required columns: `poutcome`, `month`, `pdays`, `contact`, `housing`, `job`, `duration`.
- The same feature engineering logic is applied to the entire dataset.
- Scaled features are fed to the model, and a new column `y_predicted` is added.
- The first 10 rows of the resulting dataframe are shown, and a download button provides the full dataset with predictions.

## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- lightgbm
- joblib
- streamlit