# Machine Learning | Predict credit card fraud
📌 Build a Machine Learning model to predict credit card fraud <br>
🌟 Link Google Colab: [link](https://colab.research.google.com/drive/1iOyaPzhOaHbcXXmczI4dlO-DkG-PRHCH?usp=sharing)
## :one: Project Overview:
**:round_pushpin: The main purpose of this project is:**
Automatically detect fraudulent transactions based on patterns in customer behavior, transaction amount, location, and other features 
<br> ➡️ **helps reduce financial loss.**

## :two: Dataset Description:
- **Dataset:** [link](https://drive.google.com/file/d/1cRMB2xBtaGQFCsCt7G9FyM-6WKJLjamk/view?usp=drive_link)
### 1. Categorical Columns:
   - **merchant:** Merchant who was getting paid.
   - **category:** In what area does that merchant deal.
   - **first:** first name of the card holder.
   - **last:** last name of the card holder.
   - **gender:** Gender of the cardholder.Just male and female!
   - **street:** Street of card holder residence
   - **city:** city of card holder residence
   - **state:** state of card holder residence
   - **zip:ZIP** code of card holder residence
   - **job:** trade of the card holder
### 2. Numerical Columns:
   - **amt:** Amount of money in American Dollars.
   - **lat:** latitude of card holder
   - **long:** longitude of card holder
   - **merch_lat:** latitude of the merchant
   - **merch_long:** longitude of the merchant
   - **is_fraud:** Whether the transaction is fraud(1) or not(0)

## :three: Tools used:
- **Package used:** sk-learn (scikit-learn)
- **Feature Engineering:** add_features()
- **Numeric processing function:** SimpleImputer(), StandardScaler()
- **Categorical processing:** OneHotEncoder(), TargetEncoder()
- **Training model:** LogisticRegression(), RandomForestClassifier(), XGBClassifier()
- **Model evaluation:** classification_report(), roc_auc_score()
- **Pipeline():**
      <br>- Simplifies and organizes code
      <br>- Prevents data leakage
      <br>- Ensures consistent preprocessing at training and inference time
      <br>- Modular & reusable
      <br>- Safe integration with deployment
  
## :four: Work flow:
**1. Define the problem**
<br>**2. EDA (Exploratory Data Analysis)**
<br>**3. Split train/test set**
<br>**4. Data Preprocessing**
<br>**5. Select the ML model**
<br>**6. Evaluate the model**
<br>**7. Hyperparameter tuning**
<br>**8. Evaluate the final model on the test set**
<br>**9. Save the model**
## :five: Python Code:
### 1. Define the problem
- **Business Objectives:** Build a basic Machine Learning model to predict credit card fraud
- **Type of ML problem:** Supervised (Classification)
- **Indentify the features and target:**
  - Feature:
    - trans_date_trans_time: extract days of week, time from datetime
    - merchant: name of merchant who was getting paid.
    - category: the area in which the merchant deals
    - amt: Amount of money in American Dollars
    - gender: gender of the cardholder
    - lat, long: latitude and longtitude of card holder
    - city_pop: population of the city
    - job: trade of the card holder
    - dob: extract age from date of birth
  - Target: is_fraud (whether the transaction is fraud (1) or not (0))

✏️**Feature engineered columns:**
```
from sklearn.preprocessing import FunctionTransformer
data = data.copy()
def add_features(data):
    data['transaction_hour'] = pd.to_datetime(data['trans_date_trans_time']).dt.hour
    data['transaction_weekday'] = pd.to_datetime(data['trans_date_trans_time']).dt.weekday
    data['age'] = (pd.to_datetime(data['trans_date_trans_time']) - pd.to_datetime(data['dob'])).dt.days // 365
    return data

feature_engineering = FunctionTransformer(add_features)
```
### **2. EDA (Exploratory Data Analysis)**
✏️ **Check dataset information:** data.info() and data.describe()
<br>➡️ **Output:** 0 missing values
<br><br>✏️ **Check categorical columns:**
```
import pandas as pd

# Filter columns with dtypes = object or category
category_columns = data.select_dtypes(include=['object', 'category']).columns

# Print unique values of each column
for col in category_columns:
    unique_count = data[col].nunique(dropna=False)  # tính cả NaN nếu có
    print(f" Column '{col}': {unique_count} unique values")
```
➡️ **Output:**
 - Column 'merchant': 693 unique values -> use Target Encoding
 - Column 'category': 14 unique values -> use One-Hot Encoding
 - Column 'gender': 2 unique values -> use One-Hot Encoding
 - Column 'job': 494 unique values -> use Target Encoding
### **3. Split train/test set**
- The function **ColumnTransformer() in step 4** will remove original cateogorical columns, such as 'merchant', 'gender', 'job'. All the columns that are not listed are also removed
```
from sklearn.model_selection import train_test_split

X = data.drop(columns='is_fraud')
y = data['is_fraud']
data.head()

# Split the data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

```
### **4. Data Preprocessing**
- Encode categorical columns
- Standardized numerical columns
```
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from category_encoders import TargetEncoder
from sklearn.impute import SimpleImputer

num_cols = [
    "amt",
    "lat",
    "long",
    "city_pop",
    # Feature engineered columns:
    "transaction_hour",
    "transaction_weekday",
    "age",
]
target_enc_cols = ['merchant', 'job']
onehot_cols = ['category', 'gender']

# Numeric: impute + scale
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# One-hot
onehot_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Target Encoding
target_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('target', TargetEncoder(handle_unknown='ignore'))
])

# Combine
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('target', target_transformer, target_enc_cols),
    ('onehot', onehot_transformer, onehot_cols)
])
```


### **5. Select the ML model**
- Test on 3 ML model (Logistic Regression, Random Forest, XGBoost) to choose the best-performing model
```
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_auc_score

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

pipelines = {}

for name, model in models.items():
    pipelines[name] = Pipeline([
        ('feature_engineering', feature_engineering),
        ('preprocessor', preprocessor),
        ('model', model)
    ])
```
### **6. Evaluate the model**
- **Main Evaluation Score:** ROC AUC <br>
#### ❓Reasons why ROC AUC is chosen as the main evaluation score:
- Handles class imbalance well: In fraud detection, positive class (fraud) is very rare. Therefore, accuracy is misleading (a model that predicts “non-fraud” every time could still be >99% accurate)
- ROC AUC evaluates model’s ability to rank predictions ➡️ High ROC AUC means your model effectively separates fraud from non-fraud, regardless of threshold.

```
from sklearn.metrics import classification_report, roc_auc_score

for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print(f"Model: {name}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}\n")
```
➡️ **Output:**
- From the scores, we can conclude that **XGBoost** performs best among 3 models (ROC AUC = 0.9989, Precision, recall, and f1-score for class 1 are 0.97, 0.94, and 0.95 respectively)
- **Tuning XGBoost model**


### **7. Hyperparameter tuning**

```
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

# Tập tham số
param_grid_xgb = {
    'model__n_estimators': [100, 300, 500],
    'model__max_depth': [3, 5, 7, 10],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__subsample': [0.6, 0.8, 1.0],
    'model__colsample_bytree': [0.6, 0.8, 1.0]
}

xgb = pipelines['XGBoost']

xgb_search = RandomizedSearchCV(
    xgb,
    param_distributions=param_grid_xgb,
    n_iter=20,
    scoring='roc_auc',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

xgb_search.fit(X_train, y_train)

print("Best XGB params:", xgb_search.best_params_)
print("Best XGB AUC:", xgb_search.best_score_)
```
➡️ **Output:**
- Best XGB params: {'model__subsample': 0.8, 'model__n_estimators': 500, 'model__max_depth': 10, 'model__learning_rate': 0.05, 'model__colsample_bytree': 0.6}
- Best XGB AUC: 0.9977965165126989

### **8. Evaluate the final model on the test set**

```
# Use the best model from RandomizedSearch
best_xgb = xgb_search.best_estimator_
y_pred = best_xgb.predict(X_test)
y_proba = best_xgb.predict_proba(X_test)[:, 1] # Probability of class = 1

print(classification_report(y_test, y_pred))
print("Test ROC AUC:", roc_auc_score(y_test, y_proba))
```
➡️ **Output:**

```
              precision    recall  f1-score   support

           0       0.99      1.00      1.00     18049
           1       0.97      0.93      0.95      1501

    accuracy                           0.99     19550
   macro avg       0.98      0.96      0.97     19550
weighted avg       0.99      0.99      0.99     19550

Test ROC AUC: 0.9986180930444397
```

### **9. Save the model**
```
import joblib
joblib.dump(best_xgb, "fraud_detection_model.pkl")
```
