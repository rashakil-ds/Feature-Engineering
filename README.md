# Feature Engineering

**Feature Engineering** is the process of transforming raw data into meaningful features that help machine learning models learn better patterns and make more accurate predictions.

In simple terms, it's about **creating the right inputs** for your model — the smarter the features, the better the model’s performance.

---

## Key Steps in Feature Engineering

### 1. Feature Creation
Generate new features from existing ones to add useful information.

**Examples:**
- From `Date`, create `Day`, `Month`, or `Is_Weekend`
- From `Price` and `Quantity`, create `Total_Sales = Price × Quantity`
- From `Address`, extract `City` or `Postal_Code`

---

### 2. Feature Transformation
Modify existing features to improve learning and model performance.

**Common techniques:**
- **Scaling:** Normalize or standardize numerical features  
  *(e.g., Min-Max scaling or StandardScaler)*
- **Encoding:** Convert categorical data into numeric form  
  *(e.g., One-Hot Encoding or Label Encoding)*
- **Log Transformation:** Handle skewed data distributions
- **Binning:** Group continuous values into discrete intervals

---

### 3. Feature Selection
Choose only the most important features that influence predictions.

**Methods include:**
- Correlation analysis  
- Mutual information  
- Feature importance from tree-based models  
- Recursive Feature Elimination (RFE)

---

### 4. Handling Missing Data
Fill or remove missing values to ensure clean input for training.

**Strategies:**
- Replace with mean/median/mode  
- Forward/Backward fill  
- Drop rows/columns with too many missing values

---

## Example

**Raw dataset:**

| Date | Age | Salary | City |
|------|-----|---------|------|
| 2024-05-10 | 28 | 50000 | Berlin |
| 2024-05-11 | 35 | 62000 | Munich |

**After feature engineering:**

| Day | Month | Age | Salary | City_Berlin | City_Munich |
|-----|-------|-----|---------|--------------|--------------|
| 10 | 5 | 28 | 50000 | 1 | 0 |
| 11 | 5 | 35 | 62000 | 0 | 1 |

---

## Why It Matters
Feature Engineering:
- Improves **accuracy** of ML models  
- Reduces **overfitting**  
- Enhances **interpretability**  
- Speeds up **training and inference**

---

## Example in Python

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Sample Data
data = {
    "Date": ["2024-05-10", "2024-05-11"],
    "Age": [28, 35],
    "Salary": [50000, 62000],
    "City": ["Berlin", "Munich"]
}
df = pd.DataFrame(data)

# --- Feature Creation ---
df["Date"] = pd.to_datetime(df["Date"])
df["Day"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month

# --- Feature Transformation ---
encoder = OneHotEncoder(sparse_output=False)
city_encoded = encoder.fit_transform(df[["City"]])
encoded_df = pd.DataFrame(city_encoded, columns=encoder.get_feature_names_out(["City"]))
df = pd.concat([df, encoded_df], axis=1)

# --- Feature Scaling ---
scaler = StandardScaler()
df[["Age", "Salary"]] = scaler.fit_transform(df[["Age", "Salary"]])

# --- Final Result ---
print(df)
# Feature-Engineering
