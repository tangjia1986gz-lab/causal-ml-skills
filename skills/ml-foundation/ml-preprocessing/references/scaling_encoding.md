# Scaling and Encoding for Causal Inference

## Overview

Proper scaling and encoding of variables is essential for many causal inference methods, particularly those involving distance calculations (matching, weighting) or regularization (LASSO, double machine learning). This document covers standardization, normalization, and categorical encoding with attention to causal inference requirements.

## Standardization

### Z-Score Standardization

Transforms variables to mean=0 and standard deviation=1:

```python
from preprocessing import standardize

# Standardize control variables
df_std, scaler = standardize(df, columns=['age', 'income', 'education'])

# Apply same transformation to new data
new_data_std = scaler.transform(new_data[['age', 'income', 'education']])
```

**Formula**: z = (x - mean) / std

**When to use**:
- Regularized regression (LASSO, Ridge, Elastic Net)
- Distance-based methods (propensity score matching, nearest neighbor)
- Methods sensitive to variable scales
- Comparing coefficient magnitudes

### Min-Max Normalization

Scales variables to [0, 1] range:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[columns] = scaler.fit_transform(df[columns])
```

**Formula**: x_scaled = (x - min) / (max - min)

**When to use**:
- Neural network inputs
- When you need bounded values
- Comparing variables with different units

**Caution**: Sensitive to outliers; consider robust scaling instead.

### Robust Scaling

Uses median and IQR, robust to outliers:

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
df[columns] = scaler.fit_transform(df[columns])
```

**Formula**: x_scaled = (x - median) / IQR

**When to use**:
- Data with outliers you want to retain
- When mean/std are unreliable

---

## Causal Considerations for Scaling

### What to Scale

| Variable Type | Scale? | Reasoning |
|--------------|--------|-----------|
| Continuous covariates | Yes (usually) | Improves numerical stability, enables regularization |
| Binary treatment | No | Interpretation: effect of D=1 vs D=0 |
| Continuous treatment | Consider | May help with regularization |
| Outcome | Usually No | Maintains interpretable units |
| Categorical (encoded) | Yes (one-hot) | Ensures equal contribution to distance |

### Scale Using Training Data Statistics

**Critical**: Fit scaler on training data only, then transform both train and test.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# CORRECT: Split first, then scale
train, test = train_test_split(df, test_size=0.2, random_state=42)

scaler = StandardScaler()
train[controls] = scaler.fit_transform(train[controls])  # Fit on train
test[controls] = scaler.transform(test[controls])  # Transform test with train stats

# WRONG: Scale before splitting (data leakage)
# df[controls] = scaler.fit_transform(df[controls])  # BAD!
# train, test = train_test_split(df)  # Test info leaked into scaling
```

### Same Scaling for Treatment and Control Groups

Apply identical scaling to both groups to preserve comparability:

```python
# CORRECT: Single scaler for all observations
scaler = StandardScaler()
df[controls] = scaler.fit_transform(df[controls])

# WRONG: Different scaling by treatment status
# scaler_t = StandardScaler().fit_transform(df[df['D']==1][controls])  # BAD!
# scaler_c = StandardScaler().fit_transform(df[df['D']==0][controls])  # BAD!
```

---

## Categorical Encoding

### One-Hot Encoding (Dummy Variables)

Creates binary indicators for each category:

```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Using sklearn
encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' avoids collinearity
encoded = encoder.fit_transform(df[['region', 'education_level']])
encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(['region', 'education_level'])
)

# Using pandas
df_encoded = pd.get_dummies(df, columns=['region', 'education_level'], drop_first=True)
```

**Causal considerations**:
- Drop one category (reference) for regression to avoid perfect collinearity
- Reference category affects coefficient interpretation
- All categories need representation in both treatment groups for overlap

### Label Encoding

Assigns integer values to categories:

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['region_encoded'] = le.fit_transform(df['region'])
```

**When to use**:
- Tree-based models (can handle categorical splits)
- Ordinal variables with natural ordering

**Caution**: Implies ordering that may not exist for nominal variables.

### Ordinal Encoding

For variables with natural ordering:

```python
from sklearn.preprocessing import OrdinalEncoder

# Define order
education_order = ['Less than HS', 'High School', 'Some College', 'Bachelor', 'Graduate']
encoder = OrdinalEncoder(categories=[education_order])
df['education_encoded'] = encoder.fit_transform(df[['education_level']])
```

### Target Encoding

Encodes categories by their mean outcome value:

```python
def target_encode(df, cat_col, target_col, smoothing=10):
    """
    Encode categorical variable by target mean with smoothing.

    WARNING: Can leak outcome information in causal settings.
    """
    global_mean = df[target_col].mean()
    cat_means = df.groupby(cat_col)[target_col].agg(['mean', 'count'])

    # Smoothed estimate
    smooth_mean = (cat_means['mean'] * cat_means['count'] + global_mean * smoothing) / \
                  (cat_means['count'] + smoothing)

    return df[cat_col].map(smooth_mean)
```

**Causal warning**: Target encoding uses outcome information, which can:
- Bias treatment effect estimates
- Violate the requirement to use only pre-treatment information

**Safe alternative**: Encode using treatment propensity or baseline covariates only.

### Frequency Encoding

Encodes by category frequency:

```python
def frequency_encode(df, cat_col):
    """Encode by category frequency."""
    freq_map = df[cat_col].value_counts(normalize=True).to_dict()
    return df[cat_col].map(freq_map)

df['region_freq'] = frequency_encode(df, 'region')
```

**Advantage**: Does not use outcome information; safe for causal inference.

---

## Encoding for Specific Causal Methods

### For Propensity Score Models

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# Define variable types
numeric_vars = ['age', 'income', 'years_experience']
categorical_vars = ['region', 'education_level', 'occupation']

# Create preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_vars),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_vars)
])

# Fit and transform
X_processed = preprocessor.fit_transform(df[numeric_vars + categorical_vars])

# Propensity score model
ps_model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)
ps_model.fit(X_processed, df['treatment'])
df['propensity_score'] = ps_model.predict_proba(X_processed)[:, 1]
```

### For Matching

Standardization is critical for distance-based matching:

```python
from sklearn.neighbors import NearestNeighbors

# Standardize all matching variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[matching_vars])

# One-hot encode categoricals (already included in X)
# Then standardize the full matrix

# Nearest neighbor matching
treated_idx = df[df['treatment'] == 1].index
control_idx = df[df['treatment'] == 0].index

nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(X_scaled[control_idx])

# Find matches for treated units
distances, indices = nn.kneighbors(X_scaled[treated_idx])
matched_control_idx = control_idx[indices.flatten()]
```

### For LASSO/Regularized Regression

Standardization ensures fair penalization across variables:

```python
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# Standardize ALL predictors (including dummies)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# LASSO with cross-validated penalty
lasso = LassoCV(cv=5)
lasso.fit(X_scaled, y)

# Note: Treatment variable typically NOT penalized
# Use separate handling or post-LASSO approach
```

### For Causal Forests / Tree-Based Methods

```python
# Tree-based methods can handle:
# - Unscaled numeric variables (splits are scale-invariant)
# - Label-encoded categoricals (trees learn appropriate splits)

# However, standardization can still help with:
# - Variable importance interpretation
# - Combining with linear methods
# - Neural network components

from sklearn.ensemble import GradientBoostingRegressor

# Label encode categoricals
df_encoded = df.copy()
for col in categorical_vars:
    df_encoded[col] = LabelEncoder().fit_transform(df[col])

# Fit without scaling (tree-based model)
model = GradientBoostingRegressor()
model.fit(df_encoded[features], df_encoded['outcome'])
```

---

## Handling Special Cases

### High-Cardinality Categoricals

When a categorical has many levels:

```python
# Option 1: Group rare categories
def group_rare_categories(df, col, threshold=0.01):
    """Group categories appearing less than threshold proportion."""
    freq = df[col].value_counts(normalize=True)
    rare_cats = freq[freq < threshold].index
    df[col] = df[col].replace(rare_cats, 'Other')
    return df

# Option 2: Target encoding with strong smoothing
# (Use only baseline covariates, not outcome)

# Option 3: Use embedding (for deep learning)
```

### Missing Categories in Test Data

Handle categories not seen during training:

```python
from sklearn.preprocessing import OneHotEncoder

# Allow unknown categories
encoder = OneHotEncoder(
    sparse_output=False,
    drop='first',
    handle_unknown='ignore'  # Sets all dummies to 0 for unknown
)

encoder.fit(train[categorical_vars])
train_encoded = encoder.transform(train[categorical_vars])
test_encoded = encoder.transform(test[categorical_vars])  # Unknown categories handled
```

### Binary Variables

No encoding needed, but consider:

```python
# Ensure consistent coding
df['female'] = (df['gender'] == 'Female').astype(int)

# For multiple binary variables, standardization is optional
# but can help with regularization
```

---

## Summary: Encoding Decision Guide

| Variable Type | Recommended Encoding | Causal Notes |
|--------------|---------------------|--------------|
| Continuous | Standardization (z-score) | Fit on train only |
| Binary | Keep as 0/1 | No encoding needed |
| Ordinal | Ordinal encoding or dummies | Consider if ordering matters |
| Nominal (few levels) | One-hot, drop reference | Need overlap in all categories |
| Nominal (many levels) | Group + one-hot, or frequency | Avoid target encoding |
| High-cardinality ID | Exclude or embed | Usually not valid controls |

### Key Principles

1. **Pre-treatment only**: Encoding should not use outcome information
2. **Consistency**: Same encoding for treatment and control groups
3. **Train/test separation**: Fit encoders on training data only
4. **Documentation**: Record all encoding decisions
5. **Overlap**: Ensure all encoded levels have both treated and control units

---

## Code Recipe: Complete Preprocessing Pipeline

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def create_causal_preprocessor(numeric_vars, categorical_vars):
    """
    Create preprocessing pipeline safe for causal inference.

    Parameters
    ----------
    numeric_vars : list
        Continuous variables to standardize.
    categorical_vars : list
        Categorical variables to one-hot encode.

    Returns
    -------
    ColumnTransformer
        Fitted preprocessor.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_vars),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
             categorical_vars)
        ],
        remainder='passthrough'  # Keep other columns as-is
    )

    return preprocessor

# Usage
preprocessor = create_causal_preprocessor(
    numeric_vars=['age', 'income', 'years_experience'],
    categorical_vars=['region', 'education_level']
)

# Fit on training data
X_train_processed = preprocessor.fit_transform(train[features])

# Transform test data with training statistics
X_test_processed = preprocessor.transform(test[features])
```
