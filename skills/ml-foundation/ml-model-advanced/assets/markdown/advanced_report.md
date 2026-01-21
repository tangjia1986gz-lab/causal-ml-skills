# Advanced Model Report Template

## Executive Summary

| Metric | Value |
|--------|-------|
| Model Type | {{MODEL_TYPE}} |
| Task | {{TASK}} |
| Dataset Size | {{N_SAMPLES}} samples, {{N_FEATURES}} features |
| Best CV Score | {{BEST_CV_SCORE}} |
| Test Score | {{TEST_SCORE}} |

## Model Configuration

### Selected Model

```
Model: {{MODEL_NAME}}
{{MODEL_PARAMS}}
```

### Preprocessing Pipeline

1. **Feature Scaling**: StandardScaler (required for SVM/MLP)
2. **Missing Value Handling**: {{MISSING_STRATEGY}}
3. **Categorical Encoding**: {{ENCODING_STRATEGY}}

## Training Results

### Cross-Validation Performance

| Fold | Train Score | Validation Score |
|------|-------------|------------------|
| 1 | {{CV_TRAIN_1}} | {{CV_VAL_1}} |
| 2 | {{CV_TRAIN_2}} | {{CV_VAL_2}} |
| 3 | {{CV_TRAIN_3}} | {{CV_VAL_3}} |
| 4 | {{CV_TRAIN_4}} | {{CV_VAL_4}} |
| 5 | {{CV_TRAIN_5}} | {{CV_VAL_5}} |
| **Mean** | **{{CV_TRAIN_MEAN}}** | **{{CV_VAL_MEAN}}** |
| **Std** | {{CV_TRAIN_STD}} | {{CV_VAL_STD}} |

### Overfitting Analysis

- **Train-Validation Gap**: {{OVERFIT_GAP}}
- **Overfitting Status**: {{OVERFIT_STATUS}}

Interpretation:
- Gap < 0.05: No overfitting
- Gap 0.05-0.10: Mild overfitting, acceptable
- Gap > 0.10: Significant overfitting, consider regularization

### Test Set Evaluation

{{#IF_CLASSIFICATION}}
| Metric | Value |
|--------|-------|
| Accuracy | {{ACCURACY}} |
| F1 Score | {{F1_SCORE}} |
| ROC AUC | {{ROC_AUC}} |
| Precision | {{PRECISION}} |
| Recall | {{RECALL}} |

**Confusion Matrix**

|  | Predicted 0 | Predicted 1 |
|--|-------------|-------------|
| Actual 0 | {{TN}} | {{FP}} |
| Actual 1 | {{FN}} | {{TP}} |
{{/IF_CLASSIFICATION}}

{{#IF_REGRESSION}}
| Metric | Value |
|--------|-------|
| R-squared | {{R2_SCORE}} |
| RMSE | {{RMSE}} |
| MAE | {{MAE}} |
| MAPE | {{MAPE}} |
{{/IF_REGRESSION}}

## Model-Specific Diagnostics

### SVM Diagnostics

{{#IF_SVM}}
- **Kernel**: {{KERNEL}}
- **C (Regularization)**: {{C_VALUE}}
- **Gamma**: {{GAMMA_VALUE}}
- **Number of Support Vectors**: {{N_SUPPORT_VECTORS}}
- **Support Vector Ratio**: {{SV_RATIO}}

**Interpretation**:
- High SV ratio (>50%): Model may be too complex or data is not well-separated
- Low SV ratio (<10%): Good margin separation
{{/IF_SVM}}

### MLP Diagnostics

{{#IF_MLP}}
- **Architecture**: {{ARCHITECTURE}}
- **Activation**: {{ACTIVATION}}
- **Alpha (L2)**: {{ALPHA}}
- **Learning Rate**: {{LEARNING_RATE}}
- **Iterations**: {{N_ITER}} / {{MAX_ITER}}
- **Converged**: {{CONVERGED}}
- **Final Loss**: {{FINAL_LOSS}}

**Convergence Status**: {{CONVERGENCE_STATUS}}

{{#IF_NOT_CONVERGED}}
**Recommendations**:
1. Increase max_iter (current: {{MAX_ITER}})
2. Adjust learning_rate_init (current: {{LEARNING_RATE}})
3. Reduce model complexity or increase regularization
{{/IF_NOT_CONVERGED}}
{{/IF_MLP}}

## Hyperparameter Tuning Results

{{#IF_TUNED}}
### Search Strategy

- **Method**: {{TUNING_METHOD}}
- **CV Folds**: {{TUNING_CV}}
- **Total Combinations**: {{N_COMBINATIONS}}

### Best Parameters

```
{{BEST_PARAMS}}
```

### Top 5 Configurations

| Rank | Parameters | CV Score |
|------|------------|----------|
| 1 | {{TOP1_PARAMS}} | {{TOP1_SCORE}} |
| 2 | {{TOP2_PARAMS}} | {{TOP2_SCORE}} |
| 3 | {{TOP3_PARAMS}} | {{TOP3_SCORE}} |
| 4 | {{TOP4_PARAMS}} | {{TOP4_SCORE}} |
| 5 | {{TOP5_PARAMS}} | {{TOP5_SCORE}} |
{{/IF_TUNED}}

## Model Comparison

{{#IF_COMPARISON}}
| Model | CV Score | Test Score | Training Time |
|-------|----------|------------|---------------|
{{COMPARISON_TABLE}}

**Best Model**: {{BEST_MODEL_NAME}} ({{BEST_MODEL_SCORE}})
{{/IF_COMPARISON}}

## Recommendations

### Model Selection

{{#IF_SIMPLE_SUFFICIENT}}
**Recommendation**: The simpler baseline model performs comparably to advanced models. Consider using the simpler model for:
- Better interpretability
- Faster training/prediction
- Lower maintenance burden
{{/IF_SIMPLE_SUFFICIENT}}

{{#IF_ADVANCED_BETTER}}
**Recommendation**: The advanced model provides significant improvement ({{IMPROVEMENT}}). Consider:
- Ensemble with baseline for robustness
- Further hyperparameter tuning
- Feature engineering to improve baseline
{{/IF_ADVANCED_BETTER}}

### Causal Inference Considerations

If using this model for causal inference tasks:

1. **Propensity Score Estimation**:
   - Check calibration of probability estimates
   - Ensure overlap assumption holds
   - Consider trimming extreme propensity scores

2. **Outcome Modeling in DDML**:
   - Use cross-fitting to prevent overfitting
   - Strong regularization recommended
   - Compare with gradient boosting baseline

3. **General Recommendations**:
   - Document model choices for reproducibility
   - Report sensitivity to hyperparameter choices
   - Consider interpretability requirements

## Appendix

### Feature Importance (if applicable)

{{#IF_FEATURE_IMPORTANCE}}
| Feature | Importance |
|---------|------------|
{{FEATURE_IMPORTANCE_TABLE}}
{{/IF_FEATURE_IMPORTANCE}}

### Learning Curve Analysis

{{#IF_LEARNING_CURVE}}
![Learning Curve]({{LEARNING_CURVE_PATH}})

**Interpretation**:
- {{LEARNING_CURVE_INTERPRETATION}}
{{/IF_LEARNING_CURVE}}

### Validation Curve Analysis

{{#IF_VALIDATION_CURVE}}
![Validation Curve]({{VALIDATION_CURVE_PATH}})

**Optimal Parameter**: {{OPTIMAL_PARAM}}
{{/IF_VALIDATION_CURVE}}

---

*Report generated: {{TIMESTAMP}}*
*Model version: {{MODEL_VERSION}}*
