---
name: time-series-econometrics
description: Use for time series analysis and forecasting. Triggers on ARIMA, VAR, cointegration, unit root, Granger causality, stationarity, ADF test.
---

# Time Series Econometrics for Causal Inference

## Overview

Time series econometrics provides tools for analyzing temporal data patterns and relationships. While these methods are powerful for forecasting and understanding dynamic relationships, their causal interpretations require careful consideration.

**Critical Distinction**: Granger causality and other time series methods test for *predictive* relationships, not true causal effects. A variable X "Granger-causes" Y if past values of X help predict Y beyond Y's own past - this is statistical precedence, not causation.

## When to Use This Skill

- Analyzing temporal data for stationarity and trends
- Building forecasting models (ARIMA, VAR)
- Testing for long-run equilibrium relationships (cointegration)
- Examining lead-lag relationships between variables
- Conducting Granger causality tests (with proper interpretation)
- Studying impulse responses and dynamic effects

## Key Concepts

### Stationarity
A time series is stationary if its statistical properties (mean, variance, autocorrelation) do not change over time. Most econometric methods require stationarity or appropriate differencing.

### Unit Roots
A unit root indicates a non-stationary process where shocks have permanent effects. The Augmented Dickey-Fuller (ADF) and KPSS tests help diagnose unit roots.

### ARIMA Models
Autoregressive Integrated Moving Average models capture:
- AR(p): Dependence on past values
- I(d): Differencing for stationarity
- MA(q): Dependence on past forecast errors

### VAR Models
Vector Autoregression models jointly model multiple time series, allowing for:
- Dynamic interdependencies
- Impulse response analysis
- Variance decomposition
- Granger causality testing

### Cointegration
When non-stationary series share a common stochastic trend, their linear combination may be stationary. This indicates a long-run equilibrium relationship.

## Workflow

```
1. Data Preparation
   - Visualize series
   - Check for structural breaks
   - Handle missing values

2. Stationarity Analysis
   - Plot ACF/PACF
   - Conduct ADF and KPSS tests
   - Determine differencing order

3. Model Selection
   - For single series: ARIMA with AIC/BIC
   - For multiple series: VAR with lag selection
   - Test for cointegration if I(1)

4. Estimation & Diagnostics
   - Fit chosen model
   - Check residual autocorrelation
   - Test for heteroskedasticity

5. Analysis & Interpretation
   - Granger causality tests
   - Impulse response functions
   - Variance decomposition

6. Causal Interpretation (with caution)
   - Distinguish prediction from causation
   - Consider confounders
   - Acknowledge limitations
```

## Available Tools

### Testing Functions
- `test_unit_root()` - ADF and Phillips-Perron tests
- `test_stationarity()` - KPSS test (null: stationary)
- `cointegration_test()` - Engle-Granger and Johansen tests

### Modeling Functions
- `fit_arima()` - Fit ARIMA models with auto-selection
- `fit_var()` - Fit VAR models with diagnostics
- `fit_vecm()` - Fit Vector Error Correction Models

### Analysis Functions
- `granger_causality()` - Granger causality tests
- `impulse_response()` - IRF computation and plotting
- `variance_decomposition()` - FEVD analysis

## Causal Inference Limitations

**What Time Series Methods CAN Tell You:**
- Predictive relationships
- Lead-lag dynamics
- Long-run equilibrium relationships
- Dynamic responses to shocks

**What Time Series Methods CANNOT Tell You:**
- True causal effects
- What happens under interventions
- Counterfactual outcomes

**For True Causal Inference with Time Series Data:**
- Consider difference-in-differences
- Use synthetic control methods
- Apply interrupted time series design
- Leverage natural experiments

## References

See the `references/` directory for detailed documentation on:
- Stationarity testing and unit roots
- ARIMA model specification
- VAR models and Granger causality
- Cointegration and error correction
- Causal interpretation of time series results
