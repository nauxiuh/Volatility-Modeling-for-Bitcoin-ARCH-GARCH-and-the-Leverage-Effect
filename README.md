# Volatility Modeling for Bitcoin: ARCH, GARCH, and the Leverage Effect

This project fits a progression of volatility models to Bitcoin daily returns, from simple ARCH to asymmetric GARCH variants, and asks whether the extra complexity actually helps. The findings are financially meaningful: Bitcoin volatility has strong persistence that ARCH alone handles poorly, GARCH(1,1) solves this with far fewer parameters, and unlike equities, Bitcoin does not exhibit a significant leverage effect in the period studied.

## Repository Structure

```
├── FinancialDataAnalysis4.ipynb   # Main analysis notebook
└── bitcoin_data.csv               # Bitcoin daily price data
```

## Analysis

### Part 1: ARCH(p) Model Selection

ARCH models express conditional variance as a weighted sum of past squared returns. The more lags included, the longer the "memory" of the model.

| Model | AIC | BIC |
|---|---|---|
| ARCH(1) | 1771.81 | 1783.50 |
| ARCH(2) | 1766.41 | 1781.99 |
| ARCH(3) | 1759.59 | 1779.05 |
| ARCH(4) | 1754.99 | 1778.33 |
| ARCH(5) | 1747.78 | 1774.98 |

Both AIC and BIC keep falling as p increases. Within the ARCH family, you need at least 5 lags before the fit stabilizes. This is a sign of long memory in Bitcoin's volatility: a large return today has consequences that stretch well beyond yesterday. The ARCH(1) model, which only looks back one day, is too shortsighted for this data.

**ARCH(1) unconditional moments** (estimated parameters: alpha_0 = 6.8205, alpha_1 = 0.1015):
- Mean: 0.2219
- Variance: alpha_0 / (1 - alpha_1) = 7.5914
- Kurtosis: 3(1 - alpha_1^2) / (1 - 3*alpha_1^2) = 3.0638

The kurtosis is only marginally above 3 under ARCH(1). This is a known weakness: ARCH(1) tends to understate fat tails because it cannot account for the full persistence structure.

**Is AR(0) sufficient for the mean?** The Ljung-Box test on raw returns gives a p-value of 0.5991 at lag 10. No significant serial correlation, so a constant mean model is appropriate. There is no need to fit an ARMA structure to the mean equation.

### Part 2: GARCH(1,1) - The Parsimonious Fix

GARCH(1,1) adds one extra parameter: a lagged variance term. This allows yesterday's variance to carry forward into today, which is exactly the persistence structure that ARCH needed five lags to approximate.

```
sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2
```

**Fitted parameters:** omega = 0.9267, alpha = 0.0765, beta = 0.8004

Persistence (alpha + beta) = 0.877. This means volatility shocks take a long time to decay, roughly 8 to 9 days to halve in magnitude. That reflects Bitcoin's well-known tendency to remain elevated in volatility after a major move.

**Unconditional moments:**
- Mean: 0.2048
- Variance: omega / (1 - alpha - beta) = 7.5294
- Kurtosis: 3 + 6*alpha^2 / (1 - 3*alpha^2 - 2*alpha*beta - beta^2) = 3.16

**Model fit:** AIC = 1772.57, which is comparable to ARCH(5) (AIC = 1747.78) but with 3 fewer parameters. GARCH(1,1) is the standard choice in practice for exactly this reason: it captures long-range volatility persistence without overfitting.

**Innovation distribution comparison:**

The choice of distribution for standardized residuals matters when Bitcoin returns have fat tails. Four distributions are compared across GARCH(1,1):

- Normal: baseline, tends to underestimate the probability of extreme moves
- Student-t: heavier tails, better fit for crypto
- Skewed Student-t: adds asymmetry, captures the slight negative skew in returns
- GED (Generalized Error Distribution): flexible tail behavior

The overall shape of the conditional volatility series is consistent across all four distributions. The differences show up in the magnitude of volatility spikes during extreme events, where fat-tailed distributions give more realistic estimates. For risk management or options pricing on Bitcoin, Student-t or Skewed Student-t is the better choice.

### Part 3: Symmetric vs Asymmetric Models

The leverage effect refers to the empirical observation in equity markets that stock prices falling increases volatility more than prices rising by the same amount. The intuition is that falling prices increase financial leverage and uncertainty.

Two asymmetric extensions of GARCH are tested:

**GJR-GARCH(1,1):** Adds a term that activates only when the shock is negative, allowing bad news to hit volatility harder than good news.

**EGARCH(1,1):** Models the log of variance, so the conditional variance is always positive without imposing parameter constraints. The leverage parameter gamma captures whether negative shocks have an outsized effect.

| Model | Log-Lik | AIC | BIC |
|---|---|---|---|
| GARCH(1,1) | -882.28 | 1772.57 | 1788.17 |
| GJR-GARCH(1,1) | -881.93 | 1773.87 | 1793.37 |
| EGARCH(1,1) | -881.63 | 1773.26 | 1792.76 |

GARCH(1,1) wins on both AIC and BIC. The asymmetric models have slightly higher log-likelihoods but the penalty for the extra parameter is not worth it based on the information criteria.

The asymmetry parameters themselves tell the same story: GJR-GARCH estimated gamma at 0.0397 and EGARCH at -0.0237. Both are close to zero. In equity markets these values are typically much larger in magnitude. Bitcoin in this period reacted roughly symmetrically to positive and negative shocks, which makes sense given its market structure and the absence of traditional leverage-based feedback loops that drive the equity leverage effect.

The conditional volatility plots for all three models are nearly identical, confirming that the differences in specification are not materially affecting the estimates.

## Key Takeaways

- Bitcoin volatility has long memory. ARCH(1) is not enough; AIC/BIC keep improving through ARCH(5), which is the motivation for GARCH.
- GARCH(1,1) captures the same persistence as ARCH(5) with fewer parameters. This is why it remains the industry default for volatility modeling.
- Innovation distribution matters for tail risk. Student-t and GED distributions give more realistic estimates of extreme events than assuming normality.
- Bitcoin does not have a strong leverage effect in the 2024 data. The symmetric GARCH(1,1) is statistically preferred over GJR-GARCH and EGARCH, which is a meaningful difference from equity markets.
- The AR(0) constant mean is appropriate. Ljung-Box confirms no serial correlation in returns, consistent with the findings from projects 2 and 3.

## Dependencies

```bash
pip install numpy pandas matplotlib arch statsmodels
```

## Usage

Rename the data file to `bitcoin_data.csv`, then run:

```bash
jupyter notebook FinancialDataAnalysis4.ipynb
```
