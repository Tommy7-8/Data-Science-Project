This project studies the return patterns of U.S. equities sorted by firm size in the Fama-French
tradition. Using monthly data from the Kenneth R. French Data Library, I work with readymade portfolios formed on size (10 deciles) and the Fama-French factors (including SMB) plus
the risk-free rate. The design is forecast-then-optimize: (1) predict next-month return and risk
(volatility) for each size portfolio using only lagged information; (2) convert these forecasts into
investable weights across the deciles.
Predictors include past 1–12 month returns, rolling realized volatility, and lags of common
factors (Mkt-RF, SMB). I compare a regularized linear model and a tree-based model for both
return and volatility forecasts. Evaluation follows a strict walk-forward setup: train on the past,
predict one month ahead, rebalance monthly. All transformations are refit inside the training
window. Transaction costs and turnover limits are included for realism.
Weights are computed by a long-only, fully invested optimizer. The primary implementation is
mean-variance with shrinkage covariance and weight caps; I also report a simple volatility-aware
allocation as a robustness check. Out-of-sample performance is compared with SMB, an equalweight mix of deciles, and the market, using annualized return, volatility, Sharpe and Sortino
ratios, maximum drawdown, turnover, and rolling stability plots. The question is practical: do
machine-learning forecasts of return and risk on size-sorted portfolios improve outcomes relative
to standard size benchmarks?