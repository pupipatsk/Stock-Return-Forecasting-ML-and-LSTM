# Stock Return Prediction

A pattern recognition project for predicting stock return from historical data using multiple machine learning models and techniques.

See the slide [Here](Presentation.pdf).

**Dataset**: S&P500, Yahoo Finance - [yfinance](https://pypi.org/project/yfinance/)

```shell
pip install -r requirements.txt
```

This will install all the necessary packages listed in the `requirements.txt` file.


## Outline

- Data Preprocessing
- Feature Engineering
  - Technical Indicators
    - Relative Strength Index (RSI)
    - Bollinger Bands
    - Average True Range (ATR)
    - Moving Average Convergence/Divergence (MACD)
    - Momentum
    - Lagged Return
- Data Splitting
  - Training and Testing Set: 2014 - 2023 (9y)
  - Trading Evaluation: 2023 - 2024 (1y)
- Model
  - Baseline: Naive Forecast
  - Neural Network: Architecture 1 (Timeseries) & 2 (TS+Exogenous)
    - Long Short Term Memory (LSTM)
    - Gated Recurrent Unit (GRU)
  - Classical: Regression & Classification
    - Linear regression
    - Logistic regression
    - Support Vector
    - Random Forest
    - Extreme Gradient Boosting
    - K-Nearest Neighbor
- Hyper parameters Tuning
  - Grid Search
  - Random Search
- Trading
  - Baseline Strategy: Buy and Hold
  - Equal Weight portfolio
  - Sharpe Ratio
- Analysis
  - Feature Importance

## Contributors

- [pupipatsk](https://github.com/pupipatsk) (Get)
- [warboo](https://github.com/warboo) (Veeliw)
- [markthitrin](https://github.com/markthitrin) (Mark)
- [Nacnano](https://github.com/nacnano) (Nac)
