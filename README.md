# Stock Return Forecasting
This project focuses on forecasting stock returns from historical data using various machine learning models and neural networks, specifically Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU).

Presentation Slide: [View Presentation](Presentation.pdf)

**Dataset**:S&P 500 data from Yahoo Finance, accessed via [yfinance](https://pypi.org/project/yfinance/)


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
  - Neural Network:
    - Long Short Term Memory (LSTM)
      - LSTM architecture 1 (Timeseries)
      - LSTM architecture 2 (Timeseries + Exogenous Features)
    - Gated Recurrent Unit (GRU)
      - GRU architecture 1 (Timeseries)
      - GRU architecture 2 (Timeseries + Exogenous Features)
  - Classical Models:
    - Linear Regression
    - Logistic Regression
    - Support Vector Machines (SVM)
    - Random Forest
    - Extreme Gradient Boosting (XGBoost)
    - K-Nearest Neighbors (KNN)
- Hyperparameter Tuning
  - Grid Search
  - Random Search
- Trading
  - Baseline Strategy: Buy and Hold
  - Equal Weight portfolio
  - Sharpe Ratio
- Analysis
  - Feature Importance

## Requirements
To install the necessary packages, run:
```shell
pip install -r requirements.txt
```
This will install all the necessary packages listed in the `requirements.txt` file.\
For PyTorch installation, please visit pytorch.org for specific instructions.

## Directory Structure
```txt
.
├── LICENSE
├── Presentation.pdf
├── README.md
├── archive
│   ├── DatasetLogReturn.py
│   ├── LinearRegression.ipynb
│   ├── SVM.ipynb
│   ├── SentimentAnalysis.py
│   ├── TrendsLoader.py
│   ├── download_sp500_data.py
│   ├── main-2024_05_09.ipynb
│   └── tickers_sp500.txt
├── main.ipynb
├── main.py
├── models
│   ├── GRU
│   │   ├── GRU1.pth.tar
│   │   └── GRU2.pth.tar
│   ├── LSTM
│   │   ├── LSTM1.pth.tar
│   │   ├── LSTM2.pth.tar
│   │   ├── LSTMFULL1.pth.tar
│   │   ├── train_losses-LSTMFULL1.txt
│   │   ├── train_losses.txt
│   │   ├── val_losses-LSTMFULL1.txt
│   │   └── val_losses.txt
│   ├── LinearRegression
│   │   ├── LinearRegression.sav
│   │   └── LinearRegression_log.sav
│   └── SVM
│       ├── SVM.sav
│       └── SVM_log.sav
├── notebooks
│   ├── FeatureImportance.ipynb
│   ├── GRU1.ipynb
│   ├── GRU2.ipynb
│   ├── LSTM1.ipynb
│   └── LSTM2.ipynb
├── requirements.txt
└── scripts
    ├── FeatureImportance.py
    ├── GRU1.py
    ├── GRU2.py
    ├── LSTM1.py
    ├── LSTM2.py
    └── __pycache__
```

## Contributors
- [pupipatsk](https://github.com/pupipatsk) (Pupipat Singkhorn - Get)
- [warboo](https://github.com/warboo) (Veeliw)
- [markthitrin](https://github.com/markthitrin) (Mark)
- [Nacnano](https://github.com/nacnano) (Nac)
