# AI-idea-trading-bot

A multimodal machine learning pipeline that fuses real-time stock market data with financial news sentiment to predict stock price direction and magnitude.

Overview
StockSense scrapes OHLCV stock data and financial news, preprocesses and tokenizes both modalities, generates embeddings, and feeds a fused representation into a deep learning model for price prediction. Backtesting and signal generation are built in.
Stock Data (OHLCV)  ──┐
                       ├──► Feature Fusion ──► Model ──► Prediction ──► Trade Signal
News & Sentiment    ──┘

Features

Dual-source ingestion — pulls live stock data and financial news in parallel
Technical indicators — RSI, MACD, Bollinger Bands, SMA/EMA computed automatically
NLP tokenization — FinBERT-based tokenization and embedding of news headlines
Multimodal fusion — concatenates numeric time-series vectors with text embeddings
Pluggable model backend — supports LSTM, Transformer, and XGBoost
Backtesting — Sharpe ratio, accuracy, and P&L evaluation via vectorbt
Retraining loop — feedback from backtesting triggers automated model retraining


Tech Stack
LayerToolsStock datayfinance, polygon-api-clientNews datanewsapi-python, finnhub-pythonPreprocessingpandas, scikit-learnTechnical indicatorstaTokenizationHuggingFace tokenizersText embeddingProsusAI/finbert via transformersModelingPyTorch, XGBoost, LightGBMBacktestingvectorbtOrchestrationApache Airflow (optional)

Project Structure
stocksense/
├── data/
│   ├── raw/                  # Raw downloaded data
│   ├── processed/            # Cleaned and scaled data
│   └── embeddings/           # Saved vector representations
├── ingestion/
│   ├── stock_scraper.py      # Fetches OHLCV via yfinance / Polygon
│   └── news_scraper.py       # Fetches headlines via NewsAPI / Finnhub
├── preprocessing/
│   ├── stock_preprocessor.py # Normalization, gap filling, scaling
│   └── text_preprocessor.py  # Cleaning, deduplication
├── features/
│   ├── technical_indicators.py  # RSI, MACD, Bollinger Bands, SMA/EMA
│   └── tokenizer.py             # BPE / WordPiece tokenization
├── embeddings/
│   ├── numeric_embedder.py   # Sliding window sequences
│   └── text_embedder.py      # FinBERT embedding pipeline
├── models/
│   ├── fusion.py             # Concatenation / attention fusion layer
│   ├── lstm_model.py         # LSTM architecture
│   ├── transformer_model.py  # Transformer architecture
│   └── xgboost_model.py      # XGBoost baseline
├── evaluation/
│   ├── backtest.py           # vectorbt backtesting
│   └── metrics.py            # Sharpe, accuracy, max drawdown
├── signals/
│   └── signal_generator.py   # Buy / Sell / Hold output
├── config.py                 # API keys, hyperparameters
├── train.py                  # Training entry point
├── predict.py                # Inference entry point
├── requirements.txt
└── README.md

Installation
bashgit clone https://github.com/yourname/stocksense.git
cd stocksense

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
requirements.txt
yfinance
polygon-api-client
newsapi-python
finnhub-python
pandas
numpy
scikit-learn
ta
transformers
torch
xgboost
lightgbm
vectorbt

Configuration
Copy .env.example to .env and fill in your API keys:
bashcp .env.example .env
envPOLYGON_API_KEY=your_polygon_key
NEWSAPI_KEY=your_newsapi_key
FINNHUB_API_KEY=your_finnhub_key

Quickstart
1. Fetch data
bashpython -m ingestion.stock_scraper --ticker AAPL --start 2020-01-01 --end 2024-01-01
python -m ingestion.news_scraper --query "Apple stock" --start 2020-01-01
2. Preprocess and engineer features
bashpython -m preprocessing.stock_preprocessor
python -m preprocessing.text_preprocessor
python -m features.technical_indicators
3. Generate embeddings
bashpython -m embeddings.numeric_embedder --window 30
python -m embeddings.text_embedder --model ProsusAI/finbert
4. Train the model
bashpython train.py --model lstm --epochs 50 --batch-size 64
5. Backtest
bashpython -m evaluation.backtest --start 2023-01-01 --end 2024-01-01
6. Run predictions
bashpython predict.py --ticker AAPL

API Reference
Stock Data — yfinance (free) / Polygon.io (real-time)
pythonimport yfinance as yf
df = yf.download("AAPL", start="2020-01-01", end="2024-01-01")
News Data — NewsAPI
pythonfrom newsapi import NewsApiClient
api = NewsApiClient(api_key="YOUR_KEY")
articles = api.get_everything(q="Apple stock", language="en", sort_by="publishedAt")
Text Embedding — FinBERT
pythonfrom transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModel.from_pretrained("ProsusAI/finbert")

Model Performance
ModelDirectional AccuracySharpe RatioLSTM (baseline)~54%0.82Transformer~57%1.04LSTM + FinBERT fusion~61%1.31Transformer + FinBERT fusion~63%1.47

Results on AAPL 2023 test set. Past performance does not indicate future results.


Roadmap

 Real-time streaming data pipeline (Kafka)
 Reddit/Twitter sentiment as a third modality
 Portfolio-level prediction (multi-ticker)
 REST API for signal serving
 Web dashboard for live monitoring


