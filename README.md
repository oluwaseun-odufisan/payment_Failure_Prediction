# Payment Failure and Chargeback Prediction System

## Overview

The **Payment Failure and Chargeback Prediction System** is an advanced machine learning-powered solution designed to predict payment transaction outcomes (Success, Failed, or Chargeback) in real-time, optimize transaction processing, and provide actionable analytics for fintech applications. Built with a modular architecture, it integrates deep learning (LSTM and Transformer models), gradient boosting (XGBoost), and ensemble methods to achieve high accuracy in predicting transaction statuses. The system includes a real-time API for predictions, a feedback loop for continuous model improvement, and an interactive analytics dashboard for visualizing transaction patterns and model performance.

This project is tailored for:
- **Fintech Professionals**: To reduce payment failures, mitigate chargeback risks, and optimize processor pathways.
- **Data Scientists and Researchers**: To experiment with advanced ML models (LSTM, Transformer, XGBoost, ensemble) and analyze payment data.
- **Engineers**: To deploy a scalable, production-ready system with robust API services and real-time monitoring.

The system leverages synthetic data generation to simulate realistic payment scenarios, making it ideal for testing and development in controlled environments. It is built to handle high transaction volumes, incorporate feedback for model retraining, and provide detailed visualizations for strategic decision-making.

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Data Structure](#data-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Data Generation](#data-generation)
7. [Model Details](#model-details)
8. [API Endpoints](#api-endpoints)
9. [Analytics Dashboard](#analytics-dashboard)
10. [Feedback Loop](#feedback-loop)
11. [Configuration](#configuration)
12. [Logging and Monitoring](#logging-and-monitoring)
13. [Troubleshooting](#troubleshooting)


## Features

- **Multi-Model Prediction**:
  - LSTM model for sequential transaction analysis.
  - Transformer model with custom attention layers for capturing long-range dependencies.
  - XGBoost model optimized with Optuna for feature-based predictions.
  - Ensemble meta-learner (Logistic Regression) combining predictions for improved accuracy.
- **Real-Time API**:
  - FastAPI-based endpoint (`/predict`) for real-time transaction predictions.
  - Rate limiting and Redis caching for performance and scalability.
  - Simulated Flutterwave API integration for transaction processing.
- **Analytics Dashboard**:
  - Interactive Dash-based dashboard with 15 visualizations (e.g., status distribution, fraud risk histogram, ROC curves).
  - Filters for date range, payment method, location, and model type.
  - Exportable visualizations in PNG and JSON formats.
- **Feedback Loop**:
  - Continuous model retraining triggered by transaction outcomes (0.5% chance per transaction).
  - Updates processor metrics for real-time performance tracking.
- **Data Generation**:
  - Synthetic data generation for transactions, users, merchants, and processor metrics.
  - Ensures sufficient data volume (100,000+ transactions) for sequence modeling.
- **Scalability and Robustness**:
  - Configurable parameters via `config.yaml`.
  - Comprehensive logging with file and console output.
  - Error handling and retry logic for transaction processing.
- **Visualization Support**:
  - Generates data flow, model architecture, and performance diagrams.
  - Handles Kaleido issues with fallback to JSON exports.

## Architecture

The system is modular, with distinct components interacting logically :

1. **Configuration (`Config`)**:
   - Loads settings from `config.yaml` (e.g., file paths, model hyperparameters).
   - Defines payment methods, processor pathways, and class weights.

2. **Data Preparation (`DataPreparation`)**:
   - Loads and validates CSV files (`transactions.csv`, `users.csv`, `merchants.csv`, `processor_metrics.csv`).
   - Merges data sources and performs feature engineering (e.g., `amount_log`, `risk_composite`).
   - Creates transaction sequences for deep learning models (default length: 15).

3. **Data Preprocessing (`DataPreprocessor`)**:
   - Builds a preprocessing pipeline with `RobustScaler` for numerical features and `OneHotEncoder` for categorical features.
   - Caches preprocessed data for efficiency.
   - Saves and loads preprocessor state using `pickle`.

4. **Model Building (`PaymentModel`)**:
   - **LSTM Model**: Three-layer LSTM with attention and batch normalization.
   - **Transformer Model**: Custom `DynamicTransformerLayer` with multi-head attention and GELU activation.
   - **XGBoost Model**: Hyperparameter-tuned using Optuna for optimal performance.
   - **Ensemble Model**: Combines predictions using a Logistic Regression meta-learner.
   - Saves and loads models using `h5` (TensorFlow) and `pickle` (XGBoost, ensemble).

5. **Transaction Processing (`TransactionProcessor`)**:
   - Simulates Flutterwave API calls with retry logic (default: 5 retries, 2-second delay).
   - Returns transaction outcomes with cost, latency, and success rate.

6. **Feedback Loop (`FeedbackLoop`)**:
   - Queues transaction results and updates processor metrics.
   - Triggers model retraining based on a 0.5% probability per transaction.

7. **Analytics Dashboard (`AnalyticsDashboard`)**:
   - Built with Dash, providing real-time visualizations.
   - Updates every 20 seconds with filters for data exploration.

8. **API Service (`APIService`)**:
   - FastAPI server with a `/predict` endpoint for transaction predictions.
   - Integrates with Redis for caching (2-hour TTL) and rate limiting (100 requests/minute).

9. **Main Application (`PaymentPredictionApp`)**:
   - Orchestrates initialization, training, and execution.
   - Runs API and dashboard in separate processes for concurrency.

## Data Structure

The system relies on four CSV files stored in the `data/` directory:

### transactions.csv
- **Columns**:
  - `transaction_id` (str): Unique ID (e.g., `TXN_00000001`).
  - `timestamp` (datetime): Transaction time (e.g., `2025-07-23T11:18:00`).
  - `customer_id` (str): Links to `users.csv` (e.g., `CUST_000001`).
  - `merchant_id` (str): Links to `merchants.csv` (e.g., `MERCH_0001`).
  - `amount` (float): Transaction amount (10–100,000).
  - `currency` (str): `NGN`, `USD`, `EUR`.
  - `payment_method` (str): `Card`, `Bank_Transfer`, `USSD`, `Mobile_Money`.
  - `processor_pathway` (str): e.g., `Card_Visa`, `Bank_Transfer_GTBank`.
  - `status` (str): `Success`, `Failed`, `Chargeback`.
  - `failure_reason` (str): e.g., `Network_Error`, `None`.
  - `chargeback_reason` (str): e.g., `Fraudulent`, `None`.
  - `merchant_type` (str): e.g., `E-commerce`, `Retail`.
  - `fraud_risk_score` (float): 0–1.
  - `device_type` (str): `Mobile`, `Desktop`, `Tablet`, `POS`.
  - `network_type` (str): `4G`, `3G`, `WiFi`.
  - `location` (str): e.g., `Lagos`, `Abuja`.
  - `transaction_type` (str): `Purchase`, `Refund`, `Transfer`.
  - `hour_of_day` (int): 0–23.
  - `day_of_week` (int): 0–6.
  - `is_weekend` (int): 0 or 1.
  - `time_since_last_transaction` (float): Seconds since last transaction.
- **Size**: 100,000+ rows, ensuring each customer has ≥15 transactions.

### users.csv
- **Columns**:
  - `customer_id` (str): Unique ID.
  - `user_risk_score` (float): 0–1.
  - `average_transaction_frequency` (float): Transactions per period (0.1–100).
- **Size**: 1,000 rows.

### merchants.csv
- **Columns**:
  - `merchant_id` (str): Unique ID.
  - `merchant_risk_score` (float): 0–1.
  - `merchant_type` (str): e.g., `E-commerce`, `Retail`.
- **Size**: 500 rows.

### processor_metrics.csv
- **Columns**:
  - `processor_pathway` (str): e.g., `Card_Visa`.
  - `cost` (float): 0.01–0.05.
  - `latency` (float): 100–2000 ms.
  - `success_rate` (float): 0.9–1.0.
  - `network_stability` (float): 0.85–1.0.
  - `timestamp` (datetime).
  - `availability` (float): 1.0.
  - `failure_reason` (str): e.g., `None`.
- **Size**: 10,000 rows.


## Installation

### Prerequisites
- **Python**: 3.8+
- **Dependencies**:
  ```bash
  pip install pandas numpy tensorflow scikit-learn xgboost optuna matplotlib seaborn plotly dash fastapi uvicorn redis psutil pyyaml
  ```
- **Kaleido** (for Plotly visualizations):
  ```bash
  pip install kaleido==0.2.1
  ```
- **Redis**: Install and run a Redis server locally:
  ```bash
  sudo apt-get install redis-server  # Ubuntu
  brew install redis  # macOS
  redis-server  # Start Redis
  ```

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/payment-prediction-system.git
   cd payment-prediction-system
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Create a `requirements.txt` with:
   ```text
   pandas
   numpy
   tensorflow
   scikit-learn
   xgboost
   optuna
   matplotlib
   seaborn
   plotly
   dash
   fastapi
   uvicorn
   redis
   psutil
   pyyaml
   kaleido==0.2.1
   ```

3. **Generate Data**:
   Run the data generation script to create synthetic data:
   ```bash
   python generate_payment_data.py
   ```
   This creates `data/transactions.csv`, `data/users.csv`, `data/merchants.csv`, and `data/processor_metrics.csv`.

4. **Verify Redis**:
   Ensure Redis is running on `localhost:6379`:
   ```bash
   redis-cli ping
   ```
   Expected output: `PONG`

## Usage

1. **Initialize and Train Models**:
   Run the main application to initialize data, preprocess, train models, and start services:
   ```bash
   python payment_failure_prediction_ultra.py
   ```
   This:
   - Loads and validates data.
   - Trains LSTM, Transformer, XGBoost, and ensemble models.
   - Starts the API (`http://0.0.0.0:8000`) and dashboard (`http://0.0.0.0:8050`).

2. **Access the API**:
   - **Endpoint**: `POST /predict`
     ```bash
     curl -X POST http://0.0.0.0:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
         "transaction_id": "TXN_123",
         "customer_id": "CUST_456",
         "merchant_id": "MERCH_789",
         "amount": 100.50,
         "currency": "NGN",
         "payment_method": "Card",
         "processor_pathway": "Card_Verve",
         "merchant_type": "E-commerce",
         "fraud_risk_score": 0.3,
         "device_type": "Mobile",
         "network_type": "4G",
         "location": "Lagos",
         "transaction_type": "Purchase",
         "timestamp": "2025-07-23T11:18:00",
         "hour_of_day": 11,
         "day_of_week": 2,
         "is_weekend": 0,
         "time_since_last_transaction": 3600.0
     }'
     ```
     **Response**:
     ```json
     {
       "transaction_id": "TXN_123",
       "predicted_status": "Success",
       "confidence": 0.85,
       "probabilities": {"Success": 0.85, "Failed": 0.10, "Chargeback": 0.05},
       "processor_response": {"status": "success"}
     }
     ```

3. **Access the Dashboard**:
   - Open `http://0.0.0.0:8050` in a browser.
   - Use filters (date range, payment method, location, model type) to explore visualizations.
   - Click "Export Visualizations" to save PNG and JSON files to `data/exported_visualizations/`.

4. **Monitor Logs**:
   - Check `payment_prediction_ultra.log` for runtime logs.
   - Check `generate_payment_data.log` for data generation logs.

## Data Generation

The `generate_payment_data.py` script creates synthetic data for testing and development:

- **Command**:
  ```bash
  python generate_payment_data.py
  ```

- **Output**:
  - `data/transactions.csv`: 100,000+ transactions, ensuring ≥15 transactions per customer.
  - `data/users.csv`: 1,000 users.
  - `data/merchants.csv`: 500 merchants.
  - `data/processor_metrics.csv`: 10,000 processor records.

- **Customization**:
  - Modify `num_transactions`, `num_users`, `num_merchants`, `num_processor_records` in `DataGenerator.__init__`.
  - Adjust distributions (e.g., `amount` log-normal mean/sigma, `status` weights) for specific use cases.

- **Key Features**:
  - Realistic distributions: Log-normal for amounts, uniform for risk scores, weighted statuses (85% Success, 10% Failed, 5% Chargeback).
  - Ensures sequence compatibility (`sequence_length=15`).
  - Logs generation progress to `generate_payment_data.log`.

## Model Details

### LSTM Model
- **Architecture**:
  - **Input**: Sequences of shape `(sequence_length, num_features)`.
  - **Layers**: LSTM(512) → BatchNorm → Dropout(0.4) → LSTM(256) → Attention → LSTM(128) → Dense(64, GELU) → Dense(3, softmax).
  - **Optimizer**: Adam (learning rate: 0.0003, clipnorm=1.0).
  - **Loss**: Sparse categorical crossentropy with class weights `{0: 1.0, 1: 6.0, 2: 12.0}`.
- **Training**:
  - **Epochs**: 150 (configurable).
  - **Batch Size**: 256.
  - **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard.
- **Output**: Saved to `data/models/lstm_model.h5`.

### Transformer Model
- **Architecture**:
  - **Input**: Sequences of shape `(sequence_length, num_features)`.
  - **Layers**: 4× `DynamicTransformerLayer` (12 heads, dff=1024) → GlobalAveragePooling1D → Dense(128, GELU) → BatchNorm → Dropout(0.4) → Dense(3, softmax).
  - **Custom Layer**: `DynamicTransformerLayer` with multi-head attention and GELU feed-forward network.
  - **Optimizer**: Adam (learning rate: 0.0003, clipnorm=1.0).
- **Training**: Same as LSTM.
- **Output**: Saved to `data/models/transformer_model.h5`.

### XGBoost Model
- **Architecture**:
  - Hyperparameters tuned with Optuna (n_estimators, max_depth, learning_rate, etc.).
  - **Objective**: `multi:softmax` (3 classes).
- **Training**:
  - 50 Optuna trials to maximize weighted F1-score.
  - Uses preprocessed flat features (numerical + one-hot encoded).
- **Output**: Saved to `data/models/xgb_model.pkl`.

### Ensemble Model
- **Architecture**:
  - **Meta-learner**: Logistic Regression (multinomial, max_iter=1000).
  - Combines predictions from LSTM, Transformer, and XGBoost.
- **Training**:
  - Stacks prediction probabilities into a meta-feature matrix.
- **Output**: Saved to `data/models/ensemble_model.pkl`.

### Preprocessing
- **Pipeline**:
  - **Numerical**: `RobustScaler` for columns like `amount`, `fraud_risk_score`.
  - **Categorical**: `OneHotEncoder` for columns like `payment_method`, `processor_pathway`.
- **Output**: Saved to `data/models/preprocessor.pkl`.

## API Endpoints

- **POST /predict**:
  - **Input**: JSON with transaction details (see `TransactionRequest` schema in `payment_failure_prediction_ultra.py`).
  - **Output**: Predicted status, confidence, probabilities, and processor response.
  - **Features**: Redis caching (2-hour TTL), rate limiting (100 requests/minute).
- **GET /health**:
  - **Output**: `{"status": "healthy", "timestamp": "2025-07-24T06:17:00"}`.

## Analytics Dashboard

- **URL**: `http://0.0.0.0:8050`
- **Visualizations** (15 total):
  - Status Pie Chart
  - Failure Reasons Bar
  - Chargeback Reasons Bar
  - Volume Time Series
  - Fraud Risk Histogram
  - Success Rate by Payment Method
  - Latency Box Plot
  - Cost vs. Success Rate Scatter
  - Merchant Chargeback Rate
  - Prediction Confidence
  - ROC Curve
  - Feature Importance (XGBoost/Ensemble only)
  - Attention Weights (Transformer only)
  - Transaction Velocity
  - System Metrics (CPU/memory usage)
- **Features**:
  - Filters: Date range, payment method, location, model type.
  - Auto-updates every 20 seconds.
  - Export to `data/exported_visualizations/` as PNG/JSON.
- **Dependencies**: Dash, Plotly, Kaleido.

## Feedback Loop

- **Functionality**:
  - Queues transaction results and updates `processor_metrics.csv`.
  - Retrains models with 0.5% probability per transaction.
- **Threading**: Runs in a separate thread to avoid blocking API/dashboard.
- **Metrics**: Tracks cost, latency, success rate, and network stability.

## Configuration

- **File**: `config.yaml`
- **Key Parameters**:
  - `data_dir`: `data` (default).
  - `transaction_file`: `data/transactions.csv`.
  - `user_file`: `data/users.csv`.
  - `merchant_file`: `data/merchants.csv`.
  - `processor_file`: `data/processor_metrics.csv`.
  - `model_dir`: `data/models`.
  - `sequence_length`: 15.
  - `batch_size`: 256.
  - `epochs`: 150.
  - `learning_rate`: 0.0003.
  - `cache_ttl`: 7200 (seconds).
  - `rate_limit`: 100 (requests/minute).
- **Customization**:
  - Edit `config.yaml` to override defaults.
  - Default config is created if file is missing.

## Logging and Monitoring

- **Logs**:
  - `payment_prediction_ultra.log`: Runtime logs for the main application.
  - `generate_payment_data.log`: Data generation logs.
  - **Format**: `%(asctime)s - %(levelname)s - %(message)s - [File: %(filename)s, Line: %(lineno)d, Process: %(process)d]`.
- **Metrics**:
  - Tracked via `CustomMetrics` class: request count, prediction latency, memory usage.
  - Stored in `data/metrics.log`.
- **System Monitoring**:
  - CPU and memory usage visualized in the dashboard (`system_metrics` plot).
  - Uses `psutil` for real-time metrics.

## Troubleshooting

- **Error: `NameError: name 'defaultdict' is not defined`**:
  - Ensure `from collections import defaultdict` is included in `generate_payment_data.py`.
- **Kaleido Warnings**:
  - Install `kaleido==0.2.1`:
    ```bash
    pip install kaleido==0.2.1
    ```
  - Alternatively, use a `matplotlib`-based version of `payment_failure_prediction_ultra.py` (contact maintainers for this version).
- **Redis Connection Issues**:
  - Verify Redis is running:
    ```bash
    redis-cli ping
    ```
  - Ensure `localhost:6379` is accessible.
- **Data Loading Errors**:
  - Check `data/` directory for CSV files.
  - Run `generate_payment_data.py` to recreate data.
- **Model Training Failures**:
  - Ensure sufficient memory (≥16GB RAM recommended).
  - Reduce `batch_size` or `sequence_length` in `config.yaml`.
- **Dashboard Not Loading**:
  - Verify port `8050` is not blocked.
  - Check `payment_prediction_ultra.log` for errors.

