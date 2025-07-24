import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, LayerNormalization, MultiHeadAttention, Add, Attention, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, f1_score, roc_curve
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import fastapi
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis
import json
import logging
import yaml
import os
from datetime import datetime, timedelta
import uuid
import requests
import time
import hashlib
from typing import Dict, List, Optional, Tuple, Any
import pickle
import asyncio
import threading
import queue
import warnings
import plotly.io as pio
from sklearn.utils.class_weight import compute_class_weight
import psutil
import random
import string
import shutil
from collections import defaultdict

# Custom JSON encoder for NumPy types
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - [File: %(filename)s, Line: %(lineno)d, Process: %(process)d]',
    handlers=[logging.FileHandler('payment_prediction_ultra.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Custom Metrics System
class CustomMetrics:
    def __init__(self, log_dir: str):
        self.metrics = {
            'request_counter': 0,
            'prediction_latency': [],
            'memory_usage': []
        }
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics_file = os.path.join(self.log_dir, 'metrics.log')

    def inc_request(self):
        self.metrics['request_counter'] += 1
        self._log_metrics()

    def observe_latency(self, latency: float):
        self.metrics['prediction_latency'].append(latency)
        self._log_metrics()

    def inc_memory(self, memory: float):
        self.metrics['memory_usage'].append(memory)
        self._log_metrics()

    def _log_metrics(self):
        try:
            with open(self.metrics_file, 'a') as f:
                f.write(f"{datetime.now().isoformat()} - Metrics: {self.metrics}\n")
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=FutureWarning)

# Initialize custom metrics
metrics = CustomMetrics(log_dir='data')

# Section 1: Configuration
class Config:
    def __init__(self, config_file: str = 'config.yaml'):
        self.data_dir = os.path.abspath('data')
        self.transaction_file = os.path.join(self.data_dir, 'transactions.csv')
        self.user_file = os.path.join(self.data_dir, 'users.csv')
        self.merchant_file = os.path.join(self.data_dir, 'merchants.csv')
        self.processor_file = os.path.join(self.data_dir, 'processor_metrics.csv')
        self.model_dir = os.path.join(self.data_dir, 'models')
        self.lstm_model_file = os.path.join(self.model_dir, 'lstm_model.h5')
        self.transformer_model_file = os.path.join(self.model_dir, 'transformer_model.h5')
        self.xgb_model_file = os.path.join(self.model_dir, 'xgb_model.pkl')
        self.ensemble_model_file = os.path.join(self.model_dir, 'ensemble_model.pkl')
        self.preprocessor_file = os.path.join(self.model_dir, 'preprocessor.pkl')
        self.sequence_length = 15
        self.batch_size = 256
        self.epochs = 150
        self.learning_rate = 0.0003
        self.max_retries = 5
        self.retry_delay = 2.0
        self.cache_ttl = 7200  # 2 hours
        self.rate_limit = 100  # Requests per minute
        self.class_weights = {0: 1.0, 1: 6.0, 2: 12.0}  # Success, Failed, Chargeback
        self.payment_methods = ['Card', 'Bank_Transfer', 'USSD', 'Mobile_Money']
        self.processor_pathways = [
            'Card_Visa', 'Card_Mastercard', 'Card_Verve',
            'Bank_Transfer_GTBank', 'Bank_Transfer_Zenith', 'Bank_Transfer_FirstBank',
            'USSD_Airtel', 'USSD_MTN', 'Mobile_Money_OPay', 'Mobile_Money_Paga'
        ]
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        self.load_config(config_file)

    def load_config(self, config_file: str) -> None:
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    if config:
                        self.data_dir = config.get('data_dir', self.data_dir)
                        self.transaction_file = config.get('transaction_file', self.transaction_file)
                        self.user_file = config.get('user_file', self.user_file)
                        self.merchant_file = config.get('merchant_file', self.merchant_file)
                        self.processor_file = config.get('processor_file', self.processor_file)
                        self.model_dir = config.get('model_dir', self.model_dir)
                        self.lstm_model_file = config.get('lstm_model_file', self.lstm_model_file)
                        self.transformer_model_file = config.get('transformer_model_file', self.transformer_model_file)
                        self.xgb_model_file = config.get('xgb_model_file', self.xgb_model_file)
                        self.ensemble_model_file = config.get('ensemble_model_file', self.ensemble_model_file)
                        self.preprocessor_file = config.get('preprocessor_file', self.preprocessor_file)
                        self.sequence_length = config.get('sequence_length', self.sequence_length)
                        self.batch_size = config.get('batch_size', self.batch_size)
                        self.epochs = config.get('epochs', self.epochs)
                        self.learning_rate = config.get('learning_rate', self.learning_rate)
                        self.cache_ttl = config.get('cache_ttl', self.cache_ttl)
                        logger.info("Configuration loaded successfully")
            else:
                logger.warning(f"Config file {config_file} not found, using default settings")
                self._save_default_config(config_file)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def _save_default_config(self, config_file: str) -> None:
        default_config = {
            'data_dir': self.data_dir,
            'transaction_file': self.transaction_file,
            'user_file': self.user_file,
            'merchant_file': self.merchant_file,
            'processor_file': self.processor_file,
            'model_dir': self.model_dir,
            'lstm_model_file': self.lstm_model_file,
            'transformer_model_file': self.transformer_model_file,
            'xgb_model_file': self.xgb_model_file,
            'ensemble_model_file': self.ensemble_model_file,
            'preprocessor_file': self.preprocessor_file,
            'sequence_length': self.sequence_length,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'cache_ttl': self.cache_ttl,
            'rate_limit': self.rate_limit
        }
        with open(config_file, 'w') as f:
            yaml.safe_dump(default_config, f)
        logger.info(f"Created default config file: {config_file}")

# Section 2: Data Preparation
class DataPreparation:
    def __init__(self, config: Config):
        self.config = config
        self.transactions = None
        self.users = None
        self.merchants = None
        self.processors = None
        self.merged_data_cache = None
        self.sequence_cache = {}
        self.augmentation_cache = {}

    def load_data(self) -> None:
        try:
            self.transactions = pd.read_csv(self.config.transaction_file)
            self.users = pd.read_csv(self.config.user_file)
            self.merchants = pd.read_csv(self.config.merchant_file)
            self.processors = pd.read_csv(self.config.processor_file)
            logger.info(f"Loaded {len(self.transactions)} transactions, {len(self.users)} users, "
                       f"{len(self.merchants)} merchants, {len(self.processors)} processor metrics")
            self._generate_dataflow_diagram()
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _generate_dataflow_diagram(self) -> None:
        try:
            fig = go.Figure(data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=["Transactions", "Users", "Merchants", "Processors", "Merged Data", "Augmented Data"],
                        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
                    ),
                    link=dict(
                        source=[0, 1, 2, 3, 0, 4],
                        target=[4, 4, 4, 4, 5, 5],
                        value=[len(self.transactions), len(self.users), len(self.merchants), len(self.processors),
                               len(self.transactions), len(self.transactions)]
                    )
                )
            ])
            fig.update_layout(title_text="Data Flow Diagram", font_size=12, title_x=0.5)
            fig.write_image(os.path.join(self.config.data_dir, 'dataflow_diagram.png'), scale=2)
            fig.write_json(os.path.join(self.config.data_dir, 'dataflow_diagram.json'))
            logger.info(f"Generated data flow diagram at {os.path.join(self.config.data_dir, 'dataflow_diagram.png')}")
        except Exception as e:
            logger.error(f"Error generating data flow diagram: {e}")

    def validate_data(self) -> None:
        try:
            required_cols = [
                'transaction_id', 'timestamp', 'amount', 'currency', 'payment_method',
                'processor_pathway', 'status', 'failure_reason', 'chargeback_reason',
                'customer_id', 'merchant_id', 'merchant_type', 'fraud_risk_score',
                'device_type', 'network_type', 'location', 'transaction_type',
                'hour_of_day', 'day_of_week', 'is_weekend', 'time_since_last_transaction'
            ]
            missing_cols = [col for col in required_cols if col not in self.transactions.columns]
            if missing_cols:
                logger.warning(f"Missing transaction columns: {missing_cols}, filling with defaults")
                for col in missing_cols:
                    if col in ['amount', 'fraud_risk_score', 'time_since_last_transaction']:
                        self.transactions[col] = 0.0
                    elif col in ['hour_of_day', 'day_of_week', 'is_weekend']:
                        self.transactions[col] = 0
                    else:
                        self.transactions[col] = 'Unknown'

            self.transactions['timestamp'] = pd.to_datetime(self.transactions['timestamp'], errors='coerce')
            self.transactions['amount'] = self.transactions['amount'].astype(float)
            self.transactions['fraud_risk_score'] = np.clip(self.transactions['fraud_risk_score'].astype(float), 0, 1)
            self.transactions['time_since_last_transaction'] = self.transactions['time_since_last_transaction'].astype(float)
            self.transactions['hour_of_day'] = self.transactions['hour_of_day'].astype(int)
            self.transactions['day_of_week'] = self.transactions['day_of_week'].astype(int)
            self.transactions['is_weekend'] = self.transactions['is_weekend'].astype(int)

            # Replace invalid status values with 'Success'
            self.transactions['status'] = self.transactions['status'].replace(
                {val: 'Success' for val in self.transactions['status']
                 if val not in ['Success', 'Failed', 'Chargeback']}
            )
            # Replace invalid payment_method values with 'Card'
            self.transactions['payment_method'] = self.transactions['payment_method'].replace(
                {val: 'Card' for val in self.transactions['payment_method']
                 if val not in self.config.payment_methods}
            )
            # Replace invalid processor_pathway values with 'Card_Verve'
            self.transactions['processor_pathway'] = self.transactions['processor_pathway'].replace(
                {val: 'Card_Verve' for val in self.transactions['processor_pathway']
                 if val not in self.config.processor_pathways}
            )

            self.transactions.to_csv(self.config.transaction_file, index=False)
            logger.info("Validated and saved transaction data")
            self._generate_validation_diagram()
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            raise

    def _generate_dataflow_diagram(self) -> None:
        try:
            fig = go.Figure(data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=["Transactions", "Users", "Merchants", "Processors", "Merged Data", "Augmented Data"],
                        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
                    ),
                    link=dict(
                        source=[0, 1, 2, 3, 0, 4],
                        target=[4, 4, 4, 4, 5, 5],
                        value=[len(self.transactions), len(self.users), len(self.merchants), len(self.processors),
                               len(self.transactions), len(self.transactions)]
                    )
                )
            ])
            fig.update_layout(title_text="Data Flow Diagram", font_size=12, title_x=0.5)
            # Export JSON
            fig.write_json(os.path.join(self.config.data_dir, 'dataflow_diagram.json'))
            # Try to export image, handle Kaleido failure gracefully
            try:
                fig.write_image(os.path.join(self.config.data_dir, 'dataflow_diagram.png'), scale=2)
                logger.info(f"Generated data flow diagram at {os.path.join(self.config.data_dir, 'dataflow_diagram.png')}")
            except Exception as e:
                logger.warning(f"Failed to export data flow diagram image due to Kaleido issue: {e}. JSON exported instead.")
        except Exception as e:
            logger.error(f"Error generating data flow diagram: {e}")

    def _generate_validation_diagram(self) -> None:
        try:
            fig = go.Figure(data=[
                go.Bar(
                    x=['Total Rows', 'Missing Values', 'Invalid Status', 'Invalid Payment Method'],
                    y=[
                        len(self.transactions),
                        self.transactions.isnull().sum().sum(),
                        len(self.transactions[~self.transactions['status'].isin(['Success', 'Failed', 'Chargeback'])]),
                        len(self.transactions[~self.transactions['payment_method'].isin(self.config.payment_methods)])
                    ],
                    marker_color='#1f77b4'
                )
            ])
            fig.update_layout(
                title='Data Validation Metrics',
                xaxis_title='Metric',
                yaxis_title='Count',
                title_x=0.5
            )
            # Export JSON
            fig.write_json(os.path.join(self.config.data_dir, 'validation_diagram.json'))
            # Try to export image, handle Kaleido failure gracefully
            try:
                fig.write_image(os.path.join(self.config.data_dir, 'validation_diagram.png'), scale=2)
                logger.info(f"Generated validation metrics diagram at {os.path.join(self.config.data_dir, 'validation_diagram.png')}")
            except Exception as e:
                logger.warning(f"Failed to export validation diagram image due to Kaleido issue: {e}. JSON exported instead.")
        except Exception as e:
            logger.error(f"Error generating validation diagram: {e}")

    def _generate_merge_diagram(self) -> None:
        try:
            fig = go.Figure(data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=["Transactions", "Users", "Merchants", "Processors", "Merged Data", "Feature Engineered"],
                        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
                    ),
                    link=dict(
                        source=[0, 1, 2, 3, 0, 4],
                        target=[4, 4, 4, 4, 5, 5],
                        value=[len(self.transactions), len(self.users), len(self.merchants), len(self.processors),
                               len(self.transactions), len(self.transactions)]
                    )
                )
            ])
            fig.update_layout(title_text="Data Merge and Feature Engineering Flow", font_size=12, title_x=0.5)
            fig.write_json(os.path.join(self.config.data_dir, 'merge_diagram.json'))
            try:
                fig.write_image(os.path.join(self.config.data_dir, 'merge_diagram.png'), scale=2)
                logger.info(f"Generated merge flow diagram at {os.path.join(self.config.data_dir, 'merge_diagram.png')}")
            except Exception as e:
                logger.warning(f"Failed to export merge diagram image due to Kaleido issue: {e}. JSON exported instead.")
        except Exception as e:
            logger.error(f"Error generating merge diagram: {e}")

    def _generate_augmentation_diagram(self, original_count: int, augmented_count: int) -> None:
        try:
            fig = go.Figure(data=[
                go.Bar(
                    x=['Original Sequences', 'Augmented Sequences'],
                    y=[original_count, augmented_count],
                    marker_color=['#1f77b4', '#ff7f0e']
                )
            ])
            fig.update_layout(
                title='Sequence Augmentation',
                xaxis_title='Sequence Type',
                yaxis_title='Count',
                title_x=0.5
            )
            fig.write_json(os.path.join(self.config.data_dir, 'augmentation_diagram.json'))
            try:
                fig.write_image(os.path.join(self.config.data_dir, 'augmentation_diagram.png'), scale=2)
                logger.info(f"Generated augmentation diagram at {os.path.join(self.config.data_dir, 'augmentation_diagram.png')}")
            except Exception as e:
                logger.warning(f"Failed to export augmentation diagram image due to Kaleido issue: {e}. JSON exported instead.")
        except Exception as e:
            logger.error(f"Error generating augmentation diagram: {e}")

    def _generate_sequence_diagram(self) -> None:
        try:
            fig = go.Figure(data=[
                go.Scatter(
                    x=np.arange(self.config.sequence_length),
                    y=np.ones(self.config.sequence_length),
                    mode='lines+markers',
                    name='Transaction Sequence',
                    marker=dict(size=12, color='#1f77b4'),
                    line=dict(width=3)
                )
            ])
            fig.update_layout(
                title='Transaction Sequence Structure',
                xaxis_title='Sequence Position',
                yaxis_title='Transaction',
                showlegend=True,
                title_x=0.5
            )
            fig.write_json(os.path.join(self.config.data_dir, 'sequence_diagram.json'))
            try:
                fig.write_image(os.path.join(self.config.data_dir, 'sequence_diagram.png'), scale=2)
                logger.info(f"Generated sequence structure diagram at {os.path.join(self.config.data_dir, 'sequence_diagram.png')}")
            except Exception as e:
                logger.warning(f"Failed to export sequence diagram image due to Kaleido issue: {e}. JSON exported instead.")
        except Exception as e:
            logger.error(f"Error generating sequence diagram: {e}")

    def merge_data(self) -> pd.DataFrame:
        try:
            if self.transactions is None or self.users is None or self.merchants is None or self.processors is None:
                logger.error("One or more data sources are not loaded")
                raise ValueError("Data sources not loaded")

            # Merge transactions with users
            merged_df = pd.merge(
                self.transactions,
                self.users[['customer_id', 'user_risk_score', 'average_transaction_frequency']],
                on='customer_id',
                how='left'
            )
            # Merge with merchants
            merged_df = pd.merge(
                merged_df,
                self.merchants[['merchant_id', 'merchant_risk_score']],
                on='merchant_id',
                how='left'
            )
            # Merge with processors
            merged_df = pd.merge(
                merged_df,
                self.processors[['processor_pathway', 'cost', 'latency', 'success_rate', 'network_stability']],
                on='processor_pathway',
                how='left'
            )

            # Fill missing values
            merged_df.fillna({
                'user_risk_score': 0.1,
                'average_transaction_frequency': 0.1,
                'merchant_risk_score': 0.1,
                'cost': 0.03,
                'latency': 200,
                'success_rate': 0.95,
                'network_stability': 0.9
            }, inplace=True)

            # Feature engineering
            merged_df['amount_log'] = np.log1p(merged_df['amount'])
            merged_df['risk_composite'] = 0.6 * merged_df['user_risk_score'] + 0.4 * merged_df['merchant_risk_score']

            # Cache the merged data
            self.merged_data_cache = merged_df
            merged_df.to_csv(os.path.join(self.config.data_dir, 'merged_data.csv'), index=False)
            logger.info("Merged data created and cached")
            self._generate_merge_diagram()
            return merged_df
        except Exception as e:
            logger.error(f"Error merging data: {e}")
            raise

    def create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        try:
            if self.merged_data_cache is None:
                logger.warning("Merged data cache is empty, running merge_data")
                self.merge_data()

            # Sort by timestamp and customer_id
            df = self.merged_data_cache.sort_values(['customer_id', 'timestamp'])
            feature_columns = [
                col for col in df.columns
                if col not in ['transaction_id', 'customer_id', 'merchant_id', 'timestamp', 'status',
                              'failure_reason', 'chargeback_reason']
            ]
            sequences = []
            labels = []
            status_map = {'Success': 0, 'Failed': 1, 'Chargeback': 2}

            for customer_id in df['customer_id'].unique():
                customer_data = df[df['customer_id'] == customer_id]
                if len(customer_data) >= self.config.sequence_length:
                    for i in range(len(customer_data) - self.config.sequence_length + 1):
                        seq = customer_data.iloc[i:i + self.config.sequence_length][feature_columns].values
                        label = status_map.get(customer_data.iloc[i + self.config.sequence_length - 1]['status'], 0)
                        sequences.append(seq)
                        labels.append(label)

            sequences = np.array(sequences)
            labels = np.array(labels)
            logger.info(f"Created {len(sequences)} sequences with shape {sequences.shape}")
            self.augment_data(len(sequences))
            return sequences, labels
        except Exception as e:
            logger.error(f"Error creating sequences: {e}")
            return np.array([]), np.array([])

    def augment_data(self, original_count: int) -> None:
        try:
            if self.merged_data_cache is None:
                logger.warning("Merged data cache is empty, cannot augment data")
                return

            # Simple data augmentation: add noise to numerical columns
            df = self.merged_data_cache.copy()
            numerical_cols = ['amount', 'fraud_risk_score', 'time_since_last_transaction',
                            'user_risk_score', 'average_transaction_frequency',
                            'merchant_risk_score', 'cost', 'latency', 'success_rate',
                            'network_stability', 'amount_log', 'risk_composite']

            for col in numerical_cols:
                if col in df.columns:
                    noise = np.random.normal(0, 0.1 * df[col].std(), size=len(df))
                    df[col] = df[col] + noise
                    if col in ['fraud_risk_score', 'user_risk_score', 'merchant_risk_score',
                              'success_rate', 'network_stability']:
                        df[col] = np.clip(df[col], 0, 1)
                    elif col in ['amount', 'time_since_last_transaction', 'cost', 'latency', 'amount_log']:
                        df[col] = np.clip(df[col], 0, None)

            # Append augmented data to cache
            self.merged_data_cache = pd.concat([self.merged_data_cache, df], ignore_index=True)
            self.merged_data_cache.to_csv(os.path.join(self.config.data_dir, 'augmented_data.csv'), index=False)
            logger.info(f"Augmented data, new size: {len(self.merged_data_cache)}")
            self._generate_augmentation_diagram(original_count, len(self.merged_data_cache))
        except Exception as e:
            logger.error(f"Error augmenting data: {e}")

# Section 3: Preprocessing
class DataPreprocessor:
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = None
        self.categorical_cols = [
            'currency', 'payment_method', 'processor_pathway', 'merchant_type',
            'device_type', 'network_type', 'location', 'transaction_type',
            'failure_reason', 'chargeback_reason'
        ]
        self.numerical_cols = [
            'amount', 'fraud_risk_score', 'hour_of_day', 'day_of_week',
            'is_weekend', 'time_since_last_transaction', 'user_risk_score',
            'average_transaction_frequency', 'merchant_risk_score',
            'cost', 'latency', 'success_rate', 'network_stability', 'amount_log', 'risk_composite'
        ]
        self.preprocessed_cache = {}
        self.feature_names = None

    def build_preprocessor(self, data: pd.DataFrame) -> Pipeline:
        try:
            available_cat_cols = [col for col in self.categorical_cols if col in data.columns]
            available_num_cols = [col for col in self.numerical_cols if col in data.columns]
            transformers = [
                ('num', RobustScaler(), available_num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), available_cat_cols)
            ]
            preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
            self.preprocessor = Pipeline([('preprocessor', preprocessor)])
            self.feature_names = (
                available_num_cols +
                self.preprocessor.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(available_cat_cols).tolist()
            )
            logger.info("Preprocessing pipeline built")
            self._generate_preprocessing_diagram()
            return self.preprocessor
        except Exception as e:
            logger.error(f"Error building preprocessor: {e}")
            raise

    def _generate_preprocessing_diagram(self) -> None:
        try:
            fig = go.Figure(data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=["Raw Data", "Numerical Features", "Categorical Features", "Feature Engineered", "Preprocessed Data"],
                        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
                    ),
                    link=dict(
                        source=[0, 0, 0, 1, 2, 3],
                        target=[1, 2, 3, 4, 4, 4],
                        value=[len(self.numerical_cols), len(self.categorical_cols), len(self.numerical_cols),
                               len(self.numerical_cols), len(self.categorical_cols), len(self.numerical_cols)]
                    )
                )
            ])
            fig.update_layout(title_text="Preprocessing and Feature Engineering Pipeline", font_size=12, title_x=0.5)
            fig.write_image(os.path.join(self.config.data_dir, 'preprocessing_diagram.png'), scale=2)
            fig.write_json(os.path.join(self.config.data_dir, 'preprocessing_diagram.json'))
            logger.info(f"Generated preprocessing pipeline diagram at {os.path.join(self.config.data_dir, 'preprocessing_diagram.png')}")
        except Exception as e:
            logger.error(f"Error generating preprocessing diagram: {e}")

    def preprocess_data(self, data: pd.DataFrame, training: bool = True, cache_key: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        try:
            if cache_key and cache_key in self.preprocessed_cache:
                logger.info(f"Using cached preprocessed data for key: {cache_key}")
                return self.preprocessed_cache[cache_key]

            X = data.drop(['transaction_id', 'customer_id', 'merchant_id', 'timestamp', 'status'], axis=1, errors='ignore').copy()
            if training:
                y = X['status'].map({'Success': 0, 'Failed': 1, 'Chargeback': 2})
                X = X.drop('status', axis=1, errors='ignore')
                self.preprocessor.fit(X)
                X_transformed = self.preprocessor.transform(X)
                if cache_key:
                    self.preprocessed_cache[cache_key] = (X_transformed, y)
                return X_transformed, y
            else:
                X_transformed = self.preprocessor.transform(X)
                if cache_key:
                    self.preprocessed_cache[cache_key] = (X_transformed, None)
                return X_transformed, None
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise

    def save_preprocessor(self) -> None:
        try:
            with open(self.config.preprocessor_file, 'wb') as f:
                pickle.dump(self.preprocessor, f)
            logger.info("Preprocessor saved")
        except Exception as e:
            logger.error(f"Error saving preprocessor: {e}")
            raise

    def load_preprocessor(self) -> None:
        try:
            with open(self.config.preprocessor_file, 'rb') as f:
                self.preprocessor = pickle.load(f)
            logger.info("Preprocessor loaded")
        except FileNotFoundError:
            logger.warning("Preprocessor file not found")
            self.preprocessor = None
        except Exception as e:
            logger.error(f"Error loading preprocessor: {e}")
            raise

# Section 4: Transformer Layer
class DynamicTransformerLayer(tf.keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int, dff: int, rate: float = 0.1):
        super(DynamicTransformerLayer, self).__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = Sequential([
            Dense(dff, activation='gelu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.attention_weights = None

    def call(self, x, training: bool):
        attn_output, self.attention_weights = self.mha(x, x, x, return_attention_scores=True)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Section 5: Model Building
class PaymentModel:
    def __init__(self, config: Config):
        self.config = config
        self.lstm_model = None
        self.transformer_model = None
        self.xgb_model = None
        self.ensemble_model = None
        self.preprocessor = None

    def build_lstm_model(self, input_shape: Tuple[int, int]) -> Model:
        inputs = Input(shape=input_shape)
        x = LSTM(512, return_sequences=True, return_state=False)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = LSTM(256, return_sequences=True)(x)
        x = Attention()([x, x])
        x = LSTM(128)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(64, activation='gelu')(x)
        outputs = Dense(3, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate, clipnorm=1.0),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        logger.info("LSTM model built")
        self._generate_lstm_architecture_diagram()
        return model

    def _generate_lstm_architecture_diagram(self) -> None:
        try:
            fig = go.Figure(data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=["Input", "LSTM 512", "BatchNorm", "Dropout", "LSTM 256", "Attention",
                               "LSTM 128", "BatchNorm", "Dropout", "Dense 64", "Output"],
                        color=["#1f77b4"] * 11
                    ),
                    link=dict(
                        source=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        target=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        value=[1] * 10
                    )
                )
            ])
            fig.update_layout(title_text="LSTM Model Architecture", font_size=12, title_x=0.5)
            fig.write_image(os.path.join(self.config.data_dir, 'lstm_architecture.png'), scale=2)
            fig.write_json(os.path.join(self.config.data_dir, 'lstm_architecture.json'))
            logger.info(f"Generated LSTM architecture diagram at {os.path.join(self.config.data_dir, 'lstm_architecture.png')}")
        except Exception as e:
            logger.error(f"Error generating LSTM architecture diagram: {e}")

    def build_transformer_model(self, input_shape: Tuple[int, int]) -> Model:
        inputs = Input(shape=input_shape)
        x = inputs
        for _ in range(4):
            x = DynamicTransformerLayer(d_model=input_shape[-1], num_heads=12, dff=1024, rate=0.2)(x, training=True)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = Dense(128, activation='gelu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        outputs = Dense(3, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate, clipnorm=1.0),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        logger.info("Transformer model built")
        self._generate_transformer_architecture_diagram()
        return model

    def _generate_transformer_architecture_diagram(self) -> None:
        try:
            fig = go.Figure(data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=["Input", "Transformer Layer 1", "Transformer Layer 2", "Transformer Layer 3",
                               "Transformer Layer 4", "Global Pooling", "Dense 128", "BatchNorm", "Dropout", "Output"],
                        color=["#1f77b4"] * 10
                    ),
                    link=dict(
                        source=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                        target=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                        value=[1] * 9
                    )
                )
            ])
            fig.update_layout(title_text="Transformer Model Architecture", font_size=12, title_x=0.5)
            fig.write_image(os.path.join(self.config.data_dir, 'transformer_architecture.png'), scale=2)
            fig.write_json(os.path.join(self.config.data_dir, 'transformer_architecture.json'))
            logger.info(f"Generated Transformer architecture diagram at {os.path.join(self.config.data_dir, 'transformer_architecture.png')}")
        except Exception as e:
            logger.error(f"Error generating Transformer architecture diagram: {e}")

    def build_xgb_model(self, trial: optuna.Trial) -> xgb.XGBClassifier:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'objective': 'multi:softmax',
            'num_class': 3,
            'random_state': 42,
            'n_jobs': -1
        }
        model = xgb.XGBClassifier(**params)
        logger.info("XGBoost model built with Optuna parameters")
        return model

    def optimize_xgb(self, X: np.ndarray, y: np.ndarray) -> None:
        def objective(trial):
            model = self.build_xgb_model(trial)
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return f1_score(y_val, y_pred, average='weighted')

        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=50)
            self.xgb_model = self.build_xgb_model(study.best_trial)
            logger.info(f"Optimized XGBoost with best params: {study.best_params}")
            self._generate_optuna_diagram(study)
        except Exception as e:
            logger.error(f"Error optimizing XGBoost: {e}")
            raise

    def _generate_optuna_diagram(self, study: optuna.Study) -> None:
        try:
            fig = optuna.visualization.plot_optimization_history(study)
            fig.update_layout(title='XGBoost Optimization History', title_x=0.5)
            fig.write_image(os.path.join(self.config.data_dir, 'xgb_optimization_history.png'), scale=2)
            fig.write_json(os.path.join(self.config.data_dir, 'xgb_optimization_history.json'))
            logger.info(f"Generated XGBoost optimization history diagram at {os.path.join(self.config.data_dir, 'xgb_optimization_history.png')}")
        except Exception as e:
            logger.error(f"Error generating Optuna diagram: {e}")

    def build_ensemble_model(self, X_meta: np.ndarray, y: np.ndarray) -> LogisticRegression:
        model = LogisticRegression(multi_class='multinomial', random_state=42, max_iter=1000)
        model.fit(X_meta, y)
        logger.info("Ensemble meta-learner built")
        return model

    def train_lstm(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            self.lstm_model = self.build_lstm_model((X.shape[1], X.shape[2]))
            callbacks = [
                EarlyStopping(patience=15, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5),
                ModelCheckpoint(self.config.lstm_model_file, save_best_only=True),
                TensorBoard(log_dir=os.path.join(self.config.data_dir, 'logs/lstm'))
            ]
            self.lstm_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                class_weight=self.config.class_weights,
                callbacks=callbacks
            )
            logger.info("LSTM model trained")
            self._generate_training_curve('LSTM', self.lstm_model.history.history)
        except Exception as e:
            logger.error(f"Error training LSTM: {e}")
            raise

    def train_transformer(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            self.transformer_model = self.build_transformer_model((X.shape[1], X.shape[2]))
            callbacks = [
                EarlyStopping(patience=15, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5),
                ModelCheckpoint(self.config.transformer_model_file, save_best_only=True),
                TensorBoard(log_dir=os.path.join(self.config.data_dir, 'logs/transformer'))
            ]
            self.transformer_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                class_weight=self.config.class_weights,
                callbacks=callbacks
            )
            logger.info("Transformer model trained")
            self._generate_training_curve('Transformer', self.transformer_model.history.history)
        except Exception as e:
            logger.error(f"Error training Transformer: {e}")
            raise

    def train_xgb(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            self.optimize_xgb(X, y)
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.1, random_state=42)
            self.xgb_model.fit(X_train, y_train)
            logger.info("XGBoost model trained")
            self._generate_xgb_feature_importance(X)
        except Exception as e:
            logger.error(f"Error training XGBoost: {e}")
            raise

    def train_ensemble(self, X_seq: np.ndarray, X_flat: np.ndarray, y: np.ndarray) -> None:
        try:
            X_train_seq, X_val_seq, X_train_flat, X_val_flat, y_train, y_val = train_test_split(
                X_seq, X_flat, y, test_size=0.2, random_state=42
            )
            lstm_preds = self.lstm_model.predict(X_val_seq)
            transformer_preds = self.transformer_model.predict(X_val_seq)
            xgb_preds = self.xgb_model.predict_proba(X_val_flat)
            X_meta = np.hstack([lstm_preds, transformer_preds, xgb_preds])
            self.ensemble_model = self.build_ensemble_model(X_meta, y_val)
            logger.info("Ensemble meta-learner trained")
            self._generate_ensemble_diagram()
        except Exception as e:
            logger.error(f"Error training ensemble: {e}")
            raise

    def _generate_training_curve(self, model_name: str, history: Dict) -> None:
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history['loss'], mode='lines', name='Training Loss', line=dict(color='#1f77b4')))
            fig.add_trace(go.Scatter(y=history['val_loss'], mode='lines', name='Validation Loss', line=dict(color='#ff7f0e')))
            fig.add_trace(go.Scatter(y=history['accuracy'], mode='lines', name='Training Accuracy', line=dict(color='#2ca02c')))
            fig.add_trace(go.Scatter(y=history['val_accuracy'], mode='lines', name='Validation Accuracy', line=dict(color='#d62728')))
            fig.update_layout(
                title=f'{model_name} Training Metrics',
                xaxis_title='Epoch',
                yaxis_title='Metric Value',
                showlegend=True,
                title_x=0.5
            )
            fig.write_image(os.path.join(self.config.data_dir, f'{model_name.lower()}_training_curve.png'), scale=2)
            fig.write_json(os.path.join(self.config.data_dir, f'{model_name.lower()}_training_curve.json'))
            logger.info(f"Generated {model_name} training curve at {os.path.join(self.config.data_dir, f'{model_name.lower()}_training_curve.png')}")
        except Exception as e:
            logger.error(f"Error generating {model_name} training curve: {e}")

    def _generate_xgb_feature_importance(self, X: np.ndarray) -> None:
        try:
            importances = self.xgb_model.feature_importances_
            feature_names = self.preprocessor.feature_names if self.preprocessor and self.preprocessor.feature_names else [f"feature_{i}" for i in range(X.shape[1])]
            sorted_idx = np.argsort(importances)[::-1][:10]
            fig = go.Figure(data=[
                go.Bar(x=np.array(feature_names)[sorted_idx], y=importances[sorted_idx], marker_color='#1f77b4')
            ])
            fig.update_layout(
                title='Top 10 XGBoost Feature Importance',
                xaxis_title='Feature',
                yaxis_title='Importance',
                xaxis_tickangle=45,
                title_x=0.5
            )
            fig.write_image(os.path.join(self.config.data_dir, 'xgb_feature_importance.png'), scale=2)
            fig.write_json(os.path.join(self.config.data_dir, 'xgb_feature_importance.json'))
            logger.info(f"Generated XGBoost feature importance diagram at {os.path.join(self.config.data_dir, 'xgb_feature_importance.png')}")
        except Exception as e:
            logger.error(f"Error generating XGBoost feature importance: {e}")

    def _generate_ensemble_diagram(self) -> None:
        try:
            fig = go.Figure(data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=["LSTM", "Transformer", "XGBoost", "Meta-Learner", "Final Prediction"],
                        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
                    ),
                    link=dict(
                        source=[0, 1, 2, 3],
                        target=[3, 3, 3, 4],
                        value=[1] * 4
                    )
                )
            ])
            fig.update_layout(title_text="Ensemble Model Architecture", font_size=12, title_x=0.5)
            fig.write_image(os.path.join(self.config.data_dir, 'ensemble_architecture.png'), scale=2)
            fig.write_json(os.path.join(self.config.data_dir, 'ensemble_architecture.json'))
            logger.info(f"Generated ensemble architecture diagram at {os.path.join(self.config.data_dir, 'ensemble_architecture.png')}")
        except Exception as e:
            logger.error(f"Error generating ensemble diagram: {e}")

    def evaluate_model(self, model, X: np.ndarray, y: np.ndarray, model_name: str) -> None:
        try:
            y_pred = model.predict(X)
            if model_name != 'XGBoost' and model_name != 'Ensemble':
                y_pred = np.argmax(y_pred, axis=1)
            report = classification_report(y, y_pred, target_names=['Success', 'Failed', 'Chargeback'])
            auc = roc_auc_score(y, model.predict(X), multi_class='ovr')
            logger.info(f"{model_name} - Classification Report:\n{report}")
            logger.info(f"{model_name} - ROC AUC: {auc:.4f}")

            # Confusion Matrix
            cm = confusion_matrix(y, y_pred)
            fig = go.Figure(data=[
                go.Heatmap(
                    z=cm,
                    x=['Success', 'Failed', 'Chargeback'],
                    y=['Success', 'Failed', 'Chargeback'],
                    colorscale='Blues',
                    text=cm,
                    texttemplate="%{text}",
                    showscale=True
                )
            ])
            fig.update_layout(
                title=f'{model_name} Confusion Matrix',
                xaxis_title='Predicted',
                yaxis_title='Actual',
                title_x=0.5
            )
            fig.write_image(os.path.join(self.config.data_dir, f'{model_name.lower()}_confusion_matrix.png'), scale=2)
            fig.write_json(os.path.join(self.config.data_dir, f'{model_name.lower()}_confusion_matrix.json'))
            logger.info(f"Generated {model_name} confusion matrix at {os.path.join(self.config.data_dir, f'{model_name.lower()}_confusion_matrix.png')}")

            # Precision-Recall Curve
            y_scores = model.predict(X)
            if model_name == 'XGBoost':
                y_scores = y_scores
            else:
                y_scores = y_scores
            pr_fig = go.Figure()
            for i, label in enumerate(['Success', 'Failed', 'Chargeback']):
                precisions, recalls, _ = precision_recall_curve(y == i, y_scores[:, i])
                pr_fig.add_trace(go.Scatter(x=recalls, y=precisions, mode='lines', name=f'PR {label}'))
            pr_fig.update_layout(
                title=f'{model_name} Precision-Recall Curves',
                xaxis_title='Recall',
                yaxis_title='Precision',
                title_x=0.5
            )
            pr_fig.write_image(os.path.join(self.config.data_dir, f'{model_name.lower()}_pr_curve.png'), scale=2)
            pr_fig.write_json(os.path.join(self.config.data_dir, f'{model_name.lower()}_pr_curve.json'))
            logger.info(f"Generated {model_name} precision-recall curve at {os.path.join(self.config.data_dir, f'{model_name.lower()}_pr_curve.png')}")
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            raise

    def ensemble_predict(self, X_seq: np.ndarray, X_flat: np.ndarray) -> np.ndarray:
        try:
            lstm_preds = self.lstm_model.predict(X_seq)
            transformer_preds = self.transformer_model.predict(X_seq)
            xgb_preds = self.xgb_model.predict_proba(X_flat)
            X_meta = np.hstack([lstm_preds, transformer_preds, xgb_preds])
            return self.ensemble_model.predict(X_meta)
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            raise

    def save_models(self) -> None:
        try:
            if self.lstm_model:
                self.lstm_model.save(self.config.lstm_model_file)
            if self.transformer_model:
                self.transformer_model.save(self.config.transformer_model_file)
            if self.xgb_model:
                with open(self.config.xgb_model_file, 'wb') as f:
                    pickle.dump(self.xgb_model, f)
            if self.ensemble_model:
                with open(self.config.ensemble_model_file, 'wb') as f:
                    pickle.dump(self.ensemble_model, f)
            logger.info("Models saved")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise

    def load_models(self) -> None:
        try:
            if os.path.exists(self.config.lstm_model_file):
                self.lstm_model = tf.keras.models.load_model(self.config.lstm_model_file)
            if os.path.exists(self.config.transformer_model_file):
                self.transformer_model = tf.keras.models.load_model(self.config.transformer_model_file,
                                                                   custom_objects={'DynamicTransformerLayer': DynamicTransformerLayer})
            if os.path.exists(self.config.xgb_model_file):
                with open(self.config.xgb_model_file, 'rb') as f:
                    self.xgb_model = pickle.load(f)
            if os.path.exists(self.config.ensemble_model_file):
                with open(self.config.ensemble_model_file, 'rb') as f:
                    self.ensemble_model = pickle.load(f)
            logger.info("Models loaded")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

# Section 6: Transaction Processor
class TransactionProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.api_endpoint = "https://api.flutterwave.com/v3/charges"
        self.api_key = "FLWSECK_TEST-1234567890"

    def process_transaction(self, transaction: Dict, predicted_status: str) -> Dict:
        for attempt in range(self.config.max_retries):
            try:
                response = self._simulate_flutterwave_api(transaction, predicted_status)
                logger.info(f"Transaction {transaction['transaction_id']} processed: {response['status']}")
                return response
            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error(f"Max retries reached for transaction {transaction['transaction_id']}")
                    return {
                        'status': 'failed',
                        'transaction_id': transaction['transaction_id'],
                        'error': str(e),
                        'timestamp': datetime.now().isoformat(),
                        'cost': 0,
                        'latency': 0,
                        'success_rate': 0
                    }

    def _simulate_flutterwave_api(self, transaction: Dict, predicted_status: str) -> Dict:
        success_rate = 0.95 if predicted_status == 'Success' else 0.5 if predicted_status == 'Failed' else 0.3
        if random.random() < success_rate:
            return {
                'status': 'success',
                'transaction_id': transaction['transaction_id'],
                'cost': float(random.uniform(0.01, 0.05)),
                'latency': float(random.uniform(100, 2000)),
                'success_rate': 1.0,
                'timestamp': datetime.now().isoformat()
            }
        else:
            failure_reasons = ['Network_Error', 'Fraud_Detected', 'Insufficient_Funds', 'Processor_Down']
            chargeback_reasons = ['Fraudulent', 'Disputed', 'Unauthorized']
            return {
                'status': predicted_status.lower(),
                'transaction_id': transaction['transaction_id'],
                'error': random.choice(failure_reasons) if predicted_status == 'Failed' else random.choice(chargeback_reasons),
                'timestamp': datetime.now().isoformat(),
                'cost': 0,
                'latency': 0,
                'success_rate': 0
            }

# Section 7: Feedback Loop
class FeedbackLoop:
    def __init__(self, config: Config, data_preparation: DataPreparation, preprocessor: DataPreprocessor, model: PaymentModel):
        self.config = config
        self.data_preparation = data_preparation
        self.preprocessor = preprocessor
        self.model = model
        self.transaction_queue = queue.Queue()

    def update_metrics(self, transaction_result: Dict) -> None:
        try:
            new_metric = {
                'processor_pathway': transaction_result.get('processor_pathway', 'Card_Verve'),
                'cost': float(transaction_result.get('cost', 0.03)),
                'latency': float(transaction_result.get('latency', 200)),
                'success_rate': float(transaction_result.get('success_rate', 0)),
                'timestamp': transaction_result['timestamp'],
                'network_stability': float(random.uniform(0.85, 1.0)),
                'availability': 1,
                'failure_reason': transaction_result.get('error', 'None')
            }
            new_metric_df = pd.DataFrame([new_metric])
            new_metric_df.to_csv(self.config.processor_file, mode='a', header=not os.path.exists(self.config.processor_file), index=False)
            self.transaction_queue.put(transaction_result)
            logger.info(f"Updated metrics for transaction {transaction_result['transaction_id']}")
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

    def retrain_models(self) -> None:
        try:
            self.data_preparation.load_data()
            self.data_preparation.validate_data()
            merged_data = self.data_preparation.merge_data()
            sequences, seq_labels = self.data_preparation.create_sequences()
            X_flat, y_flat = self.preprocessor.preprocess_data(merged_data, training=True)

            self.model.train_lstm(sequences, seq_labels)
            self.model.train_transformer(sequences, seq_labels)
            self.model.train_xgb(X_flat, y_flat)
            self.model.train_ensemble(sequences, X_flat, y_flat)
            self.model.save_models()
            self.preprocessor.save_preprocessor()
            logger.info("Models retrained successfully")
            self._generate_retraining_diagram()
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
            raise

    def _generate_retraining_diagram(self) -> None:
        try:
            fig = go.Figure(data=[
                go.Bar(
                    x=['LSTM', 'Transformer', 'XGBoost', 'Ensemble'],
                    y=[1, 1, 1, 1],
                    marker_color='#1f77b4'
                )
            ])
            fig.update_layout(
                title='Model Retraining Trigger',
                xaxis_title='Model',
                yaxis_title='Retrained',
                title_x=0.5
            )
            fig.write_image(os.path.join(self.config.data_dir, 'retraining_diagram.png'), scale=2)
            fig.write_json(os.path.join(self.config.data_dir, 'retraining_diagram.json'))
            logger.info(f"Generated retraining diagram at {os.path.join(self.config.data_dir, 'retraining_diagram.png')}")
        except Exception as e:
            logger.error(f"Error generating retraining diagram: {e}")

    def process_queue(self):
        while True:
            try:
                if not self.transaction_queue.empty():
                    transaction = self.transaction_queue.get()
                    if random.random() < 0.005:  # 0.5% chance to retrain
                        self.retrain_models()
                time.sleep(1)
                metrics.inc_memory(psutil.Process().memory_info().rss)
            except Exception as e:
                logger.error(f"Error processing queue: {e}")

# Section 8: Analytics Dashboard
class AnalyticsDashboard:
    def __init__(self, config: Config, data_preparation: DataPreparation, model: PaymentModel):
        self.config = config
        self.data_preparation = data_preparation
        self.model = model
        self.app = Dash(__name__)
        self.app.config.suppress_callback_exceptions = True
        # Ensure data is loaded before setting up dashboard
        if self.data_preparation.transactions is None:
            try:
                self.data_preparation.load_data()
                self.data_preparation.validate_data()
            except Exception as e:
                logger.error(f"Error loading data in AnalyticsDashboard: {e}")
                raise
        self.setup_dashboard()

    def setup_dashboard(self) -> None:
        try:
            # Check if transactions is still None
            if self.data_preparation.transactions is None:
                logger.error("Transactions data not loaded")
                raise ValueError("Transactions data not loaded")

            self.app.layout = html.Div([
                html.H1("Payment Failure & Chargeback Analytics Dashboard",
                        style={'textAlign': 'center', 'color': '#1f77b4', 'fontSize': '32px'}),
                html.Div([
                    dcc.DatePickerRange(
                        id='date-range',
                        min_date_allowed=datetime(2024, 7, 1),
                        max_date_allowed=datetime(2025, 12, 31),
                        start_date=datetime(2025, 7, 1),
                        end_date=datetime(2025, 7, 23),
                        style={'margin': '10px', 'width': '25%'}
                    ),
                    dcc.Dropdown(
                        id='payment-method-filter',
                        options=[{'label': pm, 'value': pm} for pm in self.config.payment_methods + ['All']],
                        value='All',
                        style={'margin': '10px', 'width': '20%'}
                    ),
                    dcc.Dropdown(
                        id='location-filter',
                        options=[{'label': loc, 'value': loc} for loc in
                                (self.data_preparation.transactions['location'].unique()
                                 if self.data_preparation.transactions is not None and 'location' in self.data_preparation.transactions.columns
                                 else [])] + [{'label': 'All', 'value': 'All'}],
                        value='All',
                        style={'margin': '10px', 'width': '20%'}
                    ),
                    dcc.Dropdown(
                        id='model-filter',
                        options=[{'label': m, 'value': m} for m in ['LSTM', 'Transformer', 'XGBoost', 'Ensemble']],
                        value='Ensemble',
                        style={'margin': '10px', 'width': '20%'}
                    ),
                    html.Button('Export Visualizations', id='export-button', style={'margin': '10px'})
                ], style={'display': 'flex', 'justifyContent': 'center', 'backgroundColor': '#f8f9fa', 'padding': '10px'}),
                html.Div(id='export-output', style={'textAlign': 'center', 'color': '#1f77b4'}),
                html.Div([
                    dcc.Graph(id='status-pie', style={'width': '33%', 'display': 'inline-block'}),
                    dcc.Graph(id='failure-reasons-bar', style={'width': '33%', 'display': 'inline-block'}),
                    dcc.Graph(id='chargeback-reasons-bar', style={'width': '33%', 'display': 'inline-block'}),
                    dcc.Graph(id='volume-time-series', style={'width': '50%', 'display': 'inline-block'}),
                    dcc.Graph(id='fraud-risk-histogram', style={'width': '50%', 'display': 'inline-block'}),
                    dcc.Graph(id='success-rate-payment', style={'width': '50%', 'display': 'inline-block'}),
                    dcc.Graph(id='latency-box', style={'width': '50%', 'display': 'inline-block'}),
                    dcc.Graph(id='cost-success-scatter', style={'width': '50%', 'display': 'inline-block'}),
                    dcc.Graph(id='merchant-chargeback-bar', style={'width': '50%', 'display': 'inline-block'}),
                    dcc.Graph(id='prediction-confidence', style={'width': '50%', 'display': 'inline-block'}),
                    dcc.Graph(id='roc-curve', style={'width': '50%', 'display': 'inline-block'}),
                    dcc.Graph(id='feature-importance', style={'width': '50%', 'display': 'inline-block'}),
                    dcc.Graph(id='attention-weights', style={'width': '50%', 'display': 'inline-block'}),
                    dcc.Graph(id='transaction-velocity', style={'width': '50%', 'display': 'inline-block'}),
                    dcc.Graph(id='system-metrics', style={'width': '50%', 'display': 'inline-block'})
                ], style={'padding': '20px', 'backgroundColor': '#ffffff'}),
                dcc.Interval(id='interval-component', interval=20*1000, n_intervals=0)  # Update every 20 seconds
            ])

            self.app.callback(
                [
                    Output('status-pie', 'figure'),
                    Output('failure-reasons-bar', 'figure'),
                    Output('chargeback-reasons-bar', 'figure'),
                    Output('volume-time-series', 'figure'),
                    Output('fraud-risk-histogram', 'figure'),
                    Output('success-rate-payment', 'figure'),
                    Output('latency-box', 'figure'),
                    Output('cost-success-scatter', 'figure'),
                    Output('merchant-chargeback-bar', 'figure'),
                    Output('prediction-confidence', 'figure'),
                    Output('roc-curve', 'figure'),
                    Output('feature-importance', 'figure'),
                    Output('attention-weights', 'figure'),
                    Output('transaction-velocity', 'figure'),
                    Output('system-metrics', 'figure')
                ],
                [
                    Input('date-range', 'start_date'),
                    Input('date-range', 'end_date'),
                    Input('payment-method-filter', 'value'),
                    Input('location-filter', 'value'),
                    Input('model-filter', 'value'),
                    Input('interval-component', 'n_intervals')
                ]
            )(self.update_dashboard)

            self.app.callback(
                Output('export-output', 'children'),
                Input('export-button', 'n_clicks'),
                State('date-range', 'start_date'),
                State('date-range', 'end_date'),
                State('payment-method-filter', 'value'),
                State('location-filter', 'value'),
                State('model-filter', 'value')
            )(self.export_visualizations)
            logger.info("Dashboard setup complete")
        except Exception as e:
            logger.error(f"Error setting up dashboard: {e}")
            raise

    def update_dashboard(self, start_date: str, end_date: str, payment_method: str, location: str, model_type: str, n_intervals: int) -> List[go.Figure]:
        try:
            df = self.data_preparation.merged_data_cache.copy()
            if df is None or df.empty:
                logger.warning("Merged data cache is empty")
                return [go.Figure() for _ in range(15)]
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
            if payment_method != 'All':
                df = df[df['payment_method'] == payment_method]
            if location != 'All':
                df = df[df['location'] == location]

            # 1. Status Pie Chart
            status_counts = df['status'].value_counts()
            status_fig = px.pie(
                names=status_counts.index,
                values=status_counts.values,
                title='Transaction Status Distribution',
                color_discrete_sequence=px.colors.qualitative.D3
            )
            status_fig.write_json(os.path.join(self.config.data_dir, 'status_pie.json'))
            try:
                status_fig.write_image(os.path.join(self.config.data_dir, 'status_pie.png'), scale=2)
            except Exception as e:
                logger.warning(f"Failed to export status pie image due to Kaleido issue: {e}. JSON exported instead.")

            # 2. Failure Reasons Bar
            failure_counts = df[df['status'] == 'Failed']['failure_reason'].value_counts()
            failure_fig = go.Figure(data=[
                go.Bar(x=failure_counts.index, y=failure_counts.values, marker_color='#ff7f0e')
            ])
            failure_fig.update_layout(title='Failure Reasons Breakdown', xaxis_title='Reason', yaxis_title='Count', title_x=0.5)
            failure_fig.write_json(os.path.join(self.config.data_dir, 'failure_reasons_bar.json'))
            try:
                failure_fig.write_image(os.path.join(self.config.data_dir, 'failure_reasons_bar.png'), scale=2)
            except Exception as e:
                logger.warning(f"Failed to export failure reasons bar image due to Kaleido issue: {e}. JSON exported instead.")

            # 3. Chargeback Reasons Bar
            chargeback_counts = df[df['status'] == 'Chargeback']['chargeback_reason'].value_counts()
            chargeback_fig = go.Figure(data=[
                go.Bar(x=chargeback_counts.index, y=chargeback_counts.values, marker_color='#d62728')
            ])
            chargeback_fig.update_layout(title='Chargeback Reasons Breakdown', xaxis_title='Reason', yaxis_title='Count', title_x=0.5)
            chargeback_fig.write_json(os.path.join(self.config.data_dir, 'chargeback_reasons_bar.json'))
            try:
                chargeback_fig.write_image(os.path.join(self.config.data_dir, 'chargeback_reasons_bar.png'), scale=2)
            except Exception as e:
                logger.warning(f"Failed to export chargeback reasons bar image due to Kaleido issue: {e}. JSON exported instead.")

            # 4. Volume Time Series
            df['date'] = df['timestamp'].dt.date
            volume = df.groupby('date').size().reset_index(name='count')
            volume_fig = px.line(volume, x='date', y='count', title='Transaction Volume Over Time', line_shape='spline')
            volume_fig.update_layout(title_x=0.5)
            volume_fig.write_json(os.path.join(self.config.data_dir, 'volume_time_series.json'))
            try:
                volume_fig.write_image(os.path.join(self.config.data_dir, 'volume_time_series.png'), scale=2)
            except Exception as e:
                logger.warning(f"Failed to export volume time series image due to Kaleido issue: {e}. JSON exported instead.")

            # 5. Fraud Risk Histogram
            fraud_fig = px.histogram(df, x='fraud_risk_score', nbins=50, title='Fraud Risk Score Distribution',
                                    color_discrete_sequence=['#1f77b4'])
            fraud_fig.update_layout(title_x=0.5)
            fraud_fig.write_json(os.path.join(self.config.data_dir, 'fraud_risk_histogram.json'))
            try:
                fraud_fig.write_image(os.path.join(self.config.data_dir, 'fraud_risk_histogram.png'), scale=2)
            except Exception as e:
                logger.warning(f"Failed to export fraud risk histogram image due to Kaleido issue: {e}. JSON exported instead.")

            # 6. Success Rate by Payment Method
            success_rate = df.groupby('payment_method')['status'].apply(lambda x: (x == 'Success').mean()).reset_index(name='success_rate')
            success_fig = go.Figure(data=[
                go.Bar(x=success_rate['payment_method'], y=success_rate['success_rate'], marker_color='#1f77b4')
            ])
            success_fig.update_layout(title='Success Rate by Payment Method', xaxis_title='Payment Method', yaxis_title='Success Rate', title_x=0.5)
            success_fig.write_json(os.path.join(self.config.data_dir, 'success_rate_payment.json'))
            try:
                success_fig.write_image(os.path.join(self.config.data_dir, 'success_rate_payment.png'), scale=2)
            except Exception as e:
                logger.warning(f"Failed to export success rate payment image due to Kaleido issue: {e}. JSON exported instead.")

            # 7. Latency Box Plot
            latency_fig = px.box(df, x='processor_pathway', y='latency', title='Latency by Processor Pathway', color='processor_pathway')
            latency_fig.update_layout(xaxis_title='Processor Pathway', yaxis_title='Latency (ms)', title_x=0.5)
            latency_fig.write_json(os.path.join(self.config.data_dir, 'latency_box.json'))
            try:
                latency_fig.write_image(os.path.join(self.config.data_dir, 'latency_box.png'), scale=2)
            except Exception as e:
                logger.warning(f"Failed to export latency box image due to Kaleido issue: {e}. JSON exported instead.")

            # 8. Cost vs Success Rate Scatter
            scatter_fig = px.scatter(df, x='cost', y='success_rate', color='processor_pathway', title='Cost vs Success Rate', size='amount')
            scatter_fig.update_layout(title_x=0.5)
            scatter_fig.write_json(os.path.join(self.config.data_dir, 'cost_success_scatter.json'))
            try:
                scatter_fig.write_image(os.path.join(self.config.data_dir, 'cost_success_scatter.png'), scale=2)
            except Exception as e:
                logger.warning(f"Failed to export cost success scatter image due to Kaleido issue: {e}. JSON exported instead.")

            # 9. Merchant Chargeback Rate
            chargeback_rate = df.groupby('merchant_id')['status'].apply(lambda x: (x == 'Chargeback').mean()).reset_index(name='chargeback_rate')
            chargeback_rate = chargeback_rate.sort_values('chargeback_rate', ascending=False).head(10)
            merchant_fig = go.Figure(data=[
                go.Bar(x=chargeback_rate['merchant_id'], y=chargeback_rate['chargeback_rate'], marker_color='#9467bd')
            ])
            merchant_fig.update_layout(title='Top Merchants by Chargeback Rate', xaxis_title='Merchant ID', yaxis_title='Chargeback Rate', title_x=0.5)
            merchant_fig.write_json(os.path.join(self.config.data_dir, 'merchant_chargeback_bar.json'))
            try:
                merchant_fig.write_image(os.path.join(self.config.data_dir, 'merchant_chargeback_bar.png'), scale=2)
            except Exception as e:
                logger.warning(f"Failed to export merchant chargeback bar image due to Kaleido issue: {e}. JSON exported instead.")

            # 10. Prediction Confidence
            recent_transactions = df.tail(100)
            sequences, _ = self.data_preparation.create_sequences()
            X_flat, _ = self.preprocessor.preprocess_data(recent_transactions, training=False)
            if model_type == 'LSTM':
                probs = self.model.lstm_model.predict(sequences[-100:])
            elif model_type == 'Transformer':
                probs = self.model.transformer_model.predict(sequences[-100:])
            elif model_type == 'XGBoost':
                probs = self.model.xgb_model.predict_proba(X_flat)
            else:
                probs = np.eye(3)[self.model.ensemble_predict(sequences[-100:], X_flat)]
            confidence_fig = go.Figure(data=[
                go.Scatter(x=recent_transactions['timestamp'], y=np.max(probs, axis=1), mode='lines',
                          name='Confidence', line=dict(color='#2ca02c'))
            ])
            confidence_fig.update_layout(title='Prediction Confidence (Last 100 Transactions)', xaxis_title='Timestamp',
                                       yaxis_title='Confidence', title_x=0.5)
            confidence_fig.write_json(os.path.join(self.config.data_dir, 'prediction_confidence.json'))
            try:
                confidence_fig.write_image(os.path.join(self.config.data_dir, 'prediction_confidence.png'), scale=2)
            except Exception as e:
                logger.warning(f"Failed to export prediction confidence image due to Kaleido issue: {e}. JSON exported instead.")

            # 11. ROC Curve
            sequences, labels = self.data_preparation.create_sequences()
            X_flat, y_flat = self.preprocessor.preprocess_data(self.data_preparation.merged_data_cache, training=True)
            if model_type == 'LSTM':
                y_scores = self.model.lstm_model.predict(sequences)
            elif model_type == 'Transformer':
                y_scores = self.model.transformer_model.predict(sequences)
            elif model_type == 'XGBoost':
                y_scores = self.model.xgb_model.predict_proba(X_flat)
            else:
                y_scores = np.eye(3)[self.model.ensemble_predict(sequences, X_flat)]
            roc_fig = go.Figure()
            for i, label in enumerate(['Success', 'Failed', 'Chargeback']):
                fpr, tpr, _ = roc_curve(labels == i, y_scores[:, i])
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC {label}'))
            roc_fig.update_layout(title=f'{model_type} ROC Curve', xaxis_title='False Positive Rate',
                                 yaxis_title='True Positive Rate', title_x=0.5)
            roc_fig.write_json(os.path.join(self.config.data_dir, 'roc_curve.json'))
            try:
                roc_fig.write_image(os.path.join(self.config.data_dir, 'roc_curve.png'), scale=2)
            except Exception as e:
                logger.warning(f"Failed to export ROC curve image due to Kaleido issue: {e}. JSON exported instead.")

            # 12. Feature Importance
            if model_type in ['XGBoost', 'Ensemble']:
                importances = self.model.xgb_model.feature_importances_
                feature_names = self.preprocessor.feature_names if self.preprocessor.feature_names else [f"feature_{i}" for i in range(X_flat.shape[1])]
                sorted_idx = np.argsort(importances)[::-1][:10]
                importance_fig = go.Figure(data=[
                    go.Bar(x=np.array(feature_names)[sorted_idx], y=importances[sorted_idx], marker_color='#1f77b4')
                ])
                importance_fig.update_layout(title='Top 10 Feature Importance', xaxis_title='Feature', yaxis_title='Importance',
                                           xaxis_tickangle=45, title_x=0.5)
            else:
                importance_fig = go.Figure()
                importance_fig.update_layout(title='Feature Importance (Not Available for Deep Learning Models)', title_x=0.5)
            importance_fig.write_json(os.path.join(self.config.data_dir, 'feature_importance.json'))
            try:
                importance_fig.write_image(os.path.join(self.config.data_dir, 'feature_importance.png'), scale=2)
            except Exception as e:
                logger.warning(f"Failed to export feature importance image due to Kaleido issue: {e}. JSON exported instead.")

            # 13. Attention Weights (Transformer only)
            if model_type == 'Transformer':
                sample_seq = sequences[-1:]
                transformer_layer = self.model.transformer_model.layers[1]  # First DynamicTransformerLayer
                _, attn_weights = transformer_layer.mha(sample_seq, sample_seq, sample_seq, return_attention_scores=True)
                attn_fig = go.Figure(data=[
                    go.Heatmap(z=attn_weights[0, 0], x=np.arange(self.config.sequence_length),
                              y=np.arange(self.config.sequence_length), colorscale='Viridis')
                ])
                attn_fig.update_layout(title='Transformer Attention Weights', xaxis_title='Sequence Position',
                                      yaxis_title='Sequence Position', title_x=0.5)
            else:
                attn_fig = go.Figure()
                attn_fig.update_layout(title='Attention Weights (Transformer Only)', title_x=0.5)
            attn_fig.write_json(os.path.join(self.config.data_dir, 'attention_weights.json'))
            try:
                attn_fig.write_image(os.path.join(self.config.data_dir, 'attention_weights.png'), scale=2)
            except Exception as e:
                logger.warning(f"Failed to export attention weights image due to Kaleido issue: {e}. JSON exported instead.")

            # 14. Transaction Velocity
            velocity = df.groupby(df['timestamp'].dt.hour)['transaction_id'].count().reset_index(name='count')
            velocity_fig = go.Figure(data=[
                go.Scatter(x=velocity['timestamp'], y=velocity['count'], mode='lines+markers', name='Velocity', line=dict(color='#9467bd'))
            ])
            velocity_fig.update_layout(title='Transaction Velocity by Hour', xaxis_title='Hour of Day', yaxis_title='Transaction Count', title_x=0.5)
            velocity_fig.write_json(os.path.join(self.config.data_dir, 'transaction_velocity.json'))
            try:
                velocity_fig.write_image(os.path.join(self.config.data_dir, 'transaction_velocity.png'), scale=2)
            except Exception as e:
                logger.warning(f"Failed to export transaction velocity image due to Kaleido issue: {e}. JSON exported instead.")

            # 15. System Metrics
            system_fig = go.Figure()
            system_fig.add_trace(go.Scatter(x=[datetime.now()], y=[psutil.Process().memory_info().rss / 1024**2],
                                          mode='lines', name='Memory Usage (MB)'))
            system_fig.add_trace(go.Scatter(x=[datetime.now()], y=[psutil.cpu_percent()], mode='lines', name='CPU Usage (%)'))
            system_fig.update_layout(title='System Resource Metrics', xaxis_title='Time', yaxis_title='Usage', title_x=0.5)
            system_fig.write_json(os.path.join(self.config.data_dir, 'system_metrics.json'))
            try:
                system_fig.write_image(os.path.join(self.config.data_dir, 'system_metrics.png'), scale=2)
            except Exception as e:
                logger.warning(f"Failed to export system metrics image due to Kaleido issue: {e}. JSON exported instead.")

            return [
                status_fig, failure_fig, chargeback_fig, volume_fig, fraud_fig,
                success_fig, latency_fig, scatter_fig, merchant_fig, confidence_fig,
                roc_fig, importance_fig, attn_fig, velocity_fig, system_fig
            ]
        except Exception as e:
            logger.error(f"Error updating dashboard: {e}")
            return [go.Figure() for _ in range(15)]

    def export_visualizations(self, n_clicks: int, start_date: str, end_date: str, payment_method: str, location: str, model_type: str) -> str:
        if n_clicks is None:
            return ""
        try:
            export_dir = os.path.join(self.config.data_dir, 'exported_visualizations')
            os.makedirs(export_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = os.path.join(export_dir, f'dashboard_export_{timestamp}')
            os.makedirs(export_path, exist_ok=True)

            figures = self.update_dashboard(start_date, end_date, payment_method, location, model_type, 0)
            figure_names = [
                'status_pie', 'failure_reasons_bar', 'chargeback_reasons_bar', 'volume_time_series',
                'fraud_risk_histogram', 'success_rate_payment', 'latency_box', 'cost_success_scatter',
                'merchant_chargeback_bar', 'prediction_confidence', 'roc_curve', 'feature_importance',
                'attention_weights', 'transaction_velocity', 'system_metrics'
            ]

            for fig, name in zip(figures, figure_names):
                fig.write_json(os.path.join(export_path, f'{name}.json'))
                try:
                    fig.write_image(os.path.join(export_path, f'{name}.png'), scale=2)
                except Exception as e:
                    logger.warning(f"Failed to export {name} image due to Kaleido issue: {e}. JSON exported instead.")

            # Create a summary report
            summary = {
                'export_timestamp': timestamp,
                'start_date': start_date,
                'end_date': end_date,
                'payment_method': payment_method,
                'location': location,
                'model_type': model_type,
                'exported_files': [f'{name}.json' for name in figure_names] +
                                 [f'{name}.png' for name in figure_names if os.path.exists(os.path.join(export_path, f'{name}.png'))]
            }
            with open(os.path.join(export_path, 'export_summary.json'), 'w') as f:
                json.dump(summary, f, cls=NumpyJSONEncoder, indent=2)

            logger.info(f"Visualizations exported to {export_path}")
            return f"Visualizations exported successfully to {export_path}"
        except Exception as e:
            logger.error(f"Error exporting visualizations: {e}")
            return f"Error exporting visualizations: {str(e)}"

    def run(self, host: str = '0.0.0.0', port: int = 8050) -> None:
        try:
            self.app.run_server(host=host, port=port, debug=False)
            logger.info(f"Dashboard running on http://{host}:{port}")
        except Exception as e:
            logger.error(f"Error running dashboard: {e}")
            raise

# Section 9: API Service
class APIService:
    def __init__(self, config: Config, data_preparation: DataPreparation, preprocessor: DataPreprocessor,
                 model: PaymentModel, transaction_processor: TransactionProcessor, feedback_loop: FeedbackLoop):
        self.config = config
        self.data_preparation = data_preparation
        self.preprocessor = preprocessor
        self.model = model
        self.transaction_processor = transaction_processor
        self.feedback_loop = feedback_loop
        self.app = FastAPI(title="Payment Failure Prediction API")
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        self.rate_limiter = defaultdict(lambda: {'count': 0, 'last_reset': time.time()})
        self.setup_routes()
        self.setup_middleware()

    def setup_middleware(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    class TransactionRequest(BaseModel):
        transaction_id: str = Field(..., example="TXN_123")
        customer_id: str = Field(..., example="CUST_456")
        merchant_id: str = Field(..., example="MERCH_789")
        amount: float = Field(..., gt=0, example=100.50)
        currency: str = Field(..., example="NGN")
        payment_method: str = Field(..., example="Card")
        processor_pathway: str = Field(..., example="Card_Verve")
        merchant_type: str = Field(..., example="E-commerce")
        fraud_risk_score: float = Field(..., ge=0, le=1, example=0.3)
        device_type: str = Field(..., example="Mobile")
        network_type: str = Field(..., example="4G")
        location: str = Field(..., example="Lagos")
        transaction_type: str = Field(..., example="Purchase")
        timestamp: str = Field(..., example="2025-07-23T11:18:00")
        hour_of_day: int = Field(..., ge=0, le=23, example=11)
        day_of_week: int = Field(..., ge=0, le=6, example=2)
        is_weekend: int = Field(..., ge=0, le=1, example=0)
        time_since_last_transaction: float = Field(..., ge=0, example=3600.0)

    def setup_routes(self) -> None:
        @self.app.post("/predict")
        async def predict_transaction(transaction: self.TransactionRequest, request: Request):
            start_time = time.time()
            client_ip = request.client.host
            try:
                if not self._check_rate_limit(client_ip):
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")

                metrics.inc_request()
                transaction_dict = transaction.dict()
                transaction_df = pd.DataFrame([transaction_dict])
                transaction_df['timestamp'] = pd.to_datetime(transaction_df['timestamp'])

                # Merge with user and merchant data
                user_data = self.data_preparation.users[self.data_preparation.users['customer_id'] == transaction.customer_id]
                merchant_data = self.data_preparation.merchants[self.data_preparation.merchants['merchant_id'] == transaction.merchant_id]
                processor_data = self.data_preparation.processors[
                    self.data_preparation.processors['processor_pathway'] == transaction.processor_pathway
                ].sort_values('timestamp').tail(1)

                merged_df = pd.merge(transaction_df, user_data[['customer_id', 'user_risk_score', 'average_transaction_frequency']],
                                   on='customer_id', how='left')
                merged_df = pd.merge(merged_df, merchant_data[['merchant_id', 'merchant_risk_score']],
                                   on='merchant_id', how='left')
                merged_df = pd.merge(merged_df, processor_data[['processor_pathway', 'cost', 'latency', 'success_rate', 'network_stability']],
                                   on='processor_pathway', how='left')
                merged_df.fillna({
                    'user_risk_score': 0.1,
                    'average_transaction_frequency': 0.1,
                    'merchant_risk_score': 0.1,
                    'cost': 0.03,
                    'latency': 200,
                    'success_rate': 0.95,
                    'network_stability': 0.9
                }, inplace=True)
                merged_df['amount_log'] = np.log1p(merged_df['amount'])
                merged_df['risk_composite'] = 0.6 * merged_df['user_risk_score'] + 0.4 * merged_df['merchant_risk_score']

                # Create sequence for deep learning models
                customer_transactions = self.data_preparation.transactions[
                    self.data_preparation.transactions['customer_id'] == transaction.customer_id
                ].sort_values('timestamp')
                customer_transactions = pd.concat([customer_transactions, merged_df]).tail(self.config.sequence_length)
                feature_columns = [
                    col for col in merged_df.columns
                    if col not in ['transaction_id', 'customer_id', 'merchant_id', 'timestamp', 'status']
                ]
                sequence = customer_transactions[feature_columns].values
                sequence = np.array([sequence])

                # Preprocess data
                X_flat, _ = self.preprocessor.preprocess_data(merged_df, training=False)

                # Cache predictions
                cache_key = hashlib.md5(json.dumps(transaction_dict, cls=NumpyJSONEncoder).encode()).hexdigest()
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    logger.info(f"Cache hit for transaction {transaction.transaction_id}")
                    result = json.loads(cached_result)
                else:
                    # Ensemble prediction
                    pred = self.model.ensemble_predict(sequence, X_flat)
                    pred_proba = np.eye(3)[pred][0]
                    status_map = {0: 'Success', 1: 'Failed', 2: 'Chargeback'}
                    predicted_status = status_map[pred[0]]

                    # Process transaction
                    transaction_result = self.transaction_processor.process_transaction(transaction_dict, predicted_status)
                    self.feedback_loop.update_metrics(transaction_result)

                    result = {
                        'transaction_id': transaction.transaction_id,
                        'predicted_status': predicted_status,
                        'confidence': float(np.max(pred_proba)),
                        'probabilities': {'Success': float(pred_proba[0]), 'Failed': float(pred_proba[1]), 'Chargeback': float(pred_proba[2])},
                        'processor_response': transaction_result
                    }
                    self.redis_client.setex(cache_key, self.config.cache_ttl, json.dumps(result, cls=NumpyJSONEncoder))

                latency = time.time() - start_time
                metrics.observe_latency(latency)
                metrics.inc_memory(psutil.Process().memory_info().rss)
                return result
            except Exception as e:
                logger.error(f"Error processing transaction {transaction.transaction_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}

    def _check_rate_limit(self, client_ip: str) -> bool:
        current_time = time.time()
        if current_time - self.rate_limiter[client_ip]['last_reset'] > 60:
            self.rate_limiter[client_ip] = {'count': 0, 'last_reset': current_time}
        self.rate_limiter[client_ip]['count'] += 1
        return self.rate_limiter[client_ip]['count'] <= self.config.rate_limit

    def run(self, host: str = '0.0.0.0', port: int = 8000) -> None:
        try:
            import uvicorn
            uvicorn.run(self.app, host=host, port=port)
            logger.info(f"API running on http://{host}:{port}")
        except Exception as e:
            logger.error(f"Error running API: {e}")
            raise

# Section 10: Main Application
class PaymentPredictionApp:
    def __init__(self):
        self.config = Config()
        self.data_preparation = DataPreparation(self.config)
        self.preprocessor = DataPreprocessor(self.config)
        self.model = PaymentModel(self.config)
        self.transaction_processor = TransactionProcessor(self.config)
        self.feedback_loop = FeedbackLoop(self.config, self.data_preparation, self.preprocessor, self.model)
        self.analytics_dashboard = AnalyticsDashboard(self.config, self.data_preparation, self.model)
        self.api_service = APIService(self.config, self.data_preparation, self.preprocessor,
                                    self.model, self.transaction_processor, self.feedback_loop)

    def initialize(self) -> None:
        try:
            self.data_preparation.load_data()
            self.data_preparation.validate_data()
            merged_data = self.data_preparation.merge_data()
            sequences, seq_labels = self.data_preparation.create_sequences()
            self.preprocessor.build_preprocessor(merged_data)
            X_flat, y_flat = self.preprocessor.preprocess_data(merged_data, training=True)
            self.model.train_lstm(sequences, seq_labels)
            self.model.train_transformer(sequences, seq_labels)
            self.model.train_xgb(X_flat, y_flat)
            self.model.train_ensemble(sequences, X_flat, y_flat)
            self.model.save_models()
            self.preprocessor.save_preprocessor()
            logger.info("Application initialized")
        except Exception as e:
            logger.error(f"Error initializing application: {e}")
            raise

    def run(self) -> None:
        try:
            # Start feedback loop in a separate thread
            feedback_thread = threading.Thread(target=self.feedback_loop.process_queue, daemon=True)
            feedback_thread.start()

            # Start API and dashboard in separate processes
            api_process = multiprocessing.Process(
                target=self.api_service.run,
                kwargs={'host': '0.0.0.0', 'port': 8000}
            )
            dashboard_process = multiprocessing.Process(
                target=self.analytics_dashboard.run,
                kwargs={'host': '0.0.0.0', 'port': 8050}
            )

            api_process.start()
            dashboard_process.start()

            api_process.join()
            dashboard_process.join()
            logger.info("Application running")
        except Exception as e:
            logger.error(f"Error running application: {e}")
            raise

if __name__ == "__main__":
    import multiprocessing
    app = PaymentPredictionApp()
    app.initialize()
    app.run()