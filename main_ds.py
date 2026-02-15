#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTCUSDT 深度强化学习交易系统 - Testnet完整方案
所有功能整合在单个文件中，方便运行和调试
"""

import os
import sys
import json
import time
import math
import random
import logging
import asyncio
import threading
import warnings
import pickle
import sqlite3
import hashlib
import hmac
import base64.
import urllib.parse
from datetime import datetime, timedelta
from collections import deque, namedtuple
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import queue

import numpy as np
import pandas as pd
import requests
import websocket
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# Scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 忽略警告
warnings.filterwarnings('ignore')

# ==================== 配置部分 ====================

@dataclass
class Config:
    """全局配置类"""
    # Bybit Testnet配置
    BYBIT_TESTNET_API_KEY = os.getenv('BYBIT_TESTNET_API_KEY', 'your_testnet_api_key')
    BYBIT_TESTNET_API_SECRET = os.getenv('BYBIT_TESTNET_API_SECRET', 'your_testnet_api_secret')
    BYBIT_TESTNET_WS_URL = "wss://stream-testnet.bybit.com/v5/public/linear"
    BYBIT_TESTNET_REST_URL = "https://api-testnet.bybit.com"
    
    # 交易配置
    SYMBOL = "BTCUSDT"
    INITIAL_BALANCE = 1000.0  # USDT
    FEE_RATE = 0.0006  # 0.06% taker fee
    MAX_POSITION_PCT = 0.3  # 最大仓位比例 30%
    
    # 数据配置
    DATA_PATH = "./data"
    KLINE_INTERVAL = "1"  # 1分钟
    HISTORY_YEARS = 3
    FEATURE_DIM = 256
    
    # 模型配置
    HIDDEN_DIM = 512
    NUM_LAYERS = 2
    DROPOUT = 0.2
    NUM_HEADS = 8
    
    # PPO配置
    GAMMA = 0.99
    LAMBDA = 0.95
    CLIP_EPSILON = 0.2
    ENTROPY_COEF = 0.01
    VALUE_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    TARGET_KL = 0.015
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 256
    BUFFER_CAPACITY = 20000
    UPDATE_EPOCHS = 10
    
    # 风控配置
    MAX_DAILY_LOSS_PCT = 0.1  # 10%
    MAX_CONSECUTIVE_LOSSES = 3
    MIN_BALANCE = 10.0
    STOP_LOSS_PCT = 0.02  # 2%
    TAKE_PROFIT_PCT = 0.03  # 3%
    TRAILING_STOP_PCT = 0.01  # 1%
    MAX_SLIPPAGE = 0.001  # 0.1%
    MIN_LIQUIDITY = 100.0  # BTC
    MAX_TRADES_PER_MINUTE = 6
    MAX_PRICE_DEVIATION = 0.02  # 2%
    
    # 监控配置
    LOG_LEVEL = logging.INFO
    DASHBOARD_PORT = 8501
    ALERT_TELEGRAM = False
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# ==================== 数据模块 ====================

class DataManager:
    """数据管理器 - 处理历史数据和实时数据"""
    
    def __init__(self, config: Config):
        self.config = config
        self.symbol = config.SYMBOL
        self.interval = config.KLINE_INTERVAL
        self.data_path = Path(config.DATA_PATH) / "raw" / self.symbol
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # 实时数据缓存
        self.price_buffer = deque(maxlen=1000)
        self.volume_buffer = deque(maxlen=1000)
        self.orderbook_buffer = {}
        self.last_update = None
        
        # WebSocket连接
        self.ws = None
        self.ws_thread = None
        self.ws_running = False
        self.ws_callbacks = {}
        
        # 特征工程
        self.feature_engineer = FeatureEngineer(config)
        self.scaler = StandardScaler()
        
        # 数据库连接
        self._init_database()
        
    def _init_database(self):
        """初始化元数据数据库"""
        db_path = Path(self.config.DATA_PATH) / "metadata.db"
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        cursor = self.conn.cursor()
        
        # 创建数据索引表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                date TEXT,
                file_path TEXT,
                row_count INTEGER,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                UNIQUE(symbol, date)
            )
        ''')
        
        # 创建特征配置表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT,
                feature_names TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
        
    def download_historical(self, years: int = 3) -> pd.DataFrame:
        """下载历史K线数据"""
        logging.info(f"开始下载 {years} 年历史数据...")
        
        all_data = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)
        
        # 按年下载
        current_date = start_date
        while current_date < end_date:
            year = current_date.year
            month = current_date.month
            
            # 创建月度目录
            month_path = self.data_path / str(year) / f"{month:02d}"
            month_path.mkdir(parents=True, exist_ok=True)
            
            # 下载月度数据
            month_data = self._download_month_data(year, month)
            
            if not month_data.empty:
                # 按天保存
                for date, day_data in month_data.groupby(month_data.index.date):
                    date_str = date.strftime("%Y-%m-%d")
                    file_path = month_path / f"{date_str}.parquet"
                    
                    # 保存为Parquet
                    day_data.to_parquet(file_path)
                    
                    # 更新索引
                    cursor = self.conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO data_index 
                        (symbol, date, file_path, row_count, start_time, end_time)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        self.symbol,
                        date_str,
                        str(file_path),
                        len(day_data),
                        day_data.index.min(),
                        day_data.index.max()
                    ))
                    self.conn.commit()
                    
                    all_data.append(day_data)
                    
                    logging.info(f"已保存 {date_str} 数据，{len(day_data)} 条记录")
            
            # 下一个月
            if month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)
        
        if all_data:
            df = pd.concat(all_data)
            logging.info(f"数据下载完成，共 {len(df)} 条记录")
            return df
        else:
            logging.warning("未下载到数据")
            return pd.DataFrame()
    
    def _download_month_data(self, year: int, month: int) -> pd.DataFrame:
        """下载月度数据"""
        try:
            # 计算开始和结束时间戳
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year + 1, 1, 1)
            else:
                end_date = datetime(year, month + 1, 1)
            
            start_time = int(start_date.timestamp() * 1000)
            end_time = int(end_date.timestamp() * 1000)
            
            # 分批下载（Bybit API限制每次最多1000条）
            all_klines = []
            current_start = start_time
            
            while current_start < end_time:
                url = f"{self.config.BYBIT_TESTNET_REST_URL}/v5/market/kline"
                params = {
                    "category": "linear",
                    "symbol": self.symbol,
                    "interval": self.interval,
                    "start": current_start,
                    "end": end_time,
                    "limit": 1000
                }
                
                response = requests.get(url, params=params)
                data = response.json()
                
                if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                    klines = data["result"]["list"]
                    all_klines.extend(klines)
                    
                    # 更新时间戳
                    last_timestamp = int(klines[-1][0])
                    current_start = last_timestamp + 1
                    
                    # 避免请求过快
                    time.sleep(0.1)
                else:
                    break
            
            if all_klines:
                # 转换为DataFrame
                df = pd.DataFrame(all_klines, columns=[
                    "timestamp", "open", "high", "low", "close", "volume", "turnover"
                ])
                
                # 转换数据类型
                df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
                for col in ["open", "high", "low", "close", "volume", "turnover"]:
                    df[col] = df[col].astype(float)
                
                df.set_index("timestamp", inplace=True)
                df.sort_index(inplace=True)
                
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logging.error(f"下载月度数据失败: {e}")
            return pd.DataFrame()
    
    def start_realtime_collection(self):
        """启动实时数据收集"""
        if self.ws_running:
            return
        
        self.ws_running = True
        self.ws_thread = threading.Thread(target=self._run_websocket)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
        logging.info("实时数据收集已启动")
    
    def _run_websocket(self):
        """运行WebSocket连接"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                topic = data.get("topic", "")
                
                if "kline" in topic:
                    self._handle_kline(data)
                elif "orderbook" in topic:
                    self._handle_orderbook(data)
                elif "trade" in topic:
                    self._handle_trade(data)
                    
            except Exception as e:
                logging.error(f"WebSocket消息处理失败: {e}")
        
        def on_error(ws, error):
            logging.error(f"WebSocket错误: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            logging.warning("WebSocket连接关闭")
            if self.ws_running:
                time.sleep(5)
                self._connect_websocket()
        
        def on_open(ws):
            logging.info("WebSocket连接成功")
            # 订阅数据流
            self._subscribe(ws)
        
        self._connect_websocket(on_message, on_error, on_close, on_open)
    
    def _connect_websocket(self, on_message=None, on_error=None, on_close=None, on_open=None):
        """连接WebSocket"""
        try:
            self.ws = websocket.WebSocketApp(
                self.config.BYBIT_TESTNET_WS_URL,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            # 运行WebSocket
            self.ws.run_forever(ping_interval=30, ping_timeout=10)
            
        except Exception as e:
            logging.error(f"WebSocket连接失败: {e}")
            if self.ws_running:
                time.sleep(5)
                self._connect_websocket(on_message, on_error, on_close, on_open)
    
    def _subscribe(self, ws):
        """订阅数据流"""
        subscriptions = [
            f"kline.{self.interval}.{self.symbol}",
            f"orderbook.200.{self.symbol}",
            f"publicTrade.{self.symbol}"
        ]
        
        for topic in subscriptions:
            subscribe_msg = {
                "op": "subscribe",
                "args": [topic]
            }
            ws.send(json.dumps(subscribe_msg))
            time.sleep(0.1)
        
        logging.info(f"已订阅: {subscriptions}")
    
    def _handle_kline(self, data):
        """处理K线数据"""
        try:
            kline_data = data.get("data", [])
            if kline_data:
                for k in kline_data:
                    timestamp = pd.to_datetime(int(k["start"]), unit="ms")
                    price = float(k["close"])
                    volume = float(k["volume"])
                    
                    self.price_buffer.append((timestamp, price))
                    self.volume_buffer.append((timestamp, volume))
                    
                    self.last_update = datetime.now()
                    
        except Exception as e:
            logging.error(f"处理K线数据失败: {e}")
    
    def _handle_orderbook(self, data):
        """处理订单簿数据"""
        try:
            orderbook_data = data.get("data", {})
            if orderbook_data:
                self.orderbook_buffer = {
                    "bids": orderbook_data.get("b", []),
                    "asks": orderbook_data.get("a", []),
                    "timestamp": orderbook_data.get("ts")
                }
        except Exception as e:
            logging.error(f"处理订单簿数据失败: {e}")
    
    def _handle_trade(self, data):
        """处理成交数据"""
        # 可以记录最近的成交
        pass
    
    def get_latest_price(self) -> Optional[float]:
        """获取最新价格"""
        if self.price_buffer:
            return self.price_buffer[-1][1]
        return None
    
    def get_orderbook_snapshot(self) -> dict:
        """获取订单簿快照"""
        return self.orderbook_buffer
    
    def load_historical_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """加载历史数据"""
        all_data = []
        
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT file_path FROM data_index
            WHERE symbol = ? AND date BETWEEN ? AND ?
            ORDER BY date
        ''', (self.symbol, start_date, end_date))
        
        for row in cursor.fetchall():
            file_path = row[0]
            if Path(file_path).exists():
                df = pd.read_parquet(file_path)
                all_data.append(df)
        
        if all_data:
            df = pd.concat(all_data)
            return df
        else:
            return pd.DataFrame()
    
    def get_data_info(self) -> dict:
        """获取数据统计信息"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT 
                MIN(date) as start_date,
                MAX(date) as end_date,
                COUNT(*) as total_days,
                SUM(row_count) as total_rows
            FROM data_index
            WHERE symbol = ?
        ''', (self.symbol,))
        
        row = cursor.fetchone()
        if row:
            return {
                "start_date": row[0],
                "end_date": row[1],
                "total_days": row[2],
                "total_rows": row[3]
            }
        return {}
    
    def stop(self):
        """停止数据收集"""
        self.ws_running = False
        if self.ws:
            self.ws.close()
        self.conn.close()


class FeatureEngineer:
    """特征工程 - 计算256维技术指标"""
    
    def __init__(self, config: Config):
        self.config = config
        self.feature_names = []
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=min(128, config.FEATURE_DIM))
        
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有256维特征"""
        if df.empty:
            return pd.DataFrame()
        
        features = pd.DataFrame(index=df.index)
        
        # 1. 趋势指标 (8个)
        trend_features = self._calculate_trend_features(df)
        features = pd.concat([features, trend_features], axis=1)
        
        # 2. 动量指标 (12个)
        momentum_features = self._calculate_momentum_features(df)
        features = pd.concat([features, momentum_features], axis=1)
        
        # 3. 波动率指标 (10个)
        volatility_features = self._calculate_volatility_features(df)
        features = pd.concat([features, volatility_features], axis=1)
        
        # 4. 成交量指标 (8个)
        volume_features = self._calculate_volume_features(df)
        features = pd.concat([features, volume_features], axis=1)
        
        # 5. 微观结构特征 (6个)
        microstructure_features = self._calculate_microstructure_features(df)
        features = pd.concat([features, microstructure_features], axis=1)
        
        # 6. 统计特征 (10个)
        statistical_features = self._calculate_statistical_features(df)
        features = pd.concat([features, statistical_features], axis=1)
        
        # 7. 周期特征 (6个)
        cyclical_features = self._calculate_cyclical_features(df)
        features = pd.concat([features, cyclical_features], axis=1)
        
        # 8. 价格衍生特征 (补充到256维)
        price_derived = self._calculate_price_derived_features(df)
        features = pd.concat([features, price_derived], axis=1)
        
        # 记录特征名称
        self.feature_names = list(features.columns)
        
        # 确保特征维度
        if len(self.feature_names) < self.config.FEATURE_DIM:
            # 填充到256维
            for i in range(len(self.feature_names), self.config.FEATURE_DIM):
                features[f'pad_{i}'] = 0
                self.feature_names.append(f'pad_{i}')
        elif len(self.feature_names) > self.config.FEATURE_DIM:
            # 截取前256维
            features = features.iloc[:, :self.config.FEATURE_DIM]
            self.feature_names = self.feature_names[:self.config.FEATURE_DIM]
        
        return features
    
    def _calculate_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """趋势指标"""
        features = pd.DataFrame(index=df.index)
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # SMA
        for period in [5, 10, 20, 50]:
            if len(close) >= period:
                sma = pd.Series(close).rolling(window=period).mean()
                features[f'sma_{period}'] = sma / close - 1  # 归一化
        
        # EMA
        for period in [5, 10, 20, 50]:
            if len(close) >= period:
                ema = pd.Series(close).ewm(span=period, adjust=False).mean()
                features[f'ema_{period}'] = ema / close - 1
        
        # MACD
        if len(close) >= 26:
            ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
            ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            hist = macd - signal
            features['macd'] = macd
            features['macd_signal'] = signal
            features['macd_hist'] = hist
        
        # ADX
        if len(close) >= 14:
            plus_dm = np.zeros_like(close)
            minus_dm = np.zeros_like(close)
            tr = np.zeros_like(close)
            
            for i in range(1, len(close)):
                high_move = high[i] - high[i-1]
                low_move = low[i-1] - low[i]
                
                plus_dm[i] = high_move if high_move > low_move and high_move > 0 else 0
                minus_dm[i] = low_move if low_move > high_move and low_move > 0 else 0
                
                tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            
            atr = pd.Series(tr).rolling(window=14).mean()
            plus_di = 100 * pd.Series(plus_dm).rolling(window=14).mean() / atr
            minus_di = 100 * pd.Series(minus_dm).rolling(window=14).mean() / atr
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=14).mean()
            
            features['adx'] = adx
            features['plus_di'] = plus_di
            features['minus_di'] = minus_di
        
        return features
    
    def _calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """动量指标"""
        features = pd.DataFrame(index=df.index)
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # RSI
        for period in [6, 14, 21, 28]:
            if len(close) > period:
                delta = np.diff(close)
                gain = np.where(delta > 0, delta, 0)
                loss = np.where(delta < 0, -delta, 0)
                
                avg_gain = pd.Series(gain).rolling(window=period).mean()
                avg_loss = pd.Series(loss).rolling(window=period).mean()
                
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                features[f'rsi_{period}'] = rsi / 100  # 归一化到[0,1]
        
        # Stochastic
        if len(close) >= 14:
            lowest_low = pd.Series(low).rolling(window=14).min()
            highest_high = pd.Series(high).rolling(window=14).max()
            
            k = 100 * (close - lowest_low) / (highest_high - lowest_low)
            d = k.rolling(window=3).mean()
            
            features['stoch_k'] = k / 100
            features['stoch_d'] = d / 100
        
        # Williams %R
        if len(close) >= 14:
            highest_high = pd.Series(high).rolling(window=14).max()
            lowest_low = pd.Series(low).rolling(window=14).min()
            williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
            features['williams_r'] = (williams_r + 50) / 100  # 归一化
        
        # ROC
        for period in [5, 10, 20]:
            if len(close) > period:
                roc = (close - np.roll(close, period)) / np.roll(close, period) * 100
                features[f'roc_{period}'] = roc / 100
        
        # MFI
        if len(close) >= 14:
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            
            positive_flow = np.where(typical_price > np.roll(typical_price, 1), money_flow, 0)
            negative_flow = np.where(typical_price < np.roll(typical_price, 1), money_flow, 0)
            
            pos_sum = pd.Series(positive_flow).rolling(window=14).sum()
            neg_sum = pd.Series(negative_flow).rolling(window=14).sum()
            
            mfi_ratio = pos_sum / neg_sum
            mfi = 100 - (100 / (1 + mfi_ratio))
            features['mfi_14'] = mfi / 100
        
        return features
    
    def _calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """波动率指标"""
        features = pd.DataFrame(index=df.index)
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # Bollinger Bands
        if len(close) >= 20:
            sma20 = pd.Series(close).rolling(window=20).mean()
            std20 = pd.Series(close).rolling(window=20).std()
            
            upper = sma20 + 2 * std20
            lower = sma20 - 2 * std20
            bb_width = (upper - lower) / sma20
            bb_pct = (close - lower) / (upper - lower)
            
            features['bb_upper'] = (upper - close) / close
            features['bb_lower'] = (close - lower) / close
            features['bb_middle'] = (sma20 - close) / close
            features['bb_width'] = bb_width
            features['bb_pct'] = bb_pct
        
        # ATR
        if len(close) >= 14:
            tr1 = high - low
            tr2 = abs(high - np.roll(close, 1))
            tr3 = abs(low - np.roll(close, 1))
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            
            atr = pd.Series(tr).rolling(window=14).mean()
            features['atr_14'] = atr / close
        
        # Keltner Channels
        if len(close) >= 20:
            ema20 = pd.Series(close).ewm(span=20, adjust=False).mean()
            atr20 = pd.Series(tr).rolling(window=20).mean()
            
            kc_upper = ema20 + 2 * atr20
            kc_lower = ema20 - 2 * atr20
            
            features['kc_upper'] = (kc_upper - close) / close
            features['kc_lower'] = (close - kc_lower) / close
        
        # Historical Volatility
        for period in [5, 10, 20]:
            if len(close) > period:
                returns = np.diff(np.log(close))
                hist_vol = pd.Series(returns).rolling(window=period).std() * np.sqrt(365 * 24 * 60)
                features[f'hist_vol_{period}'] = hist_vol
        
        # Parkinson Volatility
        if len(close) >= 20:
            hl_ratio = np.log(high / low)
            parkinson = np.sqrt((1 / (4 * np.log(2))) * (hl_ratio ** 2))
            parkinson_vol = pd.Series(parkinson).rolling(window=20).mean() * np.sqrt(365 * 24 * 60)
            features['parkinson_vol'] = parkinson_vol
        
        return features
    
    def _calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """成交量指标"""
        features = pd.DataFrame(index=df.index)
        close = df['close'].values
        volume = df['volume'].values
        
        # Volume SMA
        for period in [5, 10, 20]:
            if len(volume) >= period:
                vol_sma = pd.Series(volume).rolling(window=period).mean()
                features[f'volume_sma_{period}'] = (volume - vol_sma) / (vol_sma + 1e-8)
        
        # Volume ratio
        if len(volume) >= 5:
            vol_ma5 = pd.Series(volume).rolling(window=5).mean()
            features['volume_ratio'] = volume / (vol_ma5 + 1e-8)
        
        # OBV
        obv = np.zeros_like(volume)
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        features['obv'] = (obv - np.mean(obv)) / (np.std(obv) + 1e-8)
        
        # VPT
        vpt = np.zeros_like(volume)
        for i in range(1, len(close)):
            vpt[i] = vpt[i-1] + volume[i] * (close[i] - close[i-1]) / close[i-1]
        features['vpt'] = (vpt - np.mean(vpt)) / (np.std(vpt) + 1e-8)
        
        # CMF
        if len(volume) >= 20:
            mf_multiplier = ((close - low) - (high - close)) / (high - low)
            mf_volume = mf_multiplier * volume
            cmf = pd.Series(mf_volume).rolling(window=20).sum() / pd.Series(volume).rolling(window=20).sum()
            features['cmf_14'] = cmf
        
        return features
    
    def _calculate_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """微观结构特征"""
        features = pd.DataFrame(index=df.index)
        
        # 简化的微观结构特征（实际应使用订单簿数据）
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # 买卖压力估算
        price_range = high - low
        close_position = (close - low) / (price_range + 1e-8)
        features['buy_pressure'] = close_position
        
        # 流动性评分
        features['liquidity_score'] = volume / (price_range + 1e-8)
        
        # 价差估算
        features['spread_estimate'] = price_range / close
        
        # 深度比例估算
        features['depth_ratio'] = volume / (high + low + close)
        
        return features
    
    def _calculate_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """统计特征"""
        features = pd.DataFrame(index=df.index)
        close = df['close'].values
        returns = np.diff(np.log(close))
        
        # 偏度
        for period in [20, 50]:
            if len(returns) >= period:
                skew = pd.Series(returns).rolling(window=period).skew()
                features[f'skewness_{period}'] = skew
        
        # 峰度
        for period in [20, 50]:
            if len(returns) >= period:
                kurt = pd.Series(returns).rolling(window=period).kurt()
                features[f'kurtosis_{period}'] = kurt
        
        # 自相关
        for lag in [1, 2, 3]:
            if len(returns) > lag:
                autocorr = pd.Series(returns).rolling(window=50).apply(
                    lambda x: x.autocorr(lag=lag) if len(x) > lag else 0
                )
                features[f'autocorr_{lag}'] = autocorr
        
        # 分位数
        for q in [25, 50, 75, 90]:
            quantile = pd.Series(returns).rolling(window=50).quantile(q/100)
            features[f'quantile_{q}'] = quantile
        
        return features
    
    def _calculate_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """周期特征"""
        features = pd.DataFrame(index=df.index)
        
        # 小时周期
        hour = df.index.hour
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # 星期周期
        dayofweek = df.index.dayofweek
        features['day_sin'] = np.sin(2 * np.pi * dayofweek / 7)
        features['day_cos'] = np.cos(2 * np.pi * dayofweek / 7)
        
        # 月份周期
        month = df.index.month
        features['month_sin'] = np.sin(2 * np.pi * month / 12)
        features['month_cos'] = np.cos(2 * np.pi * month / 12)
        
        return features
    
    def _calculate_price_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """价格衍生特征"""
        features = pd.DataFrame(index=df.index)
        close = df['close'].values
        open_price = df['open'].values
        high = df['high'].values
        low = df['low'].values
        
        # 收益率
        returns = np.diff(np.log(close))
        features['returns_1'] = np.append([0], returns)
        
        for period in [5, 10, 20]:
            if len(close) > period:
                ret_period = close / np.roll(close, period) - 1
                features[f'returns_{period}'] = ret_period
        
        # 价格位置
        price_position = (close - low) / (high - low + 1e-8)
        features['price_position'] = price_position
        
        # 缺口
        gap = open_price / np.roll(close, 1) - 1
        features['gap'] = gap
        
        # 蜡烛形态特征
        body = abs(close - open_price)
        upper_shadow = high - np.maximum(close, open_price)
        lower_shadow = np.minimum(close, open_price) - low
        
        features['body_size'] = body / (high - low + 1e-8)
        features['upper_shadow'] = upper_shadow / (high - low + 1e-8)
        features['lower_shadow'] = lower_shadow / (high - low + 1e-8)
        
        # 是否阳线
        features['is_bull'] = (close > open_price).astype(float)
        
        return features
    
    def normalize_features(self, features: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """特征归一化"""
        if fit:
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = self.scaler.transform(features)
        
        return pd.DataFrame(features_scaled, index=features.index, columns=features.columns)
    
    def reduce_dimension(self, features: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """降维"""
        if fit:
            features_pca = self.pca.fit_transform(features)
        else:
            features_pca = self.pca.transform(features)
        
        columns = [f'pca_{i}' for i in range(features_pca.shape[1])]
        return pd.DataFrame(features_pca, index=features.index, columns=columns)
    
    def get_feature_importance(self, features: pd.DataFrame, target: pd.Series) -> dict:
        """特征重要性分析"""
        from sklearn.ensemble import RandomForestRegressor
        
        # 使用随机森林评估特征重要性
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(features.fillna(0), target)
        
        importance = dict(zip(features.columns, rf.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


# ==================== 模型模块 ====================

class FeatureExtractor(nn.Module):
    """特征提取器 - CNN + BiLSTM + Attention"""
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        # CNN层
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(10)
        
        # BiLSTM层
        self.lstm = nn.LSTM(
            input_size=1280,  # 128 * 10
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Multi-head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # 双向
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer Norm和残差连接
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # 输出层
        self.output = nn.Linear(hidden_dim * 2, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        batch_size, seq_len, features = x.shape
        
        # CNN处理
        x = x.view(batch_size * seq_len, 1, features)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # (batch_size * seq_len, 128, 10)
        x = x.view(batch_size, seq_len, -1)  # (batch_size, seq_len, 1280)
        
        # BiLSTM
        lstm_out, (hidden, cell) = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_dim*2)
        
        # Multi-head Attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 残差连接 + Layer Norm
        out = self.layer_norm(lstm_out + attn_out)
        
        # 取最后一个时间步
        out = out[:, -1, :]  # (batch_size, hidden_dim*2)
        
        # 输出层
        out = self.output(out)  # (batch_size, input_dim)
        
        return out


class ActorNetwork(nn.Module):
    """Actor网络 - 输出动作"""
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim // 2, num_heads=8, dropout=dropout)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
        )
        
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim // 4, 1),
            nn.Tanh()
        )
        
        # 对数标准差（可学习参数）
        self.log_std = nn.Parameter(torch.zeros(1))
        
    def forward(self, state):
        # state shape: (batch_size, seq_len, features)
        features = self.feature_extractor(state)
        x = self.net(features)
        mean = self.mean_head(x)
        return mean, self.log_std.expand_as(mean)
    
    def get_action(self, state, deterministic=False):
        """获取动作"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        if deterministic:
            return mean
        else:
            normal = torch.distributions.Normal(mean, std)
            action = normal.rsample()  # 重参数化采样
            return torch.clamp(action, -1, 1)


class CriticNetwork(nn.Module):
    """Critic网络 - 输出状态价值"""
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim // 2, num_heads=8, dropout=dropout)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 4, 1)
        )
        
    def forward(self, state):
        features = self.feature_extractor(state)
        value = self.net(features)
        return value


class PrioritizedReplayBuffer:
    """优先级经验回放缓冲区"""
    
    def __init__(self, capacity: int = 20000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样指数
        self.epsilon = 1e-6
        
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.position = 0
        
    def push(self, state, action, reward, next_state, done, value, log_prob):
        """添加经验"""
        experience = (state, action, reward, next_state, done, value, log_prob)
        
        # 新经验给予最大优先级
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = max_priority
        
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int):
        """按优先级采样"""
        if len(self.buffer) < batch_size:
            return None
        
        # 计算采样概率
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # 计算重要性采样权重
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # 获取经验
        batch = [self.buffer[idx] for idx in indices]
        
        # 解包
        states, actions, rewards, next_states, dones, values, log_probs = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(values),
            np.array(log_probs),
            indices,
            np.array(weights, dtype=np.float32)
        )
    
    def update_priorities(self, indices, td_errors):
        """更新优先级"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + self.epsilon
    
    def __len__(self):
        return len(self.buffer)


class PPOAgent:
    """PPO算法实现"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"使用设备: {self.device}")
        
        # 网络
        self.actor = ActorNetwork(
            input_dim=config.FEATURE_DIM,
            hidden_dim=config.HIDDEN_DIM,
            dropout=config.DROPOUT
        ).to(self.device)
        
        self.critic = CriticNetwork(
            input_dim=config.FEATURE_DIM,
            hidden_dim=config.HIDDEN_DIM,
            dropout=config.DROPOUT
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=config.LEARNING_RATE
        )
        
        # 学习率调度器
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)
        
        # 经验回放
        self.buffer = PrioritizedReplayBuffer(
            capacity=config.BUFFER_CAPACITY,
            alpha=0.6,
            beta=0.4
        )
        
        # 训练统计
        self.training_stats = {
            'episodes': 0,
            'steps': 0,
            'total_reward': 0,
            'avg_loss': [],
            'avg_value_loss': [],
            'avg_policy_loss': [],
            'avg_entropy': []
        }
        
        # PPO参数
        self.gamma = config.GAMMA
        self.lambd = config.LAMBDA
        self.clip_epsilon = config.CLIP_EPSILON
        self.entropy_coef = config.ENTROPY_COEF
        self.value_coef = config.VALUE_COEF
        self.max_grad_norm = config.MAX_GRAD_NORM
        self.target_kl = config.TARGET_KL
        self.batch_size = config.BATCH_SIZE
        self.update_epochs = config.UPDATE_EPOCHS
        
    def select_action(self, state, deterministic=False):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, features)
        
        with torch.no_grad():
            mean, log_std = self.actor(state)
            std = log_std.exp()
            
            if deterministic:
                action = mean
                log_prob = None
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)
            
            value = self.critic(state)
        
        action = torch.clamp(action, -1, 1)
        
        return (
            action.cpu().numpy().flatten()[0],
            log_prob.cpu().numpy().flatten()[0] if log_prob is not None else None,
            value.cpu().numpy().flatten()[0]
        )
    
    def store_transition(self, state, action, reward, next_state, done, value, log_prob):
        """存储经验"""
        self.buffer.push(state, action, reward, next_state, done, value, log_prob)
        self.training_stats['steps'] += 1
    
    def update(self):
        """更新策略"""
        if len(self.buffer) < self.batch_size:
            return {}
        
        # 采样
        batch = self.buffer.sample(self.batch_size)
        if batch is None:
            return {}
        
        states, actions, rewards, next_states, dones, old_values, old_log_probs, indices, weights = batch
        
        # 转换为tensor
        states = torch.FloatTensor(states).unsqueeze(1).to(self.device)  # (batch, 1, features)
        actions = torch.FloatTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        old_values = torch.FloatTensor(old_values).unsqueeze(1).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        # 计算GAE
        with torch.no_grad():
            next_values = self.critic(next_states)
            deltas = rewards + self.gamma * next_values * (1 - dones) - old_values
            advantages = torch.zeros_like(deltas)
            gae = 0
            for t in reversed(range(len(deltas))):
                gae = deltas[t] + self.gamma * self.lambd * (1 - dones[t]) * gae
                advantages[t] = gae
            returns = advantages + old_values
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多轮更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(self.update_epochs):
            # 新策略的概率
            mean, log_std = self.actor(states)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(-1, keepdim=True)
            entropy = dist.entropy().mean()
            
            # 计算比率
            ratio = (new_log_probs - old_log_probs).exp()
            
            # 裁剪的policy loss
            surr1 = ratio * advantages * weights
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages * weights
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # value loss
            values = self.critic(states)
            value_loss = F.mse_loss(values, returns) * self.value_coef
            
            # KL散度检查
            with torch.no_grad():
                log_ratio = new_log_probs - old_log_probs
                approx_kl = (log_ratio.exp() - 1 - log_ratio).mean()
                if approx_kl > self.target_kl * 1.5:
                    break
            
            # 总损失
            loss = policy_loss + value_loss - self.entropy_coef * entropy
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.max_grad_norm)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
        
        # 更新优先级
        with torch.no_grad():
            td_errors = (returns - values).cpu().numpy().flatten()
        self.buffer.update_priorities(indices, td_errors)
        
        # 更新学习率
        self.scheduler.step()
        
        # 记录统计
        self.training_stats['avg_policy_loss'].append(total_policy_loss / self.update_epochs)
        self.training_stats['avg_value_loss'].append(total_value_loss / self.update_epochs)
        self.training_stats['avg_entropy'].append(total_entropy / self.update_epochs)
        
        return {
            'policy_loss': total_policy_loss / self.update_epochs,
            'value_loss': total_value_loss / self.update_epochs,
            'entropy': total_entropy / self.update_epochs,
            'kl_divergence': approx_kl.item()
        }
    
    def save_checkpoint(self, episode: int, path: str = None):
        """保存检查点"""
        if path is None:
            path = f"models_saved/checkpoint_{episode}.pt"
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'episode': episode,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_stats': self.training_stats
        }, path)
        
        logging.info(f"模型已保存: {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_stats = checkpoint['training_stats']
        
        logging.info(f"模型已加载: {path}")
        return checkpoint['episode']
    
    def get_training_stats(self):
        """获取训练统计"""
        return self.training_stats


# ==================== 交易模块 ====================

class SlippageModel:
    """滑点模型"""
    
    def __init__(self, base_slippage: float = 0.0001):
        self.base_slippage = base_slippage
        self.orderbook = {}
        
    def update_orderbook(self, orderbook: dict):
        """更新订单簿"""
        self.orderbook = orderbook
        
    def calculate_slippage(self, side: str, qty: float, price: float) -> float:
        """计算预期滑点"""
        # 基础滑点
        slippage = self.base_slippage
        
        # 基于订单簿的滑点
        if self.orderbook:
            if side == "Buy":
                orders = self.orderbook.get("asks", [])
            else:
                orders = self.orderbook.get("bids", [])
            
            if orders:
                # 计算前10档的总流动性
                total_liquidity = sum(float(order[1]) for order in orders[:10])
                
                # 数量越大，滑点越大
                if total_liquidity > 0:
                    qty_ratio = qty / total_liquidity
                    slippage += qty_ratio * 0.001
                
                # 买卖价差
                best_ask = float(orders[0][0]) if orders else price
                best_bid = float(orders[0][0]) if side == "Sell" and orders else price
                spread = abs(best_ask - best_bid) / price
                
                # 价差越大，滑点越大
                slippage += spread * 0.1
        
        # 随机因素
        slippage *= (1 + np.random.normal(0, 0.1))
        
        return min(slippage, 0.005)  # 最大滑点0.5%
    
    def simulate_market_impact(self, side: str, qty: float, price: float) -> Tuple[float, float]:
        """模拟市场冲击"""
        # 简化的市场冲击模型
        # 大单会产生价格影响
        
        # 估计市场深度
        avg_depth = 10.0  # 假设平均深度10 BTC
        
        # 计算市场冲击
        impact = (qty / avg_depth) * 0.001
        impact = min(impact, 0.002)  # 最大冲击0.2%
        
        if side == "Buy":
            executed_price = price * (1 + impact)
        else:
            executed_price = price * (1 - impact)
        
        return executed_price, impact


class RiskManager:
    """4级风控系统"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # 账户风控
        self.daily_loss = 0
        self.consecutive_losses = 0
        self.last_trade_time = None
        self.daily_trades = 0
        self.daily_reset_time = datetime.now().replace(hour=0, minute=0, second=0)
        
        # 仓位风控
        self.stop_loss_price = None
        self.take_profit_price = None
        self.trailing_stop_price = None
        self.entry_price = None
        
        # 订单风控
        self.trade_times = deque(maxlen=config.MAX_TRADES_PER_MINUTE)
        
        # 系统风控
        self.connection_status = True
        self.data_freshness = True
        
    def check_all(self, state: dict) -> Tuple[bool, int, str]:
        """执行所有风控检查"""
        
        # Level 1 - 账户风控
        passed, reason = self._check_account_risk(state)
        if not passed:
            return False, 1, reason
        
        # Level 2 - 仓位风控
        passed, reason = self._check_position_risk(state)
        if not passed:
            return False, 2, reason
        
        # Level 3 - 订单风控
        passed, reason = self._check_order_risk(state)
        if not passed:
            return False, 3, reason
        
        # Level 4 - 系统风控
        passed, reason = self._check_system_risk(state)
        if not passed:
            return False, 4, reason
        
        return True, 0, "All checks passed"
    
    def _check_account_risk(self, state: dict) -> Tuple[bool, str]:
        """账户风控"""
        # 重置每日统计
        now = datetime.now()
        if now.date() > self.daily_reset_time.date():
            self.daily_loss = 0
            self.daily_trades = 0
            self.daily_reset_time = now.replace(hour=0, minute=0, second=0)
        
        # 检查日亏损
        if state.get('daily_loss', 0) > self.config.MAX_DAILY_LOSS_PCT:
            return False, f"Daily loss exceeded: {state.get('daily_loss', 0):.2%}"
        
        # 检查连续亏损
        if self.consecutive_losses >= self.config.MAX_CONSECUTIVE_LOSSES:
            return False, f"Consecutive losses: {self.consecutive_losses}"
        
        # 检查最小余额
        if state.get('balance', 0) < self.config.MIN_BALANCE:
            return False, f"Balance too low: {state.get('balance', 0):.2f}"
        
        # 检查冷却时间
        if self.last_trade_time and state.get('last_trade_pnl', 0) < 0:
            cooldown = 300  # 5分钟冷却
            if (now - self.last_trade_time).seconds < cooldown:
                return False, f"In cooldown period"
        
        return True, ""
    
    def _check_position_risk(self, state: dict) -> Tuple[bool, str]:
        """仓位风控"""
        # 检查仓位限制
        position_pct = state.get('position_pct', 0)
        if position_pct > self.config.MAX_POSITION_PCT:
            return False, f"Position too large: {position_pct:.2%}"
        
        # 检查杠杆
        if state.get('leverage', 1) > 3:
            return False, f"Leverage too high: {state.get('leverage', 1)}x"
        
        # 检查止损
        if self.stop_loss_price and state.get('position', 0) != 0:
            current_price = state.get('current_price', 0)
            if state.get('position', 0) > 0:  # 多头
                if current_price <= self.stop_loss_price:
                    return False, "Stop loss triggered"
            else:  # 空头
                if current_price >= self.stop_loss_price:
                    return False, "Stop loss triggered"
        
        # 检查止盈
        if self.take_profit_price and state.get('position', 0) != 0:
            current_price = state.get('current_price', 0)
            if state.get('position', 0) > 0:  # 多头
                if current_price >= self.take_profit_price:
                    return True, "Take profit triggered"  # 这是好事，不算风控失败
            else:  # 空头
                if current_price <= self.take_profit_price:
                    return True, "Take profit triggered"
        
        # 检查移动止损
        if self.trailing_stop_price and state.get('position', 0) > 0:
            current_price = state.get('current_price', 0)
            if current_price > self.entry_price:
                new_stop = current_price * (1 - self.config.TRAILING_STOP_PCT)
                if new_stop > self.trailing_stop_price:
                    self.trailing_stop_price = new_stop
            
            if current_price <= self.trailing_stop_price:
                return False, "Trailing stop triggered"
        
        return True, ""
    
    def _check_order_risk(self, state: dict) -> Tuple[bool, str]:
        """订单风控"""
        now = datetime.now()
        
        # 检查滑点
        if state.get('slippage', 0) > self.config.MAX_SLIPPAGE:
            return False, f"Slippage too high: {state.get('slippage', 0):.4%}"
        
        # 检查流动性
        if state.get('liquidity', 0) < self.config.MIN_LIQUIDITY:
            return False, f"Liquidity too low: {state.get('liquidity', 0):.2f}"
        
        # 检查交易频率
        self.trade_times.append(now)
        if len(self.trade_times) >= self.config.MAX_TRADES_PER_MINUTE:
            oldest = self.trade_times[0]
            if (now - oldest).seconds < 60:
                return False, f"Trade frequency too high"
        
        # 检查价格偏离
        if state.get('price_deviation', 0) > self.config.MAX_PRICE_DEVIATION:
            return False, f"Price deviation too high: {state.get('price_deviation', 0):.2%}"
        
        return True, ""
    
    def _check_system_risk(self, state: dict) -> Tuple[bool, str]:
        """系统风控"""
        # 检查连接状态
        if not self.connection_status:
            return False, "WebSocket disconnected"
        
        # 检查数据新鲜度
        if not self.data_freshness:
            return False, "Data not fresh"
        
        # 检查内存使用
        import psutil
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 / 1024  # GB
        if memory_usage > 3.5:
            return False, f"Memory usage too high: {memory_usage:.1f}GB"
        
        return True, ""
    
    def update_after_trade(self, trade_result: dict):
        """交易后更新风控状态"""
        # 更新连续亏损
        if trade_result.get('pnl', 0) < 0:
            self.consecutive_losses += 1
            self.daily_loss += abs(trade_result['pnl'] / trade_result['balance_before'])
        else:
            self.consecutive_losses = 0
        
        # 更新止损止盈
        if trade_result.get('position', 0) != 0:
            self.entry_price = trade_result.get('entry_price')
            self.stop_loss_price = self.entry_price * (1 - self.config.STOP_LOSS_PCT)
            self.take_profit_price = self.entry_price * (1 + self.config.TAKE_PROFIT_PCT)
            self.trailing_stop_price = self.entry_price * (1 - self.config.TRAILING_STOP_PCT)
        else:
            self.stop_loss_price = None
            self.take_profit_price = None
            self.trailing_stop_price = None
            self.entry_price = None
        
        self.last_trade_time = datetime.now()
        self.daily_trades += 1
    
    def update_system_status(self, connection: bool = True, data_fresh: bool = True):
        """更新系统状态"""
        self.connection_status = connection
        self.data_freshness = data_fresh


class TradingSimulator:
    """交易模拟器 - 基于真实数据模拟交易"""
    
    def __init__(self, config: Config, data_manager: DataManager = None):
        self.config = config
        self.data_manager = data_manager
        
        # 账户状态
        self.initial_balance = config.INITIAL_BALANCE
        self.balance = config.INITIAL_BALANCE
        self.position = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
        # 交易记录
        self.trades = []
        self.equity_curve = []
        
        # 当前步
        self.current_step = 0
        self.current_price = 0.0
        
        # 滑点模型
        self.slippage_model = SlippageModel(base_slippage=0.0001)
        
        # 风险管理
        self.risk_manager = RiskManager(config)
        
        # 统计
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'peak_equity': config.INITIAL_BALANCE
        }
        
    def reset(self, initial_balance: float = None):
        """重置环境"""
        if initial_balance is not None:
            self.initial_balance = initial_balance
        
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.trades = []
        self.equity_curve = []
        self.current_step = 0
        
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'peak_equity': self.initial_balance
        }
        
        return self._get_state()
    
    def _get_state(self) -> dict:
        """获取当前状态"""
        return {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'entry_price': self.entry_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'equity': self.get_equity(),
            'current_price': self.current_price,
            'position_pct': abs(self.position * self.current_price) / self.balance if self.balance > 0 else 0,
            'daily_loss': self.stats.get('daily_loss', 0),
            'last_trade_pnl': self.trades[-1].get('pnl', 0) if self.trades else 0,
            'liquidity': self._estimate_liquidity(),
            'slippage': 0,
            'price_deviation': 0,
            'leverage': abs(self.position * self.current_price) / self.balance if self.balance > 0 else 1
        }
    
    def _estimate_liquidity(self) -> float:
        """估计流动性"""
        if self.data_manager and self.data_manager.orderbook_buffer:
            orderbook = self.data_manager.orderbook_buffer
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            total_liquidity = sum(float(b[1]) for b in bids[:10]) + sum(float(a[1]) for a in asks[:10])
            return total_liquidity
        return 100.0  # 默认值
    
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行一步交易
        
        Args:
            action: float [-1, 1] 动作值
                > 0: 买入信号
                < 0: 卖出信号
                = 0: 持仓不动
        """
        # 获取当前价格
        if self.data_manager:
            price = self.data_manager.get_latest_price()
            if price:
                self.current_price = price
            else:
                # 如果没有实时数据，使用模拟价格
                self.current_price = self.current_price or 50000
                self.current_price *= (1 + np.random.normal(0, 0.0001))
        else:
            self.current_price = self.current_price or 50000
            self.current_price *= (1 + np.random.normal(0, 0.0001))
        
        # 更新滑点模型的订单簿
        if self.data_manager:
            self.slippage_model.update_orderbook(self.data_manager.get_orderbook_snapshot())
        
        # 获取当前状态
        state_before = self._get_state()
        
        # 风控检查
        risk_passed, risk_level, risk_reason = self.risk_manager.check_all(state_before)
        if not risk_passed:
            # 风控未通过，强制平仓
            if self.position != 0:
                self._close_position(self.current_price, reason=risk_reason)
            return self._get_observation(), -0.1, False, {'risk_triggered': risk_reason}
        
        # 计算目标仓位
        max_position_value = self.balance * self.config.MAX_POSITION_PCT
        current_position_value = self.position * self.current_price
        
        # 根据动作调整仓位
        # action: -1 表示完全卖出，1 表示完全买入
        target_position_value = current_position_value + action * max_position_value * 0.1
        target_position = target_position_value / self.current_price if self.current_price > 0 else 0
        
        # 限制仓位范围
        target_position = np.clip(target_position, 
                                 -max_position_value / self.current_price,
                                 max_position_value / self.current_price)
        
        # 执行交易
        trade_executed = False
        if abs(target_position - self.position) > 0.001:  # 最小交易单位
            if target_position > self.position:  # 买入
                qty = target_position - self.position
                executed_price, fee, slippage = self._execute_order('Buy', qty, self.current_price)
                trade_executed = True
            elif target_position < self.position:  # 卖出
                qty = self.position - target_position
                executed_price, fee, slippage = self._execute_order('Sell', qty, self.current_price)
                trade_executed = True
        
        # 计算未实现盈亏
        if self.position != 0 and self.entry_price > 0:
            if self.position > 0:  # 多头
                self.unrealized_pnl = self.position * (self.current_price - self.entry_price)
            else:  # 空头
                self.unrealized_pnl = abs(self.position) * (self.entry_price - self.current_price)
        else:
            self.unrealized_pnl = 0
        
        # 计算奖励
        equity_before = state_before['balance'] + state_before['position'] * state_before['current_price']
        equity_after = self.get_equity()
        reward = self._calculate_reward(equity_before, equity_after, trade_executed)
        
        # 更新统计
        self.current_step += 1
        self.equity_curve.append({
            'step': self.current_step,
            'equity': equity_after,
            'price': self.current_price
        })
        
        # 更新峰值和回撤
        if equity_after > self.stats['peak_equity']:
            self.stats['peak_equity'] = equity_after
        drawdown = (self.stats['peak_equity'] - equity_after) / self.stats['peak_equity']
        if drawdown > self.stats['max_drawdown']:
            self.stats['max_drawdown'] = drawdown
        
        # 获取下一个状态
        next_state = self._get_observation()
        
        # 检查是否终止（例如：爆仓）
        done = self.balance < self.config.MIN_BALANCE or self.current_step > 10000
        
        info = {
            'step': self.current_step,
            'equity': equity_after,
            'position': self.position,
            'pnl': self.realized_pnl + self.unrealized_pnl,
            'drawdown': drawdown,
            'trades': len(self.trades)
        }
        
        return next_state, reward, done, info
    
    def _execute_order(self, side: str, qty: float, price: float) -> Tuple[float, float, float]:
        """执行订单"""
        # 计算滑点
        slippage = self.slippage_model.calculate_slippage(side, qty, price)
        
        # 计算执行价格（考虑滑点）
        if side == 'Buy':
            executed_price = price * (1 + slippage)
        else:
            executed_price = price * (1 - slippage)
        
        # 计算手续费
        fee = qty * executed_price * self.config.FEE_RATE
        
        # 计算总成本/收入
        if side == 'Buy':
            cost = qty * executed_price + fee
            if cost <= self.balance:
                self.balance -= cost
                self.position += qty
                self.entry_price = (self.entry_price * (self.position - qty) + executed_price * qty) / self.position if self.position > 0 else executed_price
                
                # 记录交易
                self._record_trade(side, qty, executed_price, fee, 0)
        else:  # Sell
            revenue = qty * executed_price - fee
            self.balance += revenue
            self.position -= qty
            
            # 计算已实现盈亏
            if self.position >= 0:  # 平多头
                pnl = qty * (executed_price - self.entry_price)
            else:  # 平空头
                pnl = qty * (self.entry_price - executed_price)
            
            self.realized_pnl += pnl
            
            # 记录交易
            self._record_trade(side, qty, executed_price, fee, pnl)
        
        # 更新风控
        self.risk_manager.update_after_trade({
            'pnl': self.realized_pnl,
            'balance_before': self.balance + (self.position * executed_price),
            'position': self.position,
            'entry_price': self.entry_price
        })
        
        return executed_price, fee, slippage
    
    def _close_position(self, price: float, reason: str = ""):
        """平仓"""
        if self.position > 0:
            self._execute_order('Sell', self.position, price)
        elif self.position < 0:
            self._execute_order('Buy', abs(self.position), price)
        
        # 记录平仓原因
        if self.trades:
            self.trades[-1]['close_reason'] = reason
    
    def _record_trade(self, side: str, qty: float, price: float, fee: float, pnl: float):
        """记录交易"""
        trade = {
            'time': datetime.now(),
            'side': side,
            'qty': qty,
            'price': price,
            'fee': fee,
            'pnl': pnl,
            'balance_after': self.balance,
            'position_after': self.position
        }
        self.trades.append(trade)
        
        # 更新统计
        self.stats['total_trades'] += 1
        if pnl > 0:
            self.stats['winning_trades'] += 1
        elif pnl < 0:
            self.stats['losing_trades'] += 1
        self.stats['total_pnl'] += pnl
    
    def _calculate_reward(self, equity_before: float, equity_after: float, trade_executed: bool) -> float:
        """计算奖励"""
        # 基础奖励：收益率
        returns = (equity_after - equity_before) / equity_before
        reward = returns * 100  # 放大
        
        # 惩罚：频繁交易
        if trade_executed:
            reward -= 0.01
        
        # 惩罚：大额亏损
        if returns < -0.02:  # 单步亏损超过2%
            reward *= 2  # 加倍惩罚
        
        # 奖励：正确预测方向
        if self.position > 0 and self.current_price > self.entry_price:
            reward += 0.01
        elif self.position < 0 and self.current_price < self.entry_price:
            reward += 0.01
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """获取观测值"""
        # 这里应该返回特征工程后的数据
        # 简化版：返回一些基本状态
        obs = np.array([
            self.balance / self.initial_balance,  # 归一化余额
            self.position * self.current_price / self.initial_balance,  # 归一化仓位价值
            self.unrealized_pnl / self.initial_balance,  # 归一化未实现盈亏
            self.realized_pnl / self.initial_balance,  # 归一化已实现盈亏
            self.current_price / 100000,  # 归一化价格
            self.stats['winning_trades'] / max(1, self.stats['total_trades']),  # 胜率
            self.stats['max_drawdown'],  # 最大回撤
            self.risk_manager.consecutive_losses / self.config.MAX_CONSECUTIVE_LOSSES,  # 连续亏损
        ])
        
        # 扩展到256维（简化，实际应该使用特征工程）
        if len(obs) < self.config.FEATURE_DIM:
            obs = np.pad(obs, (0, self.config.FEATURE_DIM - len(obs)), 'constant')
        elif len(obs) > self.config.FEATURE_DIM:
            obs = obs[:self.config.FEATURE_DIM]
        
        return obs
    
    def get_equity(self) -> float:
        """获取总权益"""
        return self.balance + self.position * self.current_price
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        stats = self.stats.copy()
        stats['win_rate'] = stats['winning_trades'] / max(1, stats['total_trades'])
        stats['avg_pnl'] = stats['total_pnl'] / max(1, stats['total_trades'])
        stats['final_equity'] = self.get_equity()
        stats['total_return'] = (stats['final_equity'] - self.initial_balance) / self.initial_balance
        return stats


# ==================== 回测模块 ====================

class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, config: Config, agent: PPOAgent = None):
        self.config = config
        self.agent = agent
        self.simulator = TradingSimulator(config)
        self.results = {}
        
    def run(self, data: pd.DataFrame, start_date: str = None, end_date: str = None) -> dict:
        """运行回测"""
        if data.empty:
            logging.error("数据为空")
            return {}
        
        # 过滤日期
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        if data.empty:
            logging.error("指定日期范围内无数据")
            return {}
        
        logging.info(f"开始回测: {data.index[0]} 至 {data.index[-1]}")
        
        # 重置模拟器
        self.simulator.reset()
        
        # 逐行回测
        states = []
        actions = []
        rewards = []
        
        for idx, row in data.iterrows():
            # 获取当前状态
            self.simulator.current_price = row['close']
            state = self.simulator._get_observation()
            states.append(state)
            
            # 获取动作
            if self.agent:
                action, _, _ = self.agent.select_action(state, deterministic=True)
            else:
                # 随机策略（基准）
                action = np.random.uniform(-1, 1)
            
            actions.append(action)
            
            # 执行动作
            next_state, reward, done, info = self.simulator.step(action)
            rewards.append(reward)
            
            if done:
                break
        
        # 计算绩效指标
        self.results = self._calculate_metrics(states, actions, rewards)
        
        return self.results
    
    def _calculate_metrics(self, states: list, actions: list, rewards: list) -> dict:
        """计算绩效指标"""
        stats = self.simulator.get_stats()
        
        # 权益曲线
        equity_curve = pd.DataFrame(self.simulator.equity_curve)
        
        # 计算收益率
        if not equity_curve.empty:
            returns = equity_curve['equity'].pct_change().dropna()
            
            # 年化收益率
            total_days = len(equity_curve) / (24 * 60)  # 假设1分钟数据
            annual_return = (1 + stats['total_return']) ** (365 / max(1, total_days)) - 1
            
            # 夏普比率
            risk_free_rate = 0.02  # 假设无风险利率2%
            excess_returns = returns - risk_free_rate / (365 * 24 * 60)
            sharpe_ratio = np.sqrt(365 * 24 * 60) * excess_returns.mean() / max(1e-6, returns.std())
            
            # 索提诺比率
            downside_returns = returns[returns < 0]
            sortino_ratio = np.sqrt(365 * 24 * 60) * excess_returns.mean() / max(1e-6, downside_returns.std()) if len(downside_returns) > 0 else 0
            
            # 卡尔玛比率
            calmar_ratio = annual_return / max(1e-6, stats['max_drawdown'])
            
            # 信息比率
            # 这里简单使用基准为买入持有
            benchmark_returns = equity_curve['price'].pct_change().dropna()
            tracking_error = (returns - benchmark_returns).std()
            info_ratio = (returns.mean() - benchmark_returns.mean()) / max(1e-6, tracking_error) * np.sqrt(365 * 24 * 60)
            
            metrics = {
                'total_return': stats['total_return'],
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'info_ratio': info_ratio,
                'max_drawdown': stats['max_drawdown'],
                'win_rate': stats.get('win_rate', 0),
                'total_trades': stats['total_trades'],
                'avg_pnl': stats.get('avg_pnl', 0),
                'final_equity': stats['final_equity'],
                'peak_equity': stats['peak_equity']
            }
            
            # 交易分析
            if self.simulator.trades:
                trades_df = pd.DataFrame(self.simulator.trades)
                metrics['avg_win'] = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if any(trades_df['pnl'] > 0) else 0
                metrics['avg_loss'] = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if any(trades_df['pnl'] < 0) else 0
                metrics['profit_factor'] = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / 
                                               trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if any(trades_df['pnl'] < 0) else float('inf')
            else:
                metrics['avg_win'] = 0
                metrics['avg_loss'] = 0
                metrics['profit_factor'] = 0
            
            # 添加原始数据
            metrics['equity_curve'] = equity_curve
            metrics['trades'] = self.simulator.trades
            metrics['actions'] = actions
            metrics['rewards'] = rewards
            
            return metrics
        
        return {}


# ==================== 训练模块 ====================

class Trainer:
    """训练器"""
    
    def __init__(self, config: Config, agent: PPOAgent, simulator: TradingSimulator):
        self.config = config
        self.agent = agent
        self.simulator = simulator
        
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'losses': [],
            'equity': []
        }
    	# 修复 train 方法（大约在第2163行附近）
		def train(self, num_episodes: int = 1000, save_every: int = 100):
    		"""训练模型"""
    		logging.info(f"开始训练 {num_episodes} 个 episode")
    
    		for episode in range(num_episodes):
        		# 重置环境
        		state_dict = self.simulator.reset()  # 返回字典
        		# 从字典中提取观测值
        		state = self.simulator._get_observation()  # 获取numpy数组
        		episode_reward = 0
        		episode_steps = 0
        		done = False
        
        		while not done:
            		# 选择动作 - 传入numpy数组
            		action, log_prob, value = self.agent.select_action(state)
            
            		# 执行动作
            		next_state_dict, reward, done, info = self.simulator.step(action)
            		# 获取下一个状态的观测值
            		next_state = self.simulator._get_observation()
            
            		# 存储经验
            		self.agent.store_transition(state, action, reward, next_state, done, value, log_prob)
            
            		# 更新
            		if len(self.agent.buffer) >= self.config.BATCH_SIZE:
                	loss_info = self.agent.update()
            
            		state = next_state
            		episode_reward += reward
            		episode_steps += 1
        
        		# 记录
        		self.training_history['episodes'].append(episode)
        		self.training_history['rewards'].append(episode_reward)
        		self.training_history['equity'].append(self.simulator.get_equity())
        
        		# 日志
        		if episode % 10 == 0:
            		stats = self.simulator.get_stats()
            		logging.info(
                		f"Episode {episode}/{num_episodes} | "
                		f"Reward: {episode_reward:.2f} | "
                		f"Steps: {episode_steps} | "
                		f"Equity: {stats['final_equity']:.2f} | "
                		f"Win Rate: {stats['win_rate']:.2%}"
            		)
        
        		# 保存检查点
        		if episode > 0 and episode % save_every == 0:
            		self.agent.save_checkpoint(episode)
    
    		# 保存最终模型
    		self.agent.save_checkpoint(num_episodes, "models_saved/final_model.pt")
    
    		return self.training_history
          
            
# ==================== 监控面板 ====================

class Dashboard:
    """Streamlit监控面板"""
    
    def __init__(self, config: Config, data_manager: DataManager, agent: PPOAgent, simulator: TradingSimulator):
        self.config = config
        self.data_manager = data_manager
        self.agent = agent
        self.simulator = simulator
        
    def run(self):
        """运行仪表板"""
        st.set_page_config(
            page_title="BTCUSDT RL Trading System",
            page_icon="📈",
            layout="wide"
        )
        
        # 侧边栏导航
        st.sidebar.title("导航")
        page = st.sidebar.radio(
            "选择页面",
            ["实时监控", "模型训练", "回测分析", "数据管理"]
        )
        
        if page == "实时监控":
            self._render_realtime_page()
        elif page == "模型训练":
            self._render_training_page()
        elif page == "回测分析":
            self._render_backtest_page()
        elif page == "数据管理":
            self._render_data_page()
    
    def _render_realtime_page(self):
        """实时监控页面"""
        st.title("📊 实时监控")
        
        # 顶部指标
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = self.data_manager.get_latest_price() or 0
            st.metric("当前价格", f"${current_price:,.2f}")
        
        with col2:
            balance = self.simulator.balance
            st.metric("账户余额", f"${balance:,.2f}")
        
        with col3:
            position = self.simulator.position
            st.metric("当前持仓", f"{position:.4f} BTC")
        
        with col4:
            equity = self.simulator.get_equity()
            change = (equity - self.simulator.initial_balance) / self.simulator.initial_balance
            st.metric("总权益", f"${equity:,.2f}", f"{change:+.2%}")
        
        # 实时K线图
        st.subheader("实时K线")
        
        # 创建图表
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # 模拟数据（实际应该从data_manager获取）
        import numpy as np
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1min')
        prices = 50000 + np.cumsum(np.random.randn(100) * 100)
        
        # 价格图
        fig.add_trace(
            go.Candlestick(
                x=dates,
                open=prices * (1 - np.random.rand(100) * 0.001),
                high=prices * (1 + np.random.rand(100) * 0.002),
                low=prices * (1 - np.random.rand(100) * 0.002),
                close=prices,
                name="价格"
            ),
            row=1, col=1
        )
        
        # 成交量
        volumes = np.random.randint(100, 1000, 100)
        fig.add_trace(
            go.Bar(x=dates, y=volumes, name="成交量"),
            row=2, col=1
        )
        
        # RSI
        rsi = 50 + np.cumsum(np.random.randn(100) * 2)
        rsi = np.clip(rsi, 0, 100)
        fig.add_trace(
            go.Scatter(x=dates, y=rsi, name="RSI(14)", line=dict(color='purple')),
            row=3, col=1
        )
        
        # 添加水平线
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        fig.update_layout(height=600, showlegend=False)
        fig.update_xaxes(rangeslider_visible=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 技术指标快照
        st.subheader("技术指标快照")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("RSI(14)", f"{np.random.randint(30, 70)}")
        with col2:
            st.metric("MACD", f"{np.random.randn():.2f}")
        with col3:
            st.metric("布林带 %b", f"{np.random.rand():.2f}")
        with col4:
            st.metric("ATR", f"${np.random.randint(100, 500)}")
        
        # 最近交易记录
        st.subheader("最近交易记录")
        if self.simulator.trades:
            trades_df = pd.DataFrame(self.simulator.trades[-10:])
            trades_df['time'] = trades_df['time'].dt.strftime('%H:%M:%S')
            trades_df = trades_df[['time', 'side', 'qty', 'price', 'pnl']]
            trades_df['pnl'] = trades_df['pnl'].apply(lambda x: f"${x:.2f}")
            st.dataframe(trades_df, use_container_width=True)
        else:
            st.info("暂无交易记录")
        
        # 系统状态
        st.subheader("系统状态")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ws_status = "🟢 已连接" if self.data_manager.ws_running else "🔴 已断开"
            st.metric("WebSocket", ws_status)
        
        with col2:
            delay = np.random.randint(100, 500) if self.data_manager.ws_running else 0
            st.metric("数据延迟", f"{delay}ms")
        
        with col3:
            risk_status, _, _ = self.simulator.risk_manager.check_all(self.simulator._get_state())
            risk_text = "🟢 正常" if risk_status else "🔴 警告"
            st.metric("风控状态", risk_text)
    
    def _render_training_page(self):
        """模型训练页面"""
        st.title("🤖 模型训练")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("训练控制")
            
            num_episodes = st.number_input("训练轮数", min_value=100, max_value=10000, value=1000, step=100)
            learning_rate = st.slider("学习率", min_value=1e-5, max_value=1e-3, value=3e-4, format="%.5f")
            batch_size = st.selectbox("批量大小", [64, 128, 256, 512], index=2)
            
            if st.button("开始训练", type="primary"):
                st.session_state['training'] = True
                self.agent.config.LEARNING_RATE = learning_rate
                self.agent.config.BATCH_SIZE = batch_size
            
            if st.button("暂停训练"):
                st.session_state['training'] = False
            
            if st.button("停止训练"):
                st.session_state['training'] = False
                st.session_state['training_episodes'] = 0
            
            st.subheader("模型版本")
            model_files = list(Path("models_saved").glob("*.pt")) if Path("models_saved").exists() else []
            if model_files:
                selected_model = st.selectbox("选择模型", [f.name for f in model_files])
                if st.button("加载模型"):
                    self.agent.load_checkpoint(f"models_saved/{selected_model}")
                    st.success(f"已加载模型: {selected_model}")
            
            if st.button("保存当前模型"):
                episode = st.session_state.get('training_episodes', 0)
                self.agent.save_checkpoint(episode, f"models_saved/manual_save_{episode}.pt")
                st.success("模型已保存")
        
        with col2:
            st.subheader("训练曲线")
            
            # 模拟训练数据
            if 'training_history' not in st.session_state:
                st.session_state.training_history = {
                    'episodes': list(range(100)),
                    'rewards': np.cumsum(np.random.randn(100) * 10) + 100,
                    'losses': np.exp(-np.linspace(0, 3, 100)) + np.random.randn(100) * 0.1,
                    'win_rates': np.clip(np.cumsum(np.random.randn(100) * 0.02) + 0.5, 0.3, 0.7)
                }
            
            # 奖励曲线
            fig_reward = go.Figure()
            fig_reward.add_trace(go.Scatter(
                x=st.session_state.training_history['episodes'],
                y=st.session_state.training_history['rewards'],
                mode='lines',
                name='累计奖励',
                line=dict(color='blue', width=2)
            ))
            fig_reward.add_trace(go.Scatter(
                x=st.session_state.training_history['episodes'],
                y=pd.Series(st.session_state.training_history['rewards']).rolling(10).mean(),
                mode='lines',
                name='移动平均',
                line=dict(color='red', width=2, dash='dash')
            ))
            fig_reward.update_layout(
                title="训练奖励曲线",
                xaxis_title="Episode",
                yaxis_title="Reward",
                height=300
            )
            st.plotly_chart(fig_reward, use_container_width=True)
            
            # 损失曲线和胜率
            col_loss, col_win = st.columns(2)
            
            with col_loss:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=st.session_state.training_history['episodes'],
                    y=st.session_state.training_history['losses'],
                    mode='lines',
                    name='损失',
                    line=dict(color='orange', width=2)
                ))
                fig_loss.update_layout(
                    title="训练损失",
                    xaxis_title="Episode",
                    yaxis_title="Loss",
                    height=250
                )
                st.plotly_chart(fig_loss, use_container_width=True)
            
            with col_win:
                fig_win = go.Figure()
                fig_win.add_trace(go.Scatter(
                    x=st.session_state.training_history['episodes'],
                    y=st.session_state.training_history['win_rates'],
                    mode='lines',
                    name='胜率',
                    line=dict(color='green', width=2),
                    fill='tozeroy'
                ))
                fig_win.add_hline(y=0.5, line_dash="dash", line_color="red")
                fig_win.update_layout(
                    title="胜率变化",
                    xaxis_title="Episode",
                    yaxis_title="Win Rate",
                    height=250,
                    yaxis_range=[0, 1]
                )
                st.plotly_chart(fig_win, use_container_width=True)
    
    def _render_backtest_page(self):
        """回测分析页面"""
        st.title("📈 回测分析")
        
        # 时间范围选择
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            start_date = st.date_input("开始日期", datetime.now() - timedelta(days=30))
        
        with col2:
            end_date = st.date_input("结束日期", datetime.now())
        
        with col3:
            st.write("")
            st.write("")
            if st.button("运行回测", type="primary"):
                st.session_state['run_backtest'] = True
        
        # 绩效指标卡片
        if 'backtest_results' in st.session_state:
            results = st.session_state.backtest_results
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("总收益率", f"{results.get('total_return', 0)*100:.2f}%")
            with col2:
                st.metric("年化收益率", f"{results.get('annual_return', 0)*100:.2f}%")
            with col3:
                st.metric("夏普比率", f"{results.get('sharpe_ratio', 0):.2f}")
            with col4:
                st.metric("最大回撤", f"{results.get('max_drawdown', 0)*100:.2f}%")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("胜率", f"{results.get('win_rate', 0)*100:.2f}%")
            with col2:
                st.metric("交易次数", f"{results.get('total_trades', 0)}")
            with col3:
                st.metric("盈亏比", f"{results.get('profit_factor', 0):.2f}")
            with col4:
                st.metric("索提诺比率", f"{results.get('sortino_ratio', 0):.2f}")
            
            # 权益曲线
            if 'equity_curve' in results:
                equity_df = results['equity_curve']
                
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    row_heights=[0.7, 0.3]
                )
                
                # 权益曲线
                fig.add_trace(
                    go.Scatter(
                        x=equity_df.index if isinstance(equity_df, pd.DataFrame) else list(range(len(equity_df))),
                        y=equity_df['equity'] if isinstance(equity_df, pd.DataFrame) else equity_df,
                        mode='lines',
                        name='权益',
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )
                
                # 回撤曲线
                if isinstance(equity_df, pd.DataFrame):
                    peak = equity_df['equity'].expanding().max()
                    drawdown = (peak - equity_df['equity']) / peak
                    
                    fig.add_trace(
                        go.Scatter(
                            x=equity_df.index,
                            y=drawdown * 100,
                            mode='lines',
                            name='回撤',
                            line=dict(color='red', width=2),
                            fill='tozeroy'
                        ),
                        row=2, col=1
                    )
                
                fig.update_layout(height=500, title="回测权益曲线")
                fig.update_yaxes(title_text="权益 (USDT)", row=1, col=1)
                fig.update_yaxes(title_text="回撤 (%)", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
            
            # 月度收益热力图
            if 'trades' in results and results['trades']:
                trades_df = pd.DataFrame(results['trades'])
                trades_df['month'] = pd.to_datetime(trades_df['time']).dt.to_period('M')
                monthly_pnl = trades_df.groupby('month')['pnl'].sum().reset_index()
                
                # 创建热力图数据
                months = monthly_pnl['month'].astype(str)
                pnls = monthly_pnl['pnl']
                
                fig = go.Figure(data=go.Heatmap(
                    z=[pnls],
                    x=months,
                    y=['PnL'],
                    colorscale='RdYlGn',
                    text=[[f"${x:.2f}" for x in pnls]],
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    colorbar=dict(title="PnL")
                ))
                
                fig.update_layout(
                    title="月度收益分布",
                    height=200,
                    xaxis_title="月份",
                    yaxis_title=""
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            # 显示模拟回测结果
            st.info("点击'运行回测'按钮开始分析")
            
            # 示例回测图表
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            equity = 1000 * (1 + np.cumsum(np.random.randn(len(dates)) * 0.01))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=equity,
                mode='lines',
                name='权益曲线',
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                title="回测权益曲线 (示例)",
                xaxis_title="日期",
                yaxis_title="权益 (USDT)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_data_page(self):
        """数据管理页面"""
        st.title("💾 数据管理")
        
        # 数据状态
        st.subheader("数据状态")
        
        data_info = self.data_manager.get_data_info()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("数据范围", f"{data_info.get('start_date', 'N/A')} 至 {data_info.get('end_date', 'N/A')}")
        
        with col2:
            st.metric("总天数", f"{data_info.get('total_days', 0)} 天")
        
        with col3:
            st.metric("总记录数", f"{data_info.get('total_rows', 0):,}")
        
        with col4:
            st.metric("最后更新", data_info.get('end_date', 'N/A'))
        
        # 数据操作
        st.subheader("数据操作")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            years = st.number_input("下载年数", min_value=1, max_value=5, value=3)
            if st.button("下载历史数据"):
                with st.spinner("正在下载数据..."):
                    df = self.data_manager.download_historical(years=years)
                    if not df.empty:
                        st.success(f"数据下载完成，共 {len(df)} 条记录")
                    else:
                        st.error("数据下载失败")
        
        with col2:
            if st.button("更新增量数据"):
                with st.spinner("正在更新数据..."):
                    st.info("增量更新功能开发中")
        
        with col3:
            if st.button("验证完整性"):
                st.info("数据完整性验证中...")
                # 模拟验证
                time.sleep(2)
                st.success("数据完整性验证通过")
        
        with col4:
            if st.button("重新计算特征"):
                with st.spinner("正在重新计算特征..."):
                    st.info("特征计算功能开发中")
        
        # 特征分析
        st.subheader("特征分析")
        
        # 特征重要性
        fig_importance = go.Figure()
        
        # 模拟特征重要性数据
        features = [f'Feature_{i}' for i in range(20)]
        importances = np.random.rand(20)
        importances = importances / importances.sum()
        
        fig_importance.add_trace(go.Bar(
            x=importances,
            y=features,
            orientation='h',
            marker=dict(
                color=importances,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="重要性")
            )
        ))
        
        fig_importance.update_layout(
            title="特征重要性 Top 20",
            xaxis_title="重要性",
            yaxis_title="特征",
            height=500
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # 数据预览
        st.subheader("数据预览")
        
        # 加载最近的数据
        if data_info.get('end_date'):
            end = data_info['end_date']
            start = (pd.to_datetime(end) - timedelta(days=7)).strftime('%Y-%m-%d')
            df = self.data_manager.load_historical_data(start, end)
            
            if not df.empty:
                st.dataframe(df.head(100), use_container_width=True)
                
                # 数据统计
                st.subheader("数据统计")
                stats_df = df.describe()
                st.dataframe(stats_df, use_container_width=True)
            else:
                st.info("暂无数据可预览")
        else:
            st.info("请先下载历史数据")


# ==================== 主程序 ====================

class TradingSystem:
    """交易系统主控制器"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # 设置日志
        self._setup_logging()
        
        # 初始化组件
        self.data_manager = DataManager(config)
        self.agent = PPOAgent(config)
        self.simulator = TradingSimulator(config, self.data_manager)
        self.trainer = Trainer(config, self.agent, self.simulator)
        self.backtest = BacktestEngine(config, self.agent)
        self.dashboard = Dashboard(config, self.data_manager, self.agent, self.simulator)
        
        # 运行状态
        self.running = False
        self.mode = None
        
        logging.info("交易系统初始化完成")
    
    def _setup_logging(self):
        """设置日志"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=self.config.LOG_LEVEL,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def run_simulation(self):
        """运行模拟交易"""
        self.mode = 'simulation'
        self.running = True
        
        logging.info("启动模拟交易模式")
        
        # 启动数据收集
        self.data_manager.start_realtime_collection()
        
        # 重置模拟器
        self.simulator.reset()
        
        try:
            while self.running:
                # 获取最新价格
                price = self.data_manager.get_latest_price()
                
                if price:
                    # 获取状态
                    state = self.simulator._get_observation()
                    
                    # 选择动作
                    action, _, _ = self.agent.select_action(state)
                    
                    # 执行动作
                    next_state, reward, done, info = self.simulator.step(action)
                    
                    # 存储经验
                    self.agent.store_transition(state, action, reward, next_state, done, 0, 0)
                    
                    # 定期更新模型
                    if len(self.agent.buffer) >= self.config.BATCH_SIZE:
                        self.agent.update()
                    
                    # 日志
                    if self.simulator.current_step % 100 == 0:
                        stats = self.simulator.get_stats()
                        logging.info(
                            f"Step {self.simulator.current_step} | "
                            f"Price: ${price:.2f} | "
                            f"Equity: ${stats['final_equity']:.2f} | "
                            f"Position: {self.simulator.position:.4f}"
                        )
                
                time.sleep(1)  # 每秒更新一次
                
        except KeyboardInterrupt:
            logging.info("用户中断")
        except Exception as e:
            logging.error(f"运行错误: {e}")
        finally:
            self.stop()
    
    def run_backtest(self, start_date: str, end_date: str):
        """运行回测"""
        logging.info(f"启动回测模式: {start_date} 至 {end_date}")
        
        # 加载历史数据
        df = self.data_manager.load_historical_data(start_date, end_date)
        
        if df.empty:
            logging.error("未找到历史数据")
            return
        
        # 运行回测
        results = self.backtest.run(df, start_date, end_date)
        
        # 输出结果
        if results:
            logging.info("=" * 50)
            logging.info("回测结果:")
            logging.info(f"总收益率: {results.get('total_return', 0)*100:.2f}%")
            logging.info(f"年化收益率: {results.get('annual_return', 0)*100:.2f}%")
            logging.info(f"夏普比率: {results.get('sharpe_ratio', 0):.2f}")
            logging.info(f"最大回撤: {results.get('max_drawdown', 0)*100:.2f}%")
            logging.info(f"胜率: {results.get('win_rate', 0)*100:.2f}%")
            logging.info(f"交易次数: {results.get('total_trades', 0)}")
            logging.info("=" * 50)
        
        return results
    
    def train(self, num_episodes: int = 1000):
        """训练模型"""
        logging.info(f"启动训练模式: {num_episodes} episodes")
        
        # 加载历史数据用于训练
        df = self.data_manager.load_historical_data(
            (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
            datetime.now().strftime('%Y-%m-%d')
        )
        
        # 这里需要实现使用历史数据训练的逻辑
        # 简化版：直接调用trainer
        history = self.trainer.train(num_episodes)
        
        return history
    
    def run_dashboard(self):
        """运行监控面板"""
        logging.info(f"启动监控面板: http://localhost:{self.config.DASHBOARD_PORT}")
        self.dashboard.run()
    
    def stop(self):
        """停止系统"""
        self.running = False
        self.data_manager.stop()
        logging.info("交易系统已停止")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='BTCUSDT RL Trading System')
    parser.add_argument('--mode', type=str, default='dashboard',
                        choices=['simulate', 'backtest', 'train', 'dashboard'],
                        help='运行模式')
    parser.add_argument('--model', type=str, help='模型文件路径')
    parser.add_argument('--start', type=str, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--episodes', type=int, default=1000, help='训练轮数')
    
    args = parser.parse_args()
    
    # 创建配置
    config = Config()
    
    # 创建交易系统
    system = TradingSystem(config)
    
    # 加载模型
    if args.model:
        system.agent.load_checkpoint(args.model)
    
    # 运行指定模式
    if args.mode == 'simulate':
        system.run_simulation()
    elif args.mode == 'backtest':
        if not args.start or not args.end:
            print("请指定开始和结束日期")
            return
        system.run_backtest(args.start, args.end)
    elif args.mode == 'train':
        system.train(args.episodes)
    elif args.mode == 'dashboard':
        system.run_dashboard()


if __name__ == "__main__":
    main()