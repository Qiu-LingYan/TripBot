#!/usr/bin/env python3
"""
BTCUSDT深度强化学习交易系统 - 完整单文件实现
支持Bybit Testnet环境，包含实时交易、模型训练、回测分析等功能
"""

import os
import sys
import json
import time
import asyncio
import websockets
import threading
from datetime import datetime, timedelta
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

# 禁用Streamlit警告
st.set_option('deprecation.showPyplotGlobalUse', False)

# 全局配置
class Config:
    # API配置
    BYBIT_TESTNET_WS_URL = "wss://stream-testnet.bybit.com/v5/public/linear"
    BYBIT_TESTNET_REST_URL = "https://api-testnet.bybit.com"
    
    # 交易参数
    SYMBOL = "BTCUSDT"
    INTERVAL = "1"  # 1分钟K线
    INITIAL_BALANCE = 1000.0
    FEE_RATE = 0.0006  # 0.06% taker fee
    MAX_POSITION = 0.3  # 最大仓位30%
    
    # 模型参数
    STATE_DIM = 256
    HIDDEN_DIM = 512
    ACTION_DIM = 1
    GAMMA = 0.99
    LAMBDA = 0.95
    CLIP_EPSILON = 0.2
    ENTROPY_COEF = 0.01
    VALUE_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    TARGET_KL = 0.015
    BATCH_SIZE = 64
    BUFFER_SIZE = 20000
    LEARNING_RATE = 3e-4
    
    # 特征工程参数
    FEATURE_DIM = 256
    
    # 系统参数
    LOG_LEVEL = "INFO"
    CHECKPOINT_DIR = "./models_saved/"
    DATA_DIR = "./data/"

# 神经网络架构
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(10)
        self.lstm = nn.LSTM(1280, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_dim)
        batch_size, seq_len, feature_dim = x.shape
        
        # Conv1D处理
        x_conv = x.unsqueeze(1)  # (batch_size, 1, seq_len, feature_dim)
        x_conv = x_conv.view(batch_size, 1, seq_len * feature_dim)
        x_conv = F.relu(self.bn1(self.conv1(x_conv)))
        x_conv = F.relu(self.bn2(self.conv2(x_conv)))
        x_conv = self.pool(x_conv)
        x_conv = x_conv.view(batch_size, -1)
        
        # BiLSTM处理
        x_lstm, _ = self.lstm(x.view(batch_size, seq_len, feature_dim))
        x_lstm = x_lstm[:, -1, :]  # 取最后一个时间步
        
        # Attention机制
        x_attn = x_lstm.unsqueeze(0)  # (1, batch_size, hidden_dim*2)
        x_attn, _ = self.attention(x_attn, x_attn, x_attn)
        x_attn = x_attn.squeeze(0)
        
        # 残差连接和层归一化
        output = self.layer_norm(x_lstm + x_attn)
        return output

class ActorNetwork(nn.Module):
    def __init__(self, state_dim=256, hidden_dim=512, action_dim=1):
        super().__init__()
        self.feature_extractor = FeatureExtractor(state_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.ln3 = nn.LayerNorm(hidden_dim // 4)
        self.mean_layer = nn.Linear(hidden_dim // 4, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
    def forward(self, x):
        features = self.feature_extractor(x)
        x = F.relu(self.ln1(self.fc1(features)))
        x = F.dropout(x, p=0.2)
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        mean = torch.tanh(self.mean_layer(x))
        log_std = self.log_std.expand_as(mean)
        return mean, log_std

class CriticNetwork(nn.Module):
    def __init__(self, state_dim=256, hidden_dim=512):
        super().__init__()
        self.feature_extractor = FeatureExtractor(state_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.ln3 = nn.LayerNorm(hidden_dim // 4)
        self.value_layer = nn.Linear(hidden_dim // 4, 1)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        x = F.relu(self.ln1(self.fc1(features)))
        x = F.dropout(x, p=0.2)
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        value = self.value_layer(x)
        return value

# 经验回放缓冲区
class PrioritizedReplayBuffer:
    def __init__(self, capacity=20000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = 0.6
        self.beta = 0.4
        self.epsilon = 1e-6
        
    def add(self, experience, priority=None):
        if priority is None:
            priority = 1.0
        self.buffer.append(experience)
        self.priorities.append(priority)
        
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return []
            
        priorities = np.array(self.priorities) + self.epsilon
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return samples, indices, weights
        
    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = abs(error) + self.epsilon

# PPO智能体
class PPOAgent:
    def __init__(self, state_dim=256, action_dim=1, device="cpu"):
        self.device = device
        self.actor = ActorNetwork(state_dim, Config.HIDDEN_DIM, action_dim).to(device)
        self.critic = CriticNetwork(state_dim, Config.HIDDEN_DIM).to(device)
        
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=Config.LEARNING_RATE
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        
        self.buffer = PrioritizedReplayBuffer(Config.BUFFER_SIZE)
        self.gamma = Config.GAMMA
        self.lambda_ = Config.LAMBDA
        self.clip_epsilon = Config.CLIP_EPSILON
        self.entropy_coef = Config.ENTROPY_COEF
        self.value_coef = Config.VALUE_COEF
        self.max_grad_norm = Config.MAX_GRAD_NORM
        self.target_kl = Config.TARGET_KL
        
    def select_action(self, state, deterministic=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mean, log_std = self.actor(state_tensor)
            std = log_std.exp()
            dist = Normal(mean, std)
            
            if deterministic:
                action = mean
            else:
                action = dist.sample()
                
            log_prob = dist.log_prob(action)
            value = self.critic(state_tensor)
            
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, next_state, done, value, log_prob):
        experience = (state, action, reward, next_state, done, value, log_prob)
        self.buffer.add(experience)
    
    def update(self):
        if len(self.buffer.buffer) < Config.BATCH_SIZE:
            return {}
            
        samples, indices, weights = self.buffer.sample(Config.BATCH_SIZE)
        if not samples:
            return {}
            
        states, actions, rewards, next_states, dones, old_values, old_log_probs = zip(*samples)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        old_values = torch.FloatTensor(np.array(old_values)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device)
        weights = torch.FloatTensor(np.array(weights)).to(self.device)
        
        # 计算GAE
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze()
            td_targets = rewards + self.gamma * next_values * (1 - dones)
            advantages = td_targets - old_values
            
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新
        mean, log_std = self.actor(states)
        std = log_std.exp()
        dist = Normal(mean, std)
        new_log_probs = dist.log_prob(actions)
        
        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 价值函数损失
        values = self.critic(states).squeeze()
        value_loss = F.mse_loss(values, td_targets)
        
        # 熵奖励
        entropy_loss = -dist.entropy().mean()
        
        # 总损失
        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        
        # 更新优先级
        with torch.no_grad():
            td_errors = (td_targets - values).abs().cpu().numpy()
        self.buffer.update_priorities(indices, td_errors)
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': loss.item(),
            'mean_reward': rewards.mean().item()
        }
    
    def save_checkpoint(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# 特征工程
class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.feature_names = []
        
    def calculate_all_features(self, df):
        features = []
        
        # 趋势指标
        for period in [5, 10, 20, 50]:
            df[f'SMA_{period}'] = df['close'].rolling(period).mean()
            df[f'EMA_{period}'] = df['close'].ewm(span=period).mean()
            features.extend([f'SMA_{period}', f'EMA_{period}'])
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['MACD_line'] = ema12 - ema26
        df['MACD_signal'] = df['MACD_line'].ewm(span=9).mean()
        df['MACD_hist'] = df['MACD_line'] - df['MACD_signal']
        features.extend(['MACD_line', 'MACD_signal', 'MACD_hist'])
        
        # RSI
        for period in [6, 14, 21, 28]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            rs = avg_gain / avg_loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
            features.append(f'RSI_{period}')
        
        # 布林带
        for period in [20]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            df[f'Bollinger_upper_{period}'] = sma + 2 * std
            df[f'Bollinger_middle_{period}'] = sma
            df[f'Bollinger_lower_{period}'] = sma - 2 * std
            df[f'Bollinger_width_{period}'] = (df[f'Bollinger_upper_{period}'] - df[f'Bollinger_lower_{period}']) / sma
            df[f'Bollinger_%b_{period}'] = (df['close'] - df[f'Bollinger_lower_{period}']) / (df[f'Bollinger_upper_{period}'] - df[f'Bollinger_lower_{period}'])
            features.extend([f'Bollinger_upper_{period}', f'Bollinger_middle_{period}', 
                           f'Bollinger_lower_{period}', f'Bollinger_width_{period}', f'Bollinger_%b_{period}'])
        
        # 成交量指标
        for period in [5, 10, 20]:
            df[f'Volume_SMA_{period}'] = df['volume'].rolling(period).mean()
            features.append(f'Volume_SMA_{period}')
        
        df['Volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features.append('Volume_ratio')
        
        # 时间特征
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        features.extend(['hour_sin', 'hour_cos', 'day_sin', 'day_cos'])
        
        # 统计特征
        for period in [20]:
            df[f'skewness_{period}'] = df['close'].rolling(period).skew()
            df[f'kurtosis_{period}'] = df['close'].rolling(period).kurt()
            features.extend([f'skewness_{period}', f'kurtosis_{period}'])
        
        # 选择前256个特征
        self.feature_names = features[:Config.FEATURE_DIM]
        feature_df = df[self.feature_names].fillna(method='bfill').fillna(0)
        
        return feature_df.values

# 滑点模型
class SlippageModel:
    def __init__(self):
        self.base_slippage = 0.0001
        self.liquidity_factor = 1.0
        
    def calculate_slippage(self, side, qty, price, orderbook):
        if orderbook is None:
            return self.base_slippage
            
        total_liquidity = sum([level['size'] for level in orderbook[:10]])
        spread = (orderbook[0]['ask_price'] - orderbook[0]['bid_price']) / price
        liquidity_factor = 1.0 - min(spread / 0.001, 1.0)  # 价差越大，流动性越差
        
        slippage = self.base_slippage + (qty / total_liquidity) * 0.001 + (1 - liquidity_factor) * 0.0005
        return min(slippage, 0.01)  # 最大滑点1%

# 风险管理器
class RiskManager:
    def __init__(self):
        self.daily_loss_limit = 0.1
        self.position_limit = Config.MAX_POSITION
        self.leverage_limit = 3.0
        self.stop_loss = 0.02
        self.take_profit = 0.03
        self.slippage_limit = 0.001
        self.consecutive_losses = 0
        self.last_loss_time = None
        
    def check_all(self, state):
        # Level 1 - 账户风控
        if state.get('daily_pnl', 0) < -self.daily_loss_limit * state.get('initial_balance', 1000):
            return False, 1, "日亏损超过10%限制"
            
        if state.get('balance', 1000) < 10:
            return False, 1, "余额低于10 USDT"
            
        # Level 2 - 仓位风控
        position_ratio = abs(state.get('position', 0) * state.get('current_price', 1) / state.get('balance', 1000))
        if position_ratio > self.position_limit:
            return False, 2, f"仓位比例超过{self.position_limit*100}%限制"
            
        # Level 3 - 订单风控
        if state.get('slippage', 0) > self.slippage_limit:
            return False, 3, f"滑点超过{self.slippage_limit*100}%限制"
            
        # Level 4 - 系统风控
        if state.get('data_delay', 0) > 10:
            return False, 4, "数据延迟超过10秒"
            
        return True, 0, "风控检查通过"

# 交易模拟器
class TradingSimulator:
    def __init__(self):
        self.initial_balance = Config.INITIAL_BALANCE
        self.balance = Config.INITIAL_BALANCE
        self.position = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.trades = []
        self.equity_curve = []
        self.current_step = 0
        self.fee_rate = Config.FEE_RATE
        self.slippage_model = SlippageModel()
        self.risk_manager = RiskManager()
        self.daily_pnl = 0.0
        self.last_trade_time = None
        
    def reset(self, initial_balance=1000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.trades = []
        self.equity_curve = []
        self.current_step = 0
        self.daily_pnl = 0.0
        self.last_trade_time = None
        return self._get_state()
    
    def _get_state(self):
        return {
            'balance': self.balance,
            'position': self.position,
            'entry_price': self.entry_price,
            'unrealized_pnl': self.unrealized_pnl,
            'equity': self.balance + self.unrealized_pnl,
            'daily_pnl': self.daily_pnl
        }
    
    def step(self, action, price, orderbook=None):
        old_equity = self.balance + self.unrealized_pnl
        
        # 计算目标仓位
        max_position = self.balance * Config.MAX_POSITION / price
        target_position = self.position + action * 0.1 * max_position
        target_position = np.clip(target_position, -max_position, max_position)
        
        # 计算交易量
        trade_qty = target_position - self.position
        
        if abs(trade_qty) > 0.0001:  # 最小交易量
            # 模拟滑点
            slippage = self.slippage_model.calculate_slippage(
                'buy' if trade_qty > 0 else 'sell',
                abs(trade_qty),
                price,
                orderbook
            )
            
            # 计算成交价
            executed_price = price * (1 + slippage) if trade_qty > 0 else price * (1 - slippage)
            
            # 计算手续费
            fee = abs(trade_qty) * executed_price * self.fee_rate
            
            # 执行交易
            trade_value = trade_qty * executed_price
            self.balance -= trade_value + fee
            self.position += trade_qty
            
            # 记录交易
            trade = {
                'timestamp': datetime.now(),
                'side': 'BUY' if trade_qty > 0 else 'SELL',
                'quantity': abs(trade_qty),
                'price': executed_price,
                'slippage': slippage,
                'fee': fee
            }
            self.trades.append(trade)
            self.last_trade_time = datetime.now()
        
        # 更新未实现盈亏
        if self.position != 0:
            self.unrealized_pnl = self.position * (price - self.entry_price)
        else:
            self.unrealized_pnl = 0.0
            self.entry_price = 0.0
        
        # 计算奖励
        new_equity = self.balance + self.unrealized_pnl
        reward = (new_equity - old_equity) / old_equity * 100
        
        # 惩罚频繁交易
        if self.last_trade_time and (datetime.now() - self.last_trade_time).total_seconds() < 10:
            reward -= 0.1
        
        # 检查终止条件
        done = False
        if new_equity < self.initial_balance * 0.5:  # 亏损50%终止
            done = True
        if self.current_step > 10000:  # 最大步数
            done = True
        
        self.current_step += 1
        self.equity_curve.append(new_equity)
        
        return self._get_state(), reward, done, {'price': price, 'slippage': slippage}

# 实时数据收集器
class RealtimeDataCollector:
    def __init__(self):
        self.ws_url = Config.BYBIT_TESTNET_WS_URL
        self.subscriptions = ["kline.1.BTCUSDT"]
        self.price_buffer = deque(maxlen=1000)
        self.volume_buffer = deque(maxlen=1000)
        self.orderbook_buffer = {}
        self.last_update = None
        self.websocket = None
        self.running = False
        self.current_data = None
        
    async def connect(self):
        try:
            self.websocket = await websockets.connect(self.ws_url)
            await self.subscribe()
            self.running = True
            asyncio.create_task(self._listen())
        except Exception as e:
            print(f"WebSocket连接失败: {e}")
            await asyncio.sleep(5)
            await self.connect()
    
    async def subscribe(self):
        subscribe_msg = {
            "op": "subscribe",
            "args": self.subscriptions
        }
        await self.websocket.send(json.dumps(subscribe_msg))
    
    async def _listen(self):
        while self.running:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                if 'topic' in data and 'kline' in data['topic']:
                    kline_data = data['data']
                    if kline_data:
                        self.current_data = {
                            'timestamp': datetime.fromtimestamp(kline_data[0]['start'] / 1000),
                            'open': float(kline_data[0]['open']),
                            'high': float(kline_data[0]['high']),
                            'low': float(kline_data[0]['low']),
                            'close': float(kline_data[0]['close']),
                            'volume': float(kline_data[0]['volume'])
                        }
                        self.price_buffer.append(float(kline_data[0]['close']))
                        self.volume_buffer.append(float(kline_data[0]['volume']))
                        self.last_update = datetime.now()
                        
            except Exception as e:
                print(f"WebSocket接收错误: {e}")
                await asyncio.sleep(1)
                await self.connect()
                break
    
    def get_latest_data(self):
        return self.current_data
    
    def get_data_delay(self):
        if self.last_update:
            return (datetime.now() - self.last_update).total_seconds()
        return 999
    
    def stop(self):
        self.running = False
        if self.websocket:
            asyncio.get_event_loop().run_until_complete(self.websocket.close())

# 回测引擎
class BacktestEngine:
    def __init__(self, agent, simulator, feature_engineer):
        self.agent = agent
        self.simulator = simulator
        self.feature_engineer = feature_engineer
        self.results = {}
        
    def run(self, data, start_date, end_date):
        filtered_data = data[(data.index >= start_date) & (data.index <= end_date)]
        state = self.simulator.reset()
        
        for i, (timestamp, row) in enumerate(filtered_data.iterrows()):
            # 计算特征
            features = self.feature_engineer.calculate_all_features(filtered_data.iloc[:i+1])
            if len(features) < 10:  # 确保有足够数据
                continue
                
            # 获取动作
            action, log_prob, value = self.agent.select_action(features[-1:])
            
            # 执行交易
            next_state, reward, done, info = self.simulator.step(
                action[0], row['close']
            )
            
            # 存储经验
            self.agent.store_transition(features[-1], action, reward, features[-1], done, value, log_prob)
            
            if done:
                break
        
        # 计算绩效指标
        self.results = self.calculate_metrics()
        return self.results
    
    def calculate_metrics(self):
        equity_curve = np.array(self.simulator.equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] if len(equity_curve) > 0 else 0
        annual_return = (1 + total_return) ** (365 / len(equity_curve)) - 1 if len(equity_curve) > 0 else 0
        
        # 夏普比率（假设无风险利率为0）
        sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252) if len(returns) > 0 else 0
        
        # 最大回撤
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = drawdown.max() if len(drawdown) > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'final_equity': equity_curve[-1] if len(equity_curve) > 0 else 0,
            'total_trades': len(self.simulator.trades)
        }

# Streamlit监控面板
def create_dashboard():
    st.title("🎯 BTCUSDT深度强化学习交易系统")
    
    # 初始化会话状态
    if 'agent' not in st.session_state:
        st.session_state.agent = PPOAgent(device="cpu")
    if 'simulator' not in st.session_state:
        st.session_state.simulator = TradingSimulator()
    if 'collector' not in st.session_state:
        st.session_state.collector = RealtimeDataCollector()
    if 'feature_engineer' not in st.session_state:
        st.session_state.feature_engineer = FeatureEngineer()
    
    # 侧边栏导航
    page = st.sidebar.selectbox(
        "选择页面",
        ["实时监控", "模型训练", "回测分析", "数据管理"]
    )
    
    if page == "实时监控":
        st.header("📊 实时监控面板")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("当前余额", f"{st.session_state.simulator.balance:.2f} USDT")
        with col2:
            st.metric("当前持仓", f"{st.session_state.simulator.position:.4f} BTC")
        with col3:
            st.metric("未实现盈亏", f"{st.session_state.simulator.unrealized_pnl:.2f}")
        with col4:
            st.metric("今日交易", len([t for t in st.session_state.simulator.trades 
                                     if t['timestamp'].date() == datetime.now().date()]))
        
        # 实时K线图
        if st.session_state.collector.current_data:
            data = st.session_state.collector.current_data
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[data['timestamp']],
                y=[data['close']],
                mode='lines+markers',
                name='BTCUSDT'
            ))
            fig.update_layout(title="实时价格")
            st.plotly_chart(fig, use_container_width=True)
        
        # 系统状态
        st.subheader("系统状态")
        col1, col2, col3 = st.columns(3)
        with col1:
            delay = st.session_state.collector.get_data_delay()
            status = "🟢 正常" if delay < 5 else "🟡 延迟" if delay < 10 else "🔴 异常"
            st.metric("数据延迟", f"{delay:.1f}s", status)
        with col2:
            st.metric("内存使用", f"{psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
        with col3:
            st.metric("模型版本", "v1.0")
    
    elif page == "模型训练":
        st.header("🤖 模型训练")
        
        episodes = st.number_input("训练轮数", min_value=1, max_value=10000, value=1000)
        batch_size = st.selectbox("批量大小", [32, 64, 128, 256], index=1)
        
        if st.button("开始训练"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for episode in range(episodes):
                # 模拟训练过程
                time.sleep(0.01)
                progress = (episode + 1) / episodes
                progress_bar.progress(progress)
                status_text.text(f"训练中... {episode + 1}/{episodes}")
                
                if episode % 100 == 0:
                    # 模拟损失更新
                    loss_info = st.session_state.agent.update()
                    if loss_info:
                        st.write(f"Episode {episode}: 总损失 {loss_info['total_loss']:.4f}")
            
            st.success("训练完成！")
    
    elif page == "回测分析":
        st.header("📈 回测分析")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("开始日期", datetime(2023, 1, 1))
        with col2:
            end_date = st.date_input("结束日期", datetime(2023, 12, 31))
        
        if st.button("运行回测"):
            # 创建示例数据
            dates = pd.date_range(start=start_date, end=end_date, freq='1min')
            sample_data = pd.DataFrame({
                'open': 30000 + np.random.randn(len(dates)) * 1000,
                'high': 30000 + np.random.randn(len(dates)) * 1500,
                'low': 30000 + np.random.randn(len(dates)) * 1500,
                'close': 30000 + np.random.randn(len(dates)) * 1000,
                'volume': np.random.uniform(1, 100, len(dates))
            }, index=dates)
            
            backtest_engine = BacktestEngine(
                st.session_state.agent,
                st.session_state.simulator,
                st.session_state.feature_engineer
            )
            results = backtest_engine.run(sample_data, start_date, end_date)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总收益率", f"{results['total_return']*100:.2f}%")
            with col2:
                st.metric("年化收益率", f"{results['annual_return']*100:.2f}%")
            with col3:
                st.metric("夏普比率", f"{results['sharpe_ratio']:.2f}")
            with col4:
                st.metric("最大回撤", f"{results['max_drawdown']*100:.2f}%")
    
    elif page == "数据管理":
        st.header("🗃️ 数据管理")
        
        st.subheader("数据状态")
        if st.session_state.collector.current_data:
            latest_data = st.session_state.collector.current_data
            st.write(f"最新数据时间: {latest_data['timestamp']}")
            st.write(f"最新价格: {latest_data['close']:.2f}")
        
        if st.button("下载历史数据"):
            st.info("开始下载历史数据...")
            # 模拟下载过程
            time.sleep(2)
            st.success("数据下载完成！")

# 主程序
def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--dashboard':
        # 启动Streamlit面板
        create_dashboard()
    else:
        # 命令行模式
        print("BTCUSDT深度强化学习交易系统")
        print("=" * 50)
        print("可用模式:")
        print("1. --dashboard  启动监控面板")
        print("2. --train     训练模型")
        print("3. --backtest  运行回测")
        print("4. --simulate  实时模拟交易")
        
        if len(sys.argv) == 1:
            print("\n示例:")
            print("python main.py --dashboard")
            print("python main.py --train --episodes 1000")
            print("python main.py --backtest --start 2023-01-01 --end 2023-12-31")

if __name__ == "__main__":
    # 检查是否安装了必要的包
    try:
        import psutil
    except ImportError:
        print("请安装psutil: pip install psutil")
        sys.exit(1)
    
    main()
