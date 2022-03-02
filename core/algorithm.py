import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import *
from statsmodels.tsa.stattools import adfuller

from core.tools import *
from core.account import *


class PairTrader:
    def __init__(self, X, Y, z_crit, z_sl, z_tp, trainval_split, window, trade_size, pause_after_sl=0):
        assert len(Y) == len(X)
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.z_crit = z_crit
        self.z_sl = z_sl
        self.z_tp = z_tp
        self.trainval_split = trainval_split
        self.window = window
        self.trade_size = trade_size
        self.pause_after_sl = pause_after_sl
    
    def fit_model(self):
        self.b1, self.b0, self.adf_p = fit_model(self.X[:self.trainval_split], self.Y[:self.trainval_split])
        self.spread = self.Y - self.b1*self.X
        self.ma = rollingMA(self.spread, self.window)
        self.sd = rollingSD(self.spread, self.window) # Note: if window is too small; sd may be 0
        self.z = (self.spread - self.ma)/self.sd
    
    def test_model(self):
        self.account = Account(self.trade_size)
        self.logs = []
        pause = 0
        for i in range(self.trainval_split, len(self.Y)):
            if pause > 0:
                pause -= 1
                continue
            n = self.trade_size/self.Y[i]
            if len(self.account.positions) == 0:
                if -self.z_crit - self.z_sl < self.z[i] < -self.z_crit: # long spread
                    self.account.update_position('Y', n, self.Y[i]) # buy Y
                    self.account.update_position('X', -n*self.b1, self.X[i]) # sell b1*X
                    stoploss = (-self.z_crit - self.z_sl)*self.sd[i] + self.ma[i]
                    takeprofit = (-self.z_crit + self.z_tp)*self.sd[i] + self.ma[i]
                    info = {'status': 'L', 'spread': self.spread[i], 'stoploss': stoploss, 'takeprofit': takeprofit, 'balance': self.account.total_balance}
                    self.logs.append((i, info))
                elif self.z_crit < self.z[i] < self.z_crit + self.z_sl: # short spread
                    self.account.update_position('Y', -n, self.Y[i]) # sell Y
                    self.account.update_position('X', n*self.b1, self.X[i]) # buy b1*X
                    stoploss = (self.z_crit + self.z_sl)*self.sd[i] + self.ma[i]
                    takeprofit = (self.z_crit - self.z_tp)*self.sd[i] + self.ma[i]
                    info = {'status': 'S', 'spread': self.spread[i], 'stoploss': stoploss, 'takeprofit': takeprofit, 'balance': self.account.total_balance}
                    self.logs.append((i, info))
            else:
                last_info = self.logs[-1][1]
                if last_info['status'] == 'L':
                    if self.spread[i] < last_info['stoploss']:
                        self.account.update_position('Y', 'close', self.Y[i]) # sell Y
                        self.account.update_position('X', 'close', self.X[i]) # buy b1*X
                        info = {'status': 'SL', 'spread': self.spread[i], 'balance': self.account.total_balance}
                        pause = self.pause_after_sl
                        self.logs.append((i, info))
                    elif self.spread[i] > last_info['takeprofit']:
                        self.account.update_position('Y', 'close', self.Y[i]) # sell Y
                        self.account.update_position('X', 'close', self.X[i]) # buy b1*X
                        info = {'status': 'TP', 'spread': self.spread[i], 'balance': self.account.total_balance}
                        self.logs.append((i, info))
                elif last_info['status'] == 'S':
                    if self.spread[i] > last_info['stoploss']:
                        self.account.update_position('Y', 'close', self.Y[i]) # buy Y
                        self.account.update_position('X', 'close', self.X[i]) # sell b1*X
                        info = {'status': 'SL', 'spread': self.spread[i], 'balance': self.account.total_balance}
                        pause = self.pause_after_sl
                        self.logs.append((i, info))
                    elif self.spread[i] < last_info['takeprofit']:
                        self.account.update_position('Y', 'close', self.Y[i]) # buy Y
                        self.account.update_position('X', 'close', self.X[i]) # sell b1*X
                        info = {'status': 'TP', 'spread': self.spread[i], 'balance': self.account.total_balance}
                        self.logs.append((i, info))
        return self.account, self.logs
    
    def plot(self, type, figsize=(20, 10), zoom=False, markersize=78):
        self.longs = [i[0] for i in self.logs if i[1]['status'] == 'L']
        self.shorts = [i[0] for i in self.logs if i[1]['status'] == 'S']
        self.stop_loss = [i[0] for i in self.logs if i[1]['status'] == 'SL']
        self.take_profit = [i[0] for i in self.logs if i[1]['status'] == 'TP']
        plot = plt.figure(figsize=figsize)
        plt.axvline(self.trainval_split, linestyle='--', color='black', label='Train/Val Split')
        if type.lower() == 'spread':
            y = self.spread
            plt.plot(y, label='Spread')
            plt.plot(self.ma, linestyle='--', linewidth=.5, color='black', label='{} Tick Moving Average'.format(self.window))
            plt.plot(self.ma+self.z_crit*self.sd, linestyle='--', linewidth=.5, color='black', label='MA + {}*sigma'.format(self.z_crit))
            plt.plot(self.ma-self.z_crit*self.sd, linestyle='--', linewidth=.5, color='black', label='MA - {}*sigma'.format(self.z_crit))
        if type.lower() == 'z':
            y = self.z
            plt.plot(y, label='Z Scores (Standarized Spread)')
            plt.axhline(self.z_crit, linestyle='--', linewidth=.5, color='black')
            plt.axhline(-self.z_crit, linestyle='--', linewidth=.5, color='black')
        plt.scatter(self.longs, y[self.longs], marker=6, s=markersize, color='black', label='Long Spread')
        plt.scatter(self.shorts, y[self.shorts], marker=7, s=markersize, color='black', label='Short Spread')
        plt.scatter(self.stop_loss, y[self.stop_loss], marker='x', s=markersize, color='red', label='Stop Loss')
        plt.scatter(self.take_profit, y[self.take_profit], marker='o', s=markersize, color='green', label='Take Profit')
        plt.legend(loc=3)
        if zoom:
            plt.xlim(self.trainval_split, len(y))
        return plot
    
    def get_df(self):
        idx = []
        values = []
        for i, j in self.logs:
            idx.append(i)
            values.append(j)
        df = pd.DataFrame(values, index=idx)
        df['diffs'] = df['balance'].diff()
        return df