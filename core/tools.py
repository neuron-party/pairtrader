import numpy as np
from scipy.stats import *
from statsmodels.tsa.stattools import adfuller

def rollingMA(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    ma = (cumsum[n:] - cumsum[:-n]) / n
    ma_padded = np.insert(ma, 0, np.full(n-1, np.nan))
    return ma_padded

def rollingSD(x, n):
    sd = np.full(len(x), np.nan)
    for i in range(n, len(x)+1):
        window = x[i-n:i]
        sd[i-1] = np.std(window, ddof=1)
    return sd

def fit_model(X, Y):
    b1, b0 = linregress(x=X, y=Y)[0:2]
    resids = Y - (b1*X + b0)
    adf_p = adfuller(resids, autolag='AIC')[1] # augmented dickey fuller p-value
    return b1, b0, adf_p