# -*- coding: utf-8 -*-
"""
@author: Tomasz Porzycki
Trade profitability analysis for trades based on various indicators signals
Mainly used MACD, simple moving average, expontential moving average
Assumptions:
    1) Trade is profitable if, profit >0
    2) Buy / sell happen the following day of the signal
    3) Buy / sell are taken 10% from the open price towards close price
"""

#Import libraries
import pandas_datareader.data as pdr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
from bs4 import BeautifulSoup
import requests
from sklearn.preprocessing import MinMaxScaler

from trading_functions import *

#pd.set_option('display.max_columns', None)

#DATA PREPARATION FOR ANALYSIS
#The starting year for analysis
year_start = 2000

#Original columns in the stock data download
columns_original = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']

#Trading indexes
columns_index = ['macd_hist', 'rsi_index', 'sts_index', 'trix_index', 'wiliamsR', 'cci_index', 'roc_index', 'ult_index', 'fi_index', 'mfi_index', 'bop_index', 'eom_index', 'obv_index', 'ao_index', 'atr_index', 'vr_index', 'adi_index']

#Buy/sell signals based on trading indexes
columns_signals = ['macd_signal', 'rsi_signal', 'sts_signal', 'trix_signal', 'wiliamsR_signal', 'cci_signal', 'roc_signal', 'ult_signal', 'fi_signal', 'mfi_signal', 'bop_signal', 'eom_signal', 'ao_signal']

#Moving averages periods
columns_sma = ['sma10_5', 'sma20_5', 'sma20_10', 'sma50_10', 'sma50_20', 'sma100_20', 'sma100_50', 'sma200_50', 'sma200_100']
columns_ema = ['ema10_5', 'ema20_5', 'ema20_10', 'ema50_10', 'ema50_20', 'ema100_20', 'ema100_50', 'ema200_50', 'ema200_100']
columns_sma_signal = ['sma10_5_signal', 'sma20_5_signal', 'sma20_10_signal', 'sma50_10_signal', 'sma50_20_signal', 'sma100_20_signal', 'sma100_50_signal', 'sma200_50_signal', 'sma200_100_signal']
columns_ema_signal = ['ema10_5_signal', 'ema20_5_signal', 'ema20_10_signal', 'ema50_10_signal', 'ema50_20_signal', 'ema100_20_signal', 'ema100_50_signal', 'ema200_50_signal', 'ema200_100_signal']

#Features calculated for modelling
columns_features = ['p_volume_1', 'p_volume_5', 'p_volume_10', 'p_volume_20',
                    'p_day_close', 'p_day_high',
                    'p_day_close1', 'p_day_close2', 'p_day_close3', 'p_day_close4', 'p_day_close5',
                    'p_day_open1',
                    'p_avg_close3', 'p_avg_close5', 'p_avg_close10', 'p_avg_close20',
                    'p_range_close3', 'p_range_open3', 'p_range_close5', 'p_range_open5', 'p_range_close10', 'p_range_open10',
                    'p_max_close5', 'p_max_close10', 'p_max_close20', 'p_max_high5', 'p_max_high10', 'p_max_high20']

columns_features_purchase_open = ['ps_open_close']

#Additional columns analysis
columns_analysis = ['type', 'Date_purchase', 'Open_purchase', 'Close_purchase', 'invested_stocks', 'invested_money', 'profit', 'trade_days', 'profitable']
#Additional columns analysis
columns_analysis_today = ['type']
columns_features_analysis = ['profitable_1', 'profitable_2', 'profitable_3', 'previous_exit1', 'previous_exit2', 'previous_exit3']
signals_for_analysis = ['sma50_20_signal','sma50_10_signal','sma20_5_signal','sma20_10_signal','sma10_5_signal','sma100_50_signal','macd_signal']

#Calculate holidays - to calculate the duration of trades excluding weekends and holidays
holidays = create_holidays(start_year=year_start, end_year=datetime.today().year+1, country='POL')

#DOWNLOAD HISTORICAL STOCK DATA FROM STOOQ
#Read tickers to check on Stooq
tickers = pd.read_csv('000_gpw_tickers.csv')
#Filter tickers with volume >1000 to eliminate small companies
tickers = list(tickers[tickers['Volume'] > 1000]['Ticker'])

#Load the previously saved stock data history
gpw_stocks = pd.read_csv('001_gpw_stocks_history.csv', index_col=0, parse_dates=[2], dayfirst=True)

#Define the date to update the stock data
date_start = gpw_stocks['Date'].max() + timedelta(days = 1)
#date_start = datetime(1990,1,1).date()
date_end = datetime.now()

#Download stock info for all tickers
for ticker in tickers:
    #Download stock info from Stooq
    stock = pdr.get_data_stooq('{}.PL'.format(ticker), start=date_start, end=date_end)
    #Move date from index to column
    stock = stock.reset_index()
    #Order according date
    try:
        stock.sort_values('Date', axis=0, ascending=True, inplace=True)
    except:
        pass
    #Insert Ticker info in first column
    stock.insert(loc=0, column='Ticker', value=ticker)
    
    #Append ticker stocks to database
    gpw_stocks = gpw_stocks.append(stock, ignore_index=True)
    
#Save result to file
gpw_stocks['Volume'].fillna(1, inplace=True)
gpw_stocks.to_csv('001_gpw_stocks_history.csv')

#CALCULATE INDICATORS
print("Start time:", datetime.now())
#Create dataframe which stores all data stocks with MACD
columns_stock_signals =  columns_original + columns_sma + columns_ema + columns_sma_signal + columns_ema_signal + columns_index + columns_signals + columns_features + columns_features_purchase_open
gpw_stocks_signals = pd.DataFrame(columns = columns_stock_signals)

#A loop to calculate macd for all stocks
for ticker in set(gpw_stocks['Ticker']):
    #select the stock info related to the ticker
    stock = gpw_stocks[gpw_stocks['Ticker'] == ticker]
  
    stock = calculate_indexes(stock)
    stock = calculate_signals(stock)
    stock = calculate_features(stock)
    #stock = stock[stock['Date'] >= '{}-01-01'.format(year_start)]
    #Exclude data for the first year from market entry to avoid 0 values
    stock = stock[stock['Date'] >= stock['Date'].min() + timedelta(days = 365)]
    gpw_stocks_signals = gpw_stocks_signals.append(stock[columns_stock_signals], ignore_index=True)
    
    #Counter
    #print('Completion: {} %'.format(round(np.where(gpw_stocks['Ticker'].unique() == ticker)[0][0]/len(gpw_stocks['Ticker'].unique())*100,2)))
    
print("End time:", datetime.now())

#Remove EKP, because irregular
gpw_stocks_signals = gpw_stocks_signals.drop(gpw_stocks_signals[gpw_stocks_signals['Ticker'] == 'EKP'].index, axis=0)

gpw_stocks_signals.to_csv('002_gpw_stocks_signals.csv')

#CALCULATE HISTORICAL TRADES PERFORMANCE
#Parameters
invest = 10000 #I assume to invest 10000 for each trade
percent = 0.1 #I assume buying at price 20% from the open towards close price

#Create dataframe to store results
analysis_result = pd.DataFrame(columns = columns_original + columns_analysis + columns_features_analysis + columns_signals + columns_sma + columns_ema + columns_sma_signal + columns_ema_signal + columns_index + columns_features + columns_features_purchase_open)

print("Start time:", datetime.now())
#Make analysis ticker per ticker
for ticker in set(gpw_stocks_signals['Ticker']):
    #Filter the rows for the Ticker
    stock_analysis = gpw_stocks_signals[gpw_stocks_signals['Ticker'] == ticker]
    
    #Shift by 1 day, because purchase happens the day after
    stock_analysis[columns_signals + columns_sma + columns_ema + columns_sma_signal + columns_ema_signal + columns_index + columns_features] = stock_analysis[columns_signals + columns_sma + columns_ema + columns_sma_signal + columns_ema_signal + columns_index + columns_features].shift(1)
    
    #Make profit analysis based on differet signals - SIGNAL LOOP
    for signal in ['macd_signal'] + columns_sma_signal + columns_ema_signal: #signals_for_analysis:
        
        #filter only the rows with signal
        analysis = stock_analysis[(stock_analysis[signal]==1) | (stock_analysis[signal]==-1)]
        
        #if the first row signal is 'sell' - remove row
        try:
            if analysis.iloc[0][signal] == -1:
                analysis.drop(analysis.index[0], axis=0, inplace=True)
            else:
                pass
        except:
            pass
            
        #As everything will be analyzed in exit I shift by 1
        analysis[['Date_purchase', 'Open_purchase', 'Close_purchase']] = analysis[['Date', 'Open', 'Close']].shift(1)
        analysis[columns_sma + columns_ema + columns_sma_signal + columns_ema_signal + columns_index + columns_signals + columns_features + columns_features_purchase_open] = analysis[columns_sma + columns_ema + columns_sma_signal + columns_ema_signal + columns_index + columns_signals + columns_features + columns_features_purchase_open].shift(1)
    
        #OVERALL PROFIT ANALYSIS
        #filter rows with buy signal
        analysis = analysis[analysis[signal]==1]
        
        #calculate invested stocks for each purchase
        analysis['invested_stocks'] = (invest / analysis['Open_purchase']).astype('int') #Calculate number of stocks to be purchased
        analysis['invested_money'] = (analysis['Open_purchase'] + (analysis['Close_purchase'] - analysis['Open_purchase']) * percent) * analysis['invested_stocks']
        analysis['profit'] = (analysis['Open'] + (analysis['Close'] - analysis['Open']) * percent) * analysis['invested_stocks'] - analysis['invested_money']
        analysis['trade_days'] = np.busday_count(analysis['Date_purchase'].fillna(datetime(1900,1,1)).astype('str').tolist(), analysis['Date'].fillna(datetime(1900,1,1)).astype('str').tolist(), holidays=holidays)
        analysis['profitable'] = analysis['profit'].apply(lambda x: 1 if x>0 else 0)
        
        #Calculate additional features
        analysis['profitable_1'] = analysis['profitable'].shift(1)
        analysis['profitable_2'] = analysis['profitable'].shift(2)
        analysis['profitable_3'] = analysis['profitable'].shift(3)
        analysis['previous_exit1'] = np.busday_count(analysis['Date'].shift(1).fillna(datetime(1900,1,1)).astype('str').tolist(), analysis['Date_purchase'].astype('str').tolist(), holidays=holidays)
        analysis['previous_exit2'] = np.busday_count(analysis['Date'].shift(2).fillna(datetime(1900,1,1)).astype('str').tolist(), analysis['Date_purchase'].astype('str').tolist(), holidays=holidays)
        analysis['previous_exit3'] = np.busday_count(analysis['Date'].shift(3).fillna(datetime(1900,1,1)).astype('str').tolist(), analysis['Date_purchase'].astype('str').tolist(), holidays=holidays)
        
        analysis = analysis.tail(-3) #Delete the first 2 rows, where previous exit is unavailable. Previous exit is one of the most important features, therefore better to keep it accurate
    
        #Specify which signal is trigered
        analysis['type'] = signal
        
        analysis_result = analysis_result.append(analysis, ignore_index=True)

#Order columns
analysis_result = analysis_result[columns_original + columns_analysis + columns_features_analysis + columns_signals + columns_sma + columns_ema + columns_sma_signal + columns_ema_signal + columns_index + columns_features + columns_features_purchase_open]
        
print("End time:", datetime.now())

analysis_result.to_csv('003_analysis_result.csv')