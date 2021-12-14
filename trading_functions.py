# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 21:32:27 2021

@author: Tomasz
"""

#TRADING FUNCTIONS

#The holidays' list is used to calculate trading days between buy and sell without considering holiday days
def create_holidays(start_year, end_year, country='POL'):
    """Create a list of holiday dates in specified country.
    
    Args:
    start_year: int. first year
    end_year: int. last year
    country: string. ISO code of the country

    Returns:
    list of holiday dates    
    """
    import holidays
    holidays_temp = list(dict(holidays.CountryHoliday(country, years=range(start_year, end_year))).keys())
    holidays_final = [x.strftime("%Y-%m-%d") for x in holidays_temp]
    return holidays_final

periods = [5,10,20,50,100,200]
def create_ma_columns(*args):
    """Create list of columns for moving averages
    
    Args:
        numbers representing the days over which to calculate the average
        
    Returns:
        4 lists of
        - sma periods
        - ema periods
        - sma signal periods
        - ema signal periods
    """
    
    #Moving averages combinations
    periods_diff = []
    for period in range(len(args)):
        if period == 0:
            pass
        elif period == 1:
            periods_diff.append(str(args[period]) + '_' + str(args[period-1]))
        else:
            periods_diff.append(str(args[period]) + '_' + str(args[period-2]))
            periods_diff.append(str(args[period]) + '_' + str(args[period-1]))
    #Create columns for moving averages index and signals
    columns_sma = []
    columns_ema = []
    columns_sma_signal = []
    columns_ema_signal = []
    for period in periods_diff:
            columns_sma.append('sma{}'.format(period))
            columns_ema.append('ema{}'.format(period))
            columns_sma_signal.append('sma{}_signal'.format(period))
            columns_ema_signal.append('ema{}_signal'.format(period))
    return columns_sma, columns_ema, columns_sma_signal, columns_ema_signal

#This function calculates moving average(either simple or exponential) for certain period
def calculate_moving_average(stock, period, exp=False):
    """Calculate moving average
    
    Args:
    stock: dataframe. sorted by date containing historical stock data Open, Close, High, Low, Volume
    period: int. Number of periods to calculate average on
    exp: False - simple average, True - exponential average

    Returns:
    Series with moving average 
    """
    if exp==False:
        ma = stock['Close'].rolling(period).mean()
    elif exp==True:
        ma = stock['Close'].ewm(span=period).mean()
    return ma

def find_max(df, ticker, start_date, end_date):
    df_ticker = df[df['Ticker'] == ticker]
    return df_ticker[(df_ticker['Date'] >= start_date) & (df_ticker['Date'] <= end_date)]['Close'].max()

#This function calculate trading indicators like MACD, RSI, etc
def calculate_indexes(stock, periods_diff=['10_5', '20_5', '20_10', '50_10', '50_20', '100_20', '100_50', '200_50', '200_100']):
    """Calculate stock indexes:
    moving average, MACD, rsi, trix, wiliamsR, cci, roc
    
    Args:
    stock: dataframe. sorted by date containing historical stock data Open, Close, High, Low, Volume
    periods_diff: list of strings in format period_period - to calculate moving averages period crossing

    Returns:
    dataframe with index columns   
    """
    
    #Import libraries
    import ta #https://technical-analysis-library-in-python.readthedocs.io/en/latest/index.html
    import pandas as pd
    
    #calculate MACD
    stock['macd_hist'] = ta.trend.MACD(stock['Close'], window_slow = 26, window_fast = 12, window_sign = 9, fillna = False).macd_diff()
    
    #Calculate moving averages
    for period in periods_diff:
        #calculate difference between two moving averages of different periods
        stock['sma{}'.format(period)] = (calculate_moving_average(stock,int(period.split('_')[1])) - calculate_moving_average(stock,int(period.split('_')[0]))) / calculate_moving_average(stock,int(period.split('_')[0])) * 100
        stock['ema{}'.format(period)] = (calculate_moving_average(stock,int(period.split('_')[1]), exp=True) - calculate_moving_average(stock,int(period.split('_')[0]), exp=True)) / calculate_moving_average(stock,int(period.split('_')[0]), exp=True) * 100
    
    #Calculate Relative Strength Index
    stock['rsi_index'] = ta.momentum.RSIIndicator(stock['Close'], window = 14, fillna = False).rsi()
    
    #Calculate Stochastic Oscillator
    stock['sts_index'] = ta.momentum.StochasticOscillator(stock['High'], stock['Low'], stock['Close'], window = 14, smooth_window = 3, fillna = False).stoch_signal()
    
    #Calculate Triple Exponential Average (TRIX)
    stock['trix_index'] = ta.trend.TRIXIndicator(stock['Close'], window = 15, fillna = False).trix()
    
    #Calculate Wiliams R
    stock['wiliamsR'] = ta.momentum.WilliamsRIndicator(stock['High'], stock['Low'], stock['Close'], lbp = 10, fillna = False).williams_r()
    
    #Calculate Commodity Channel Index (CCI)
    #Typical Price High+Low+Close / 3
    stock['cci_index'] = ta.trend.CCIIndicator(stock['High'], stock['Low'], stock['Close'], window = 14, constant = 0.015, fillna = True).cci()
    
    #Calculate Rate Of Change (ROC)
    stock['roc_index'] = ta.momentum.roc(stock['Close'], window = 15, fillna = False)
    
    #Calculate Ultimate Oscillator (ULT)
    stock['ult_index'] = ta.momentum.UltimateOscillator(stock['High'], stock['Low'], stock['Close'], window1 = 7, window2 = 14, window3 = 28, weight1 = 4.0, weight2 = 2.0, weight3 = 1.0, fillna = True).ultimate_oscillator()
    
    #Calculate Force Index (FI)
    stock['fi_index'] = ta.volume.ForceIndexIndicator(stock['Close'], stock['Volume'], window = 13, fillna = False).force_index()
    
    #Calculate Money Flow Index (MFI)
    stock['mfi_index'] = ta.volume.MFIIndicator(stock['High'], stock['Low'], stock['Close'], stock['Volume'], window = 14, fillna = False).money_flow_index()
    
    #Calculate Balance of Power (BOP)
    bop_day = (stock['Close'] - stock['Open']) / (stock['High'] - stock['Low'])
    stock['bop_index'] = bop_day.rolling(14).mean()

    #Calculate Ease of Movement (EOM)
    stock['eom_index'] = ta.volume.EaseOfMovementIndicator(stock['High'], stock['Low'], stock['Volume'], window = 14, fillna = False).sma_ease_of_movement()
    
    #Calculate On Balance Volume
    stock['obv_index'] = ta.volume.OnBalanceVolumeIndicator(stock['Close'], stock['Volume'], fillna = False).on_balance_volume()
    
    #Calculate Awesome Oscillator
    stock['ao_index'] = ta.momentum.AwesomeOscillatorIndicator(stock['High'], stock['Low']).awesome_oscillator()
    
    #Average True Range ATR
    stock['atr_index'] = ta.volatility.AverageTrueRange(stock['High'], stock['Low'], stock['Close'], window=14, fillna=False).average_true_range()
    
    #Volatility ratio = today range/atr
    stock['vr_index'] = pd.DataFrame({'1':abs(stock['High'] - stock['Low']), '2':abs(stock['High'] - stock['Close'].shift(1)), '3':abs(stock['Close'].shift(1) - stock['Low'])}).max(axis=1) / ta.volatility.AverageTrueRange(stock['High'], stock['Low'], stock['Close'], window=14, fillna=False).average_true_range()
    
    #Accumulation / Distribution Index
    stock['adi_index'] = ta.volume.AccDistIndexIndicator(stock['High'], stock['Low'], stock['Close'], stock['Volume'], fillna = False).acc_dist_index()
    
    stock = stock.fillna(0)
    
    return stock

#This function calculate buy/sell signals for the trading indicators. This is based on https://www.biznesradar.pl/information/wskazniki-analizy-technicznej
def calculate_signals(stock, periods_diff=['10_5', '20_5', '20_10', '50_10', '50_20', '100_20', '100_50', '200_50', '200_100']):
    """Calculate buy (1) / sell (-1) signals based on indicators:
    moving average, MACD, rsi, trix, wiliamsR, cci, roc
    
    Args:
    stock: dataframe. sorted by date containing historical stock data Open, Close, High, Low, Volume

    Returns:
    dataframe with index columns   
    """
    
    #Calculate MACD buy/sell signal
    hist_sign = stock['macd_hist'].apply(lambda x: 1 if x>0 else 0)
    stock['macd_signal'] = hist_sign.diff(1) #Trigger a sign, when the histogram changes sign
    
    #Iterate thorugh moving averages - Moving average loop
    for period in periods_diff:
        #calculate signal, when two moving average lines cross
        stock['sma{}_signal'.format(period)] = stock['sma{}'.format(period)].apply(lambda x: 1 if x>0 else 0).diff(1)
        stock['ema{}_signal'.format(period)] = stock['ema{}'.format(period)].apply(lambda x: 1 if x>0 else 0).diff(1)
        
    #RSI signal buy - RSI in range(25-75) & RSI>RSI avg last 4 days
    stock.loc[(stock['rsi_index'] > 25) & (stock['rsi_index'] < 75) & (stock['rsi_index'] > stock['rsi_index'].shift(1).rolling(4).mean()), 'rsi_signal'] = 1
    stock.loc[(stock['rsi_index'] > 25) & (stock['rsi_index'] < 75) & (stock['rsi_index'] < stock['rsi_index'].shift(1).rolling(4).mean()), 'rsi_signal'] = -1
    stock['rsi_signal'] = stock['rsi_signal'].fillna(0)
    
    #STS signal buy - STS in range(20,80) & STS>STS avg 4 days
    stock.loc[(stock['sts_index'] > 20) & (stock['sts_index'] < 80) & (stock['sts_index'] > stock['sts_index'].shift(1).rolling(4).mean()), 'sts_signal'] = 1
    stock.loc[(stock['sts_index'] > 20) & (stock['sts_index'] < 80) & (stock['sts_index'] < stock['sts_index'].shift(1).rolling(4).mean()), 'sts_signal'] = -1
    stock['sts_signal'] = stock['sts_signal'].fillna(0)
    
    #Calculate trix signal
    stock['trix_signal'] = stock['trix_index'].apply(lambda x: 1 if x>0 else -1)
    
    #WiliamsR signal buy - wiliamsR in range(-0.8,-0,2) and wiliamsR>wiliamsR avg 4 days
    stock.loc[(stock['wiliamsR'] > -0.8) & (stock['wiliamsR'] < -0.2) & (stock['wiliamsR'] > stock['wiliamsR'].shift(1).rolling(4).mean()), 'wiliamsR_signal'] = 1
    stock.loc[(stock['wiliamsR'] > -0.8) & (stock['wiliamsR'] < -0.2) & (stock['wiliamsR'] < stock['wiliamsR'].shift(1).rolling(4).mean()), 'wiliamsR_signal'] = -1
    stock['wiliamsR_signal'] = stock['wiliamsR_signal'].fillna(0)
    
    #CCI signal buy - cciin range(-200,200) & cci>cci avf 4 days
    stock.loc[(stock['cci_index'] > -200) & (stock['cci_index'] < 200) & (stock['cci_index'] > stock['cci_index'].shift(1).rolling(4).mean()), 'cci_signal'] = 1
    stock.loc[(stock['cci_index'] > -200) & (stock['cci_index'] < 200) & (stock['cci_index'] < stock['cci_index'].shift(1).rolling(4).mean()), 'cci_signal'] = -1
    stock['cci_signal'] = stock['cci_signal'].fillna(0)
        
    #ROC signal
    stock['roc_signal'] = stock['roc_index'].apply(lambda x: 1 if x>0 else -1)
    
    #ULT signal buy - ult in range(30,70) & ult>ult avg 4 days
    stock.loc[(stock['ult_index'] > 30) & (stock['ult_index'] < 70) & (stock['ult_index'] > stock['ult_index'].shift(1).rolling(4).mean()), 'ult_signal'] = 1
    stock.loc[(stock['ult_index'] > 30) & (stock['ult_index'] < 70) & (stock['ult_index'] < stock['ult_index'].shift(1).rolling(4).mean()), 'ult_signal'] = -1
    stock['ult_signal'] = stock['ult_signal'].fillna(0)
    
    #FI signal
    stock['fi_signal'] = stock['fi_index'].apply(lambda x: 1 if x>0 else -1)
    
    #MFI signal buy - mfi in range(25,75) mfi>mfi avg 4 days
    stock.loc[(stock['mfi_index'] > 25) & (stock['mfi_index'] < 75) & (stock['mfi_index'] > stock['mfi_index'].shift(1).rolling(4).mean()), 'mfi_signal'] = 1
    stock.loc[(stock['mfi_index'] > 25) & (stock['mfi_index'] < 75) & (stock['mfi_index'] < stock['mfi_index'].shift(1).rolling(4).mean()), 'mfi_signal'] = -1
    stock['mfi_signal'] = stock['mfi_signal'].fillna(0)
    
    #BOP signal
    stock['bop_signal'] = stock['bop_index'].apply(lambda x: 1 if x>0 else -1)
    
    #EMV indicator
    stock['eom_signal'] = stock['eom_index'].apply(lambda x: 1 if x>0 else -1)
    
    #Awesome Oscillator signal
    stock.loc[stock['ao_index'] > stock['ao_index'].shift(1), 'ao_signal'] = 1
    stock.loc[stock['ao_index'] < stock['ao_index'].shift(1), 'ao_signal'] = -1
    stock['ao_signal'] = stock['ao_signal'].fillna(0)
    
    return stock

#This is to calculate features for machine learning model based on price and volume
#The features are my personal try. No science behind
def calculate_features(stock):
    """Calculate features for machine learning
    
    Args:
    stock: dataframe. sorted by date containing historical stock data Open, Close, High, Low, Volume

    Returns:
    dataframe with index columns   
    """    
    #VOLUME
    #Volume on signal day vs average volume on previous days
    stock['p_volume_1'] = ((stock['Volume'] - stock['Volume'].shift(1)) / stock['Volume'].shift(1)) * 100
    stock['p_volume_5'] = ((stock['Volume'] - stock['Volume'].shift(1).rolling(5).mean()) / stock['Volume'].shift(1).rolling(5).mean()) * 100
    stock['p_volume_10'] = ((stock['Volume'] - stock['Volume'].shift(1).rolling(10).mean()) / stock['Volume'].shift(1).rolling(10).mean()) * 100
    stock['p_volume_20'] = ((stock['Volume'] - stock['Volume'].shift(1).rolling(20).mean()) / stock['Volume'].shift(1).rolling(20).mean()) * 100 
    
    #PRICE
    #Price on the day of the signal
    stock['p_day_close'] = (stock['Close'] - stock['Open']) / stock['Open'] * 100
    stock['p_day_high'] = (stock['High'] - stock['Low']) / stock['Low'] * 100
    
    #Price vs previous day
    stock['p_day_close1'] = (stock['Close'] - stock['Close'].shift(1)) / stock['Close'].shift(1) * 100
    stock['p_day_close2'] = (stock['Close'] - stock['Close'].shift(2)) / stock['Close'].shift(2) * 100
    stock['p_day_close3'] = (stock['Close'] - stock['Close'].shift(3)) / stock['Close'].shift(3) * 100
    stock['p_day_close4'] = (stock['Close'] - stock['Close'].shift(4)) / stock['Close'].shift(4) * 100
    stock['p_day_close5'] = (stock['Close'] - stock['Close'].shift(5)) / stock['Close'].shift(5) * 100
    stock['p_day_open1'] = (stock['Open'] - stock['Close'].shift(1)) / stock['Close'].shift(1) * 100
    
    #Price vs previous days average
    stock['p_avg_close3'] = (stock['Close'] - stock['Close'].shift(1).rolling(3).mean()) / stock['Close'].shift(1).rolling(3).mean() * 100
    stock['p_avg_close5'] = (stock['Close'] - stock['Close'].shift(1).rolling(5).mean()) / stock['Close'].shift(1).rolling(5).mean() * 100
    stock['p_avg_close10'] = (stock['Close'] - stock['Close'].shift(1).rolling(10).mean()) / stock['Close'].shift(1).rolling(10).mean() * 100
    stock['p_avg_close20'] = (stock['Close'] - stock['Close'].shift(1).rolling(20).mean()) / stock['Close'].shift(1).rolling(20).mean() * 100

    #Price vs previous days price range
    stock['p_range_close3'] = (stock['Close'].shift(1).rolling(3).max()-stock['Close'].shift(1).rolling(3).min()) / stock['Close'] * 100
    stock['p_range_open3'] = (stock['Close'].shift(1).rolling(3).max()-stock['Close'].shift(1).rolling(3).min()) / stock['Open'] * 100
    stock['p_range_close5'] = (stock['Close'].shift(1).rolling(5).max()-stock['Close'].shift(1).rolling(5).min()) / stock['Close'] * 100
    stock['p_range_open5'] = (stock['Close'].shift(1).rolling(5).max()-stock['Close'].shift(1).rolling(5).min()) / stock['Open'] * 100
    stock['p_range_close10'] = (stock['Close'].shift(1).rolling(10).max()-stock['Close'].shift(1).rolling(10).min()) / stock['Close'] * 100
    stock['p_range_open10'] = (stock['Close'].shift(1).rolling(10).max()-stock['Close'].shift(1).rolling(10).min()) / stock['Open'] * 100
    
    #Price vs previous days max Close and High
    stock['p_max_close5'] = (stock['Close'] - stock['Close'].shift(1).rolling(5).max()) / stock['Close'].shift(1).rolling(5).max() * 100
    stock['p_max_close10'] = (stock['Close'] - stock['Close'].shift(1).rolling(10).max()) / stock['Close'].shift(1).rolling(10).max() * 100
    stock['p_max_close20'] = (stock['Close'] - stock['Close'].shift(1).rolling(20).max()) / stock['Close'].shift(1).rolling(20).max() * 100   
    stock['p_max_high5'] = (stock['Close'] - stock['High'].shift(1).rolling(5).mean()) / stock['High'].shift(1).rolling(5).mean() * 100
    stock['p_max_high10'] = (stock['Close'] - stock['High'].shift(1).rolling(10).mean()) / stock['High'].shift(1).rolling(10).mean() * 100
    stock['p_max_high20'] = (stock['Close'] - stock['High'].shift(1).rolling(20).mean()) / stock['High'].shift(1).rolling(20).mean() * 100
    
    #Compare close price at signal with open price purchase
    stock['ps_open_close'] = (stock['Open'] - stock['Close'].shift(1)) / stock['Close'].shift(1) * 100
    
    stock = stock.fillna(0)
    
    return stock