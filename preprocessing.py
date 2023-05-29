import numpy as np
import talib as tb

def preprocess(df):
	df = features(df)
	df = df.drop(["Open", "High", "Low", "Close", "Volume", "Adj Close"], axis=1).dropna()
	return df

def features(data):
    
    for i in [2, 3, 4, 5, 6, 7]:
                            
        # Rolling Mean
        data[f"Close{i}"] = data["Close"].rolling(i).mean()
        data[f"Volume{i}"] = data["Volume"].rolling(i).mean()
        
        # Rolling Standart Deviation                               
        data[f"Low_std{i}"] = data["Low"].rolling(i).std()
        data[f"High_std{i}"] = data["High"].rolling(i).std()
        data[f"CLose{i}"] = data["Close"].rolling(i).std()
        
        # Stock return for the next i days
        data[f"Close{i}"] = data["Close"].shift(i)
        
        # Rolling Maximum and Minimum
        data[f"Close{i}"] = data["Close"].rolling(i).max()
        data[f"Close{i}"] = data["Close"].rolling(i).min()
        
        # Rolling Quantile
        data[f"Close{i}"] = data["Close"].rolling(i).quantile(1)
    
    
    
    #Decoding the time of the year
    data["Day"] = data.index.day
    data["Month"] = data.index.month
    data["Year"] = data.index.year
    data["day_year"] = data.index.day_of_year
    data["Weekday"] = data.index.weekday
                  
    #Upper and Lower shade
    data["Upper_Shape"] = data["High"]-np.maximum(data["Open"], data["Close"])
    data["Lower_Shape"] = np.minimum(data["Open"], data["Close"])-data["Low"]

    data['EMA_9'] = data['Close'].ewm(9).mean().shift()
    data['SMA_5'] = data['Close'].rolling(5).mean().shift()
    data['SMA_10'] = data['Close'].rolling(10).mean().shift()
    data['SMA_15'] = data['Close'].rolling(15).mean().shift()
    data['SMA_30'] = data['Close'].rolling(30).mean().shift()
    data['MA5'] = tb.MA(data["Close"], timeperiod=5)
    data['MA10'] = tb.MA(data["Close"], timeperiod=10)
    data['MA20'] = tb.MA(data["Close"], timeperiod=20)
    data['MA60'] = tb.MA(data["Close"], timeperiod=60)
    data['MA120'] = tb.MA(data["Close"], timeperiod=120)
    data['MA5'] = tb.MA(data["Volume"], timeperiod=5)
    data['MA10'] = tb.MA(data["Volume"], timeperiod=10)
    data['MA20'] = tb.MA(data["Volume"], timeperiod=20)
    data['ADX'] = tb.ADX(data["High"], data["Low"], data["Close"], timeperiod=14)
    data['ADXR'] = tb.ADXR(data["High"], data["Low"], data["Close"], timeperiod=14)
    data['MACD'] = tb.MACD(data["Close"], fastperiod=12, slowperiod=26, signalperiod=9)[0]
    data['RSI'] = tb.RSI(data["Close"], timeperiod=14)
    data['BBANDS_U'] = tb.BBANDS(data["Close"], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0]
    data['BBANDS_M'] = tb.BBANDS(data["Close"], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1]
    data['BBANDS_L'] = tb.BBANDS(data["Close"], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2]
    data['AD'] = tb.AD(data["High"], data["Low"], data["Close"], data["Volume"])
    data['ATR'] = tb.ATR(data["High"], data["Low"], data["Close"], timeperiod=14)
    data['HT_DC'] = tb.HT_DCPERIOD(data["Close"])


    return data

