from fredapi import Fred
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date
import yfinance as yf
from sklearn.preprocessing import StandardScaler
class MacroPipeline():
    def __init__(self):
        def download(codes,date2):
      
                                                                                                                                                      

            key='bb358fb7479df683ef3c8fb6df7c3ebf'
            fred = Fred(api_key=key)
            data={}
            for i in codes.values():
                data[i]=fred.get_series(i,observation_start=date2)
            macro=pd.DataFrame.from_dict(data)
    
            return macro

   
        period = '10y'
        interval = '1mo'
        self.tickers = ['TSLA', 'AAPL','MSFT','AMZN','NVDA','GOOGL','SPY']
        self.feat = ['High','Low','Close','Volume']
        self.X = yf.download(tickers ='SPY',period = period, interval = interval)['Close']

        codes={'Corporate': 'BAA10Y','10Y':'DGS10','1Y': 'DGS1'}
        macro = download( codes,self.X.index[0]).resample('MS').ffill()

       #macro=download(codes)
        macro.columns=list(codes.keys())


       # macro['Time']=((macro['10Y']-macro['1Y'])/((macro['10Y']).expanding().mean()))*100
        macro['Confidence']=((macro['10Y']-macro['Corporate'])/((macro['10Y']).expanding().mean()))*100
        macro = pd.DataFrame(StandardScaler().fit_transform(macro.values),columns = macro.columns,index = macro.index)


        print(self.X.index[0])
        print(macro.index[0])

        
     #   print(macro.head())
     #   print(self.X.shape)
        newjawn = pd.concat([macro,self.X],axis =1)
        print(newjawn.head()) 


m = MacroPipeline()