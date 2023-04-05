import quandl
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
today = dt.date.today()
diff = dt.timedelta(weeks = 52* 10)
tickers = ['WIKI/TSLA', 'WIKI/AAPL']
        #   'WIKI/MSFT','WIKI/AMZN','WIKI/NVDA','WIKI/GOOGL','WIKI/SPY']

yearsago = today - diff
d = quandl.get(['WIKI/TSLA','WIKI/AAPL'],start_date= str(yearsago), end_date= str(today))
#print(d.columns)
d2 = pd.DataFrame(index = d.index)
for t in tickers:
    d2[('Close',t)] = d[t+' - Close']
    d2[('High',t)] = d[t+' - High']
    d2[('Open',t)] =d[t +' - Open']
print(d2.loc[:,('Close',tickers[0])].tail()) 

