import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from YFpipeline import Pipe
import matplotlib.pyplot as plt

class Frac():

    def OGfrac(self,x, d):
  
        if np.isnan(np.sum(x)):
            return None

        n = len(x)
        if n < 2:
            return None

        x = np.subtract(x, np.mean(x))

    # calculate weights
        weights = [0] * n
        weights[0] = -d
        for k in range(2, n):
            weights[k - 1] = weights[k - 2] * (k - 1 - d) / k

    # difference series
        ydiff = list(x)

        for i in range(0, n):
            dat = x[:i]
            w = weights[:i]
            ydiff[i] = x[i] + np.dot(w, dat[::-1])

        return ydiff
    
    @staticmethod
    def getWeights(d, lags):
        # obtain the weights via the taylor expansion of the binomial backshift operator
        w=[1]
        for k in range(1,lags):
            w.append(-w[-1]* ((d-k+1))/k)
        w=np.array(w).reshape(-1,1)
        
        return w

    def CutOffFind(self,order,cutoff,start_lags):
        val  = np.inf
        lags = start_lags
        
        while abs(val) > cutoff:
            w = self.getWeights(order,lags)
            val = w[len(w)-1]
            lags +=1
    
        return lags  
    
    # extend from staticmethod into self class access staticmethod 
    # loop over columns
    
    def TSDiff(self,series,order, tau,st = 1):
        lag_cutoff = self.CutOffFind(order,tau,st)
        print(lag_cutoff)
        w = self.getWeights(order,lag_cutoff)

        res = 0 
        # adapt series over columns 
        
        
        for k in range(lag_cutoff):
            res += w[k] * series.shift(k).fillna(0)
        return res[lag_cutoff:]
    

    @staticmethod
    def DickeyFullerStation(X,feature):
        pvalue=[]
        i=0
    
        #for feature in (X.columns):
        pvalue = adfuller(X.values.reshape(-1,1))[1]
        if  pvalue <= 0.05:
            print("Stationary feature {} with p-value {}".format(feature,pvalue))
            i+=1
        else:
            print("Non-Stationary feature {} with p-value {}".format(feature, pvalue))
            i+=1
    
    def GraphTSDiff(self,X, d=0.6):
        
        fig , ax = plt.subplots(self.X.shape[1],1, figsize = (15,6))
        cnt = 0 
        self.X = X

        # insert Deflated sharpe ratio
        tau = 1e-3
        for c in self.X.columns:

            diffseries = self.TSDiff(self.X[c],d,tau )

            self.DickeyFullerStation(diffseries,c)

            print('Regular log series')
            
            self.DickeyFullerStation(self.X[c],c)

            ts1 = np.arange(0,self.X.shape[0],1)

            ts2 = np.arange(0,len(diffseries),1)

            ax[cnt].plot(ts1,self.X[c], c='r',label = 'Log Change')
            ax[cnt].plot(ts2,diffseries,c = 'b',label = 'Frac diff' + str(d))

            ax[cnt].set_title('Series Graphs')
            ax[cnt].set_ylabel(' Log_Returns ')
            ax[cnt].set_xlabel('TTM')
            ax[cnt].legend()
            cnt +=1 

        plt.show()
 
    
#f= Frac()
#f.GraphTSDiff()

