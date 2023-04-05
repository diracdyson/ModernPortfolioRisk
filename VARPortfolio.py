import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.stats.stattools import durbin_watson as db
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.stats.api as sms
from YFpipeline import Pipe
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from FracDiff import Frac



class TimeSeriesModels():
    def __init__(self, d= 0.25 , tau = 1e-3 ):
        
        p = Pipe()
        p.LogChange()
        p.PctChangeAndVol()
        self.X =  p.RemoveFeat()

        #self.X.loc[:,('20DayRet_Close', 'TSLA')] = np.log(self.X.loc[:,('20DayRet_Close', 'TSLA')]) - np.log(self.X.loc[:,('20DayRet_Close', 'TSLA')].shift(1))
       # self.X = self.X.dropna()
        self.X_c = self.X.copy()
        self.ppath = '/Users/teacher/Desktop/PortfolioML/Models/predss/' 
       # self.X = p.PCASVD()
        self.InvTransform = pd.DataFrame(index = np.arange(0,self.X.shape[0], 1))

        for c in self.X.columns:
            f = Frac()
            currf = np.array(f.OGfrac(self.X[c].values.reshape(-1,1),d)).reshape(-1,1)
            self.InvTransform[c] = self.X[c].values.reshape(-1,1) - currf
            self.X[c] = currf


        self.X= self.X.dropna()

        

        # enforce stationairty of feat ('20DayRet_Close', 'TSLA')

        self.X_test  = self.X.iloc[int(0.7 * self.X.shape[0]): self.X.shape[0], :]
        self.X  = self.X.iloc[0:int(0.7 * self.X.shape[0]),:]
        self.tickers = ['TSLA', 'AAPL','MSFT','AMZN','NVDA','GOOGL','SPY']
        self.targetcol = '1DayRet_'+'Close'
        self.DickeyFullerStation(self.X)

      #  self.GrangerTest(self.X)



    @staticmethod
    def ShapiroNormal(res):
        shapiro_test=stats.shapiro(res)
        if shapiro_test.pvalue >= 0.05:
            print('the residuals are normal with pvalue {}'.format(shapiro_test.pvalue))
        else:
            print(' the residuals are not normal with p value {}'.format(shapiro_test.pvalue))

    @staticmethod
    def BreuschHetero(res,X):
        test = sms.het_breuschpagan(res,X,robust=False)
        if test[1] <= 0.05:
            print("heteroskedastic with pvalue {}".format(test[1]))
        else:
            print("homoskedastic with pvalue {}".format(test[1]))


    @staticmethod
    def DickeyFullerStation(X):
        pvalue=[]
        i=0
    
   
        for feature in (X.columns):
            pvalue.append(adfuller(X[feature].values)[1])
            if  pvalue[i] <= 0.05:
                print("Stationary feature {} with p-value {}".format(feature,pvalue[i]))
                i+=1
            else:
                print("Non-Stationary feature {} with p-value {}".format(feature,pvalue[i]))
                i+=1


    @staticmethod
    def GrangerTest(X,lag_order):

        for f1 in X.columns:
            
            for f2 in X.columns:
                print('With respect to {} Linear Combo of {}'.format(f1,f2))
                
                cols = [f1,f2]
                grang = grangercausalitytests(X[cols],[lag_order])

    #https://gist.github.com/18182324/fd8d8751ed3d0874fc3aaf95a09c76fa

    # co integration implementation burrowed from ^ 
    # ensure the Endog obeys cointegration to reduce autocorrelation
    @staticmethod
    def JohansenCoint(endog,lag_order):
        
        res = coint_johansen(endog,k_ar_diff = lag_order, det_order=0)

        output = pd.DataFrame([res.lr2,res.lr1],index=['max_eig_stat',"trace_stat"])
        
        print(output.T,'\n')
        
        print("Critical values(90%, 95%, 99%) of max_eig_stat\n",res.cvm,'\n')
        print("Critical values(90%, 95%, 99%) of trace_stat\n",res.cvt,'\n')
        

    def VectorAR(self,filex = '.csv'):
        
        cnt = 0 
        fig, ax = plt.subplots(len(self.tickers), 1, figsize=(12, 12))
        predictions = pd.DataFrame()
        predictions.index = np.arange(0,self.X_test.shape[0],1)


        endog = self.X.loc[:,(self.targetcol)].values

        model_var = VAR(endog ,exog = self.X.drop(self.targetcol ,axis = 1).values)

        fit = model_var.fit(maxlags = 10,ic = 'aic')

        lag_order = fit.k_ar

        #self.JohansenCoint(endog,lag_order)

       # self.GrangerTest(self.X,lag_order)

        print(fit.summary())

       # fit.plot_acorr()

        forecasts,upper, lower = fit.forecast_interval(y = endog[-lag_order:],steps = self.X_test.shape[0], exog_future = self.X_test.drop(self.targetcol,axis = 1).values)
        
        ts= np.arange(0,self.X_test.shape[0],1)

      #  print(forecasts.shape)
      #  print(upper.shape)
     #   print(lower.shape)

        

           # pm.tsdisplay(self.X.loc[:,(self.targetcol,t)].values.reshape(-1,1), lag_max=90, title="Sunspots", show=True)
             

        for (t, tf) in zip(range(0,len(self.tickers)), self.tickers):
            forecasts1 = forecasts[:,t].reshape(-1,1) - self.InvTransform[(self.targetcol,tf)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]].values.reshape(-1,1)
           # print(fit.test_causality())
            lower1 = lower[:,t].reshape(-1,1) - self.InvTransform[(self.targetcol,tf)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]].values.reshape(-1,1)

            upper1 = upper[:,t].reshape(-1,1) - self.InvTransform[(self.targetcol,tf)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]].values.reshape(-1,1)

            lower1.shape = (self.X_test.shape[0],)
            upper1.shape = (self.X_test.shape[0],)
            
            predictions[self.tickers[t]+'_forecast_'] = forecasts1 
            
            y_test = self.X_test.loc[:,(self.targetcol,tf)].values.reshape(-1,1)   - self.InvTransform[(self.targetcol,tf)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]].values.reshape(-1,1)

            ax[cnt].plot(ts,forecasts1,marker ='o',c = 'r',label='statsmodels: SARIMAX ' +'With order: ' + str(lag_order))
            ax[cnt].plot(ts,y_test,c='b',label='Actual ')
            ax[cnt].fill_between(ts, lower1,upper1, color ='orange',label="Confidence Intervals")
    
            ax[cnt].set_title('Forecast for ' + self.tickers[t])
            ax[cnt].set_xlabel('Time')
            ax[cnt].set_ylabel(self.tickers[t]+' Returns')
            ax[cnt].legend(loc = 'upper left')

            

            # other stats appear within model summary

            #self.ShapiroNormal(residuals)

          #  self.BreuschHetero(residuals,self.X_c.drop(t,axis =1 , level =1).values)
            cnt+= 1 
        
        predictions.to_csv(self.ppath + 'VAR'+'_forecast_'+filex, index = False)
        plt.show()


tsm = TimeSeriesModels()

tsm.VectorAR()