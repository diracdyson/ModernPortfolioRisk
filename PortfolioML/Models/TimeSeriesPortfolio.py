from YFpipeline import Pipe
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.stattools import durbin_watson as db
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from scipy import stats
import pickle5 as pickle
import numpy as np
import pandas as pd 

class TimeSeriesModels():
    def __init__(self,lagd= False):
        
        p = Pipe()
        p.LogChange()
        p.PctChangeAndVol()
        self.X =  p.RemoveFeat()
        self.X.loc[:,('20DayRet_Close', 'TSLA')] = np.log(self.X.loc[:,('20DayRet_Close', 'TSLA')]) - np.log(self.X.loc[:,('20DayRet_Close', 'TSLA')].shift(1))
        self.X= self.X.dropna()
        self.X_c = self.X.copy()
        self.ppath = '/Users/teacher/Desktop/PortfolioML/Models/predss/' 
        self.X_test = self.X.iloc[int(0.9* self.X.shape[0]):self.X.shape[0],:]
        self.X=self.X.iloc[0:int(0.9* self.X.shape[0]),:]

        self.DickeyFullerStation(self.X)
        self.tickers = ['TSLA', 'AAPL','MSFT','AMZN','NVDA','GOOGL','SPY']
        self.targetcol = '1DayRet_'+'Close'
        

        if lagd:
            fig, ax = plt.subplots(len(self.tickers),1,figsize =(50,60))
            fig2, ax2 = plt.subplots(len(self.tickers),1,figsize =(50,60))
            for ind, t in enumerate(self.tickers):

                plot_acf(self.X.loc[:,(self.targetcol,t)],title="acf "+t,ax=ax[ind])
                plot_pacf(self.X.loc[:,(self.targetcol,t)],title="pacf "+t,method='ywmle', ax=ax2[ind])
        
            plt.show()

    @staticmethod
    def SaveModel(model,model_file_name,path=None):
        pickle.dump(model,open(path+model_file_name+'.pkl',"wb"))

    @staticmethod  
    def LoadModel(model_file_name,path=None)->object: 
        model = pickle.load(open(path+model_file_name+'.pkl',"rb"))
        return model


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
    def forecast_one_step(model):
        fc, conf_int = model.predict(n_periods=1, return_conf_int=True)
        return (
        fc.tolist()[0],
        np.asarray(conf_int).tolist()[0])


    def SARIMAX(self,filex = '.csv'):
        cnt = 0 
        fig, ax = plt.subplots(len(self.tickers), 1, figsize=(12, 12))
        
        for t in self.tickers:

           # pm.tsdisplay(self.X.loc[:,(self.targetcol,t)].values.reshape(-1,1), lag_max=90, title="Sunspots", show=True)
             
            stepwise_fit = pm.auto_arima(self.X.loc[:,(self.targetcol,t)].values.reshape(-1,1), start_p=1, start_q=1,
                             max_p=3, max_q=3, m=1,
                             start_P=0, seasonal=False,
                             d=2, D=0, trace=True,
                             error_action='trace',
                             suppress_warnings=True, 
                             stepwise=True) 
            
            # stepwise_fit.summary()
            #print((stepwise_fit.get_params()))

            stepwise_dicta = stepwise_fit.get_params()

            model_a = pm.arima.ARIMA([],suppress_warnings=True).set_params(**(stepwise_dicta))
             
             #include exo terms after fitting SARIMAX 

            model_a.fit(self.X.loc[:,(self.targetcol,t)].values.reshape(-1,1),exog = self.X.drop(t,axis= 1, level =1) )
        
            model_a.summary()

           # model_a.plot_diagnostics()

            forecastsrobust=[]
            confidence_intervalsrobust=[]
            
            y_test = self.X_test.loc[:,(self.targetcol,t)].values.reshape(-1,1)
            ts = np.arange(0,self.X_test.shape[0],1)
            predictions = pd.DataFrame(index = ts)
            
            for new_ob in y_test:
    
                fc,conf = self.forecast_one_step(model_a)
                
                forecastsrobust.append(fc)
                confidence_intervalsrobust.append(conf)
    
    
                model_a.update(new_ob)

            
            forecastsrobust=np.array(forecastsrobust).reshape(-1,1)
            CI=np.array(confidence_intervalsrobust)

            predictions[t+'_forecast_'] = forecastsrobust

            ax[cnt].plot(ts,forecastsrobust,marker ='o',c = 'r',label='statsmodels: SARIMAX ' +'With order: ' + str(stepwise_dicta['order']))
            ax[cnt].plot(ts,y_test,c='b',label='Actual ')
            ax[cnt].fill_between(ts, CI[:,0],CI[:,1], color ='orange',label="Confidence Intervals")

            ax[cnt].set_title('Forecast for ' + t)
            ax[cnt].set_xlabel('Time')
            ax[cnt].set_ylabel(t+' Returns')

            ax[cnt].legend(loc = 'upper left')

            residuals = model_a.resid()

            # other stats appear within model summary

            self.ShapiroNormal(residuals)

            self.BreuschHetero(residuals,self.X_c.drop(t,axis =1 , level =1).values)
            cnt+= 1 
        
        predictions.to_csv(self.ppath + 'SARIMAX'+'_forecast_'+filex)
        plt.show()

tsm = TimeSeriesModels()
tsm.SARIMAX()


        
