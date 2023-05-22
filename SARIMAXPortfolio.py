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
from statsmodels.tsa.stattools import kpss
from FracDiff import Frac
from statsmodels.tsa.api import ARDL
from statsmodels.tsa.ardl import ardl_select_order
from arch.univariate import ARX, StudentsT, GARCH

class TimeSeriesModels():
    
    def __init__(self,lagd= False,d=0.35,perc = 0.75):
        
      #  p = Pipe()
     #   p.LogChange()
     #   p.PctChangeAndVol()
     #   self.X =  p.RemoveFeat()
     #   self.X.loc[:,('20DayRet_Close', 'TSLA')] = np.log(self.X.loc[:,('20DayRet_Close', 'TSLA')]) - np.log(self.X.loc[:,('20DayRet_Close', 'TSLA')].shift(1))
     #   self.X= self.X.dropna()

        p = Pipe()
        p.LogChange()
        p.PctChangeAndVol()
        self.X =  p.RemoveFeat()
        self.X= p.RemoveOtherFeat()
        self.copy= self.X.copy()
        
       # self.X.loc[:,('20DayRet_Close', 'TSLA')] = np.log(self.X.loc[:,('20DayRet_Close', 'TSLA')]) - np.log(self.X.loc[:,('20DayRet_Close', 'TSLA')].shift(1))
        self.X = self.X.dropna()

       # self.DickeyFullerStation(self.X)
       # self.KPSSStation(self.X)
        self.InvTransform = pd.DataFrame(index = np.arange(0,self.X.shape[0], 1))

        for c in self.X.drop('MacroData',axis=1).columns:
            f = Frac()
            col = self.X[c].values.reshape(-1,1)
            #print(col)
            d=f.FindOptFracd(col)
            print(d)
            currf = np.array(f.OGfrac(col,d)).reshape(-1,1)
            self.InvTransform[c] =  col - currf
            self.X[c] = currf


        self.X = self.DickeyFullerStation(self.X)
        #self.X = self.X.drop('MarcoData',axis =1 )
        print(self.InvTransform.values.shape)
        #print()

        self.X = self.X.dropna()

       # self.DickeyFullerStation(self.X)

       # self.KPSSStation(self.X)


     #   self.X= self.X.dropna()
        #self.X= 100 * self.X
        
        self.X_c = self.X.copy()
        self.ppath = '/Users/teacher/Desktop/PortfolioML/Models/predss/' 
        self.X_test = self.X.iloc[int(perc* self.X.shape[0]):self.X.shape[0],:]
        self.X=self.X.iloc[0:int(perc* self.X.shape[0]),:]

       # self.DickeyFullerStation(self.X)
        self.tickers = ['TSLA', 'AAPL','MSFT','AMZN','NVDA','GOOGL','SPY']
        self.targetcol = '1DayRet_'+'Close'


        #print('inv trans {}'.format(self.InvTransform[(self.targetcol,'TSLA')][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]].values.reshape(-1,1).shape))

        #print(self.X.loc[:,(self.targetcol)].sum(axis = 1))

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
                X=X.drop(feature,axis=1)
                print("Non-Stationary feature {} with p-value {}".format(feature,pvalue[i]))
                i+=1

        return X 


    
    @staticmethod
    def KPSSStation(X):
        pvalue=[]
        i=0
    
        for feature in (X.columns):
            pvalue.append(kpss(X[feature].values)[1])
            if  pvalue[i] <= 0.05:
                print("Stationary feature {} with p-value {}".format(feature,pvalue[i]))
                i+=1
            else:
                print("Non-Stationary feature {} with p-value {}".format(feature,pvalue[i]))
                i+=1

    @staticmethod
    def forecast_one_step(model,exog):
        fc, conf_int = model.predict(n_periods=1,exog=exog, return_conf_int=True)
        return (
        fc.tolist()[0],
        np.asarray(conf_int).tolist()[0])


    def SARIMAX(self,filex = '.csv'):
        cnt = 0 
        fig, ax = plt.subplots(len(self.tickers), 1, figsize=(12, 12))
        ts = np.arange(0,self.X_test.shape[0],1)
        predictions = pd.DataFrame(index = ts)
        
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

            print(stepwise_dicta)

            model_a = pm.arima.ARIMA([]).set_params(**(stepwise_dicta))
             
             #include exo terms after fitting SARIMAX 

            model_a.fit(self.X.loc[:,(self.targetcol,t)].values.reshape(-1,1),exog = self.X.drop(t,axis= 1, level =1).values )
        
            print(model_a.summary())


            model_a.plot_diagnostics()

            forecastsrobust=[]
            confidence_intervalsrobust=[]

          #  print('inv trans {}'.format(self.InvTransform.loc[self.X.shape[0]:self.X.shape[0] +self.X_test.shape[0],(self.targetcol,t)].values.reshape(-1,1).shape))
            y_test = self.X_test.loc[:,(self.targetcol,t)].values.reshape(-1,1) + self.InvTransform[(self.targetcol,t)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]].values.reshape(-1,1)
            print('shape of y_test {}'.format(y_test.shape))
           
            
            for row, new_ob in enumerate(y_test):
    
                fc,conf = self.forecast_one_step(model_a,self.X_test.drop(t,axis= 1, level =1).iloc[row,:].values)
                
                forecastsrobust.append(fc)
                confidence_intervalsrobust.append(conf)
    
                model_a.update(new_ob)

            forecastsrobust=np.array(forecastsrobust).reshape(-1,1) + self.InvTransform[(self.targetcol,t)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]].values.reshape(-1,1)
           # print(type(confidence_intervalsrobust))
            CI=np.array(confidence_intervalsrobust)
            #print(CI[:,0].shape)
           # print(self.InvTransform[(self.targetcol,t)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]].values.reshape(-1,1).shape)
           # print(CI[:,0].reshape(-1,1).shape)
           # print(self.InvTransform[(self.targetcol,t)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]].values.reshape(-1,1).shape)

            lower_ci = CI[:,0].reshape(-1,1) + self.InvTransform[(self.targetcol,t)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]].values.reshape(-1,1)
            upper_ci = CI[:,1].reshape(-1,1) + self.InvTransform[(self.targetcol,t)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]].values.reshape(-1,1)

            lower_ci.shape = (self.X_test.shape[0],)
            upper_ci.shape = (self.X_test.shape[0],)


           # print(lower_ci.shape)
          #  print(upper_ci.shape)
            y_testa = self.copy[(self.targetcol,t)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]]
#

            predictions[t+'_forecast_'] = forecastsrobust 
            
            ax[cnt].fill_between(ts, lower_ci,upper_ci, color ='orange',label="Confidence Intervals")
            ax[cnt].scatter(ts,forecastsrobust,marker ='o',c = 'r',label='statsmodels: SARIMAX ' +'With order: ' + str(stepwise_dicta['order']))
            ax[cnt].scatter(ts,y_test,c='b',label='Actual frac ')
            ax[cnt].scatter(ts,y_testa,c='g',label='Actual ')

            ax[cnt].set_title('Forecast for ' + t)
            ax[cnt].set_xlabel('Time')
            ax[cnt].set_ylabel(t+' Returns')

            ax[cnt].legend(loc = 'upper left')

            residuals = model_a.resid()


            # other stats appear within model summary

            self.ShapiroNormal(residuals)

            self.BreuschHetero(residuals,self.X_c.drop(t,axis =1 , level =1).values)
            cnt+= 1 
        
        predictions.to_csv(self.ppath + 'SARIMAX'+'_forecast_'+filex,index = False)
        plt.show()



    def ARDL(self, filex= '.csv'):

        cnt = 0 
        fig, ax = plt.subplots(len(self.tickers),1 , figsize=(12,12))
        tr =  np.arange(0,self.X.shape[0],1)
        self.X.index= tr
        ts = np.arange(0,self.X_test.shape[0],1)

        predictions = pd.DataFrame(index = ts)
        #ids = []
       # for t in self.tickers:
        #ids = np.array(ids)

        for t in self.tickers:
           # lagdict = dict()
           # for c in self.X.drop(t,axis= 1, level =1).columns:
           #     lagdict[c] = 4

            #print(self.X.head())
            #print(self.X.columns)
            ids = [ ]

            self.exogfeat = self.X.drop((self.targetcol,t),axis = 1).columns
            for x in self.exogfeat:
                ids.append(np.where( self.X.columns==x ))
            
            ids= np.array( ids).reshape(-1,1)
            print(ids)
            

            #exog = self.X.drop(t,axis = 1, level = 1)
            opt_order = ardl_select_order(endog = self.X.loc[:,(self.targetcol,t)].values.reshape(-1,1),glob = True, exog = self.X.iloc[:,1:3].values,ic = 'aic',trend ='n',maxorder=3 , maxlag=3)
           # opt_order = ardl_select_order(endog = self.X.loc[:,(self.targetcol,t)].values.reshape(-1,1),maxorder = 3, glob = True, ic = 'aic',trend ='n')
           
            print('pass')

           # print(opt_order.aic)

          #  print('b4')
            model_a = opt_order.model.fit()
            #model_a = ARDL([],suppress_warnings = True).set_params(**opt_order)

           # print('aFTER ') 


            #model_a.fit(self.X.loc[:,(self.targetcol,t)].values.reshape(-1,1),exog = self.X.drop(t,axis= 1, level =1) )
        
            print(model_a.summary())
            print('blrt')


            #model_a.plot_diagnostics()


            forecastsrobust=[]
            confidence_intervalsrobust=[]

          #  print('inv trans {}'.format(self.InvTransform.loc[self.X.shape[0]:self.X.shape[0] +self.X_test.shape[0],(self.targetcol,t)].values.reshape(-1,1).shape))
            y_test = self.X_test.loc[:,(self.targetcol,t)].values.reshape(-1,1) - self.InvTransform[(self.targetcol,t)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]].values.reshape(-1,1)
            print('shape of y_test {}'.format(y_test.shape))


           
            
            for new_ob in y_test:
    
                fc,conf = self.forecast_one_step(model_a)
                
                forecastsrobust.append(fc)
                confidence_intervalsrobust.append(conf)
    
    
                model_a.update(new_ob)

            forecastsrobust=np.array(forecastsrobust).reshape(-1,1) - self.InvTransform[(self.targetcol,t)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]].values.reshape(-1,1)
           # print(type(confidence_intervalsrobust))
            CI=np.array(confidence_intervalsrobust)
            #print(CI[:,0].shape)
           # print(self.InvTransform[(self.targetcol,t)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]].values.reshape(-1,1).shape)
           # print(CI[:,0].reshape(-1,1).shape)
           # print(self.InvTransform[(self.targetcol,t)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]].values.reshape(-1,1).shape)

            lower_ci = CI[:,0].reshape(-1,1) - self.InvTransform[(self.targetcol,t)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]].values.reshape(-1,1)
            upper_ci = CI[:,1].reshape(-1,1) - self.InvTransform[(self.targetcol,t)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]].values.reshape(-1,1)

            lower_ci.shape = (self.X_test.shape[0],)
            upper_ci.shape = (self.X_test.shape[0],)


           # print(lower_ci.shape)
          #  print(upper_ci.shape)
#

            predictions[t+'_forecast_'] = forecastsrobust

            #ax[cnt].plot(ts,forecastsrobust,marker ='o',c = 'r',label='statsmodels: ARDL ' +'With order: ' + str(opt_order))
            #ax[cnt].plot(ts,y_test,c='b',label='Actual ')
            #ax[cnt].fill_between(ts, lower_ci,upper_ci, color ='orange',label="Confidence Intervals")

            #ax[cnt].set_title('Forecast for ' + t)
            #ax[cnt].set_xlabel('Time')
           # ax[cnt].set_ylabel(t+' Returns')

            #ax[cnt].legend(loc = 'upper left')

            residuals = model_a.resid()


            # other stats appear within model summary

            self.ShapiroNormal(residuals)

            self.BreuschHetero(residuals,self.X_c.drop(t,axis =1 , level =1).values)
            cnt+= 1 
        
        predictions.to_csv(self.ppath + 'ARDL'+'_forecast_'+filex,index = False)
       # plt.show()

    def ARXGARCH(self,filex = '.csv'):
        cnt = 0 
        fig, ax = plt.subplots(len(self.tickers), 1, figsize=(12, 12))
        ts = np.arange(0,self.X_test.shape[0],1)
        predictions = pd.DataFrame(index = ts)
        
        for t in self.tickers:

           # pm.tsdisplay(self.X.loc[:,(self.targetcol,t)].values.reshape(-1,1), lag_max=90, title="Sunspots", show=True)
             
            
          #  forecastsrobust=[]
           # confidence_intervalsrobust=[]
          #  vars = []

          #  print('inv trans {}'.format(self.InvTransform.loc[self.X.shape[0]:self.X.shape[0] +self.X_test.shape[0],(self.targetcol,t)].values.reshape(-1,1).shape))
            y_test = self.X_test.loc[:,(self.targetcol,t)].values.reshape(-1,1) - self.InvTransform[(self.targetcol,t)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]].values.reshape(-1,1)
            print('shape of y_test {}'.format(y_test.shape))
            self.X_newexog = pd.DataFrame(index = np.arange(0,self.X.shape[0],1))
            self.X_newtest = pd.DataFrame(index = np.arange(0,self.X_test.shape[0],1))
            self.X_cn = pd.DataFrame(index = np.arange(0,self.X_c.shape[0],1))


            dicttest= dict()
            for c in self.X_test.drop((self.targetcol,t),axis =1).columns:
                self.X_newexog[c[0]+c[1]] = self.X[c].values
                self.X_newtest[c[0]+c[1]] = self.X_test[c].values.reshape(-1,1)
                dicttest[c[0]+c[1]] = self.X_test[c].values
                self.X_cn[c[0]+c[1]]  = self.X_c[c].values.reshape(-1,1)
                  

    
            ar = ARX(self.X.loc[:,(self.targetcol,t)].values.reshape(-1,1), x =self.X_newexog, lags = [1,2,3,4])
            ar.volatility = GARCH(p=1, q=1,o = 0)
            ar.distribution = StudentsT()
            model_a = ar.fit(update_freq= 1, disp='off')


            result =  model_a.forecast(horizon=y_test.shape[0],x =dicttest ,reindex = False) 
            print( model_a.summary())
            forecastrobust = result.mean.values.reshape(-1,1) - self.InvTransform[(self.targetcol,t)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]].values.reshape(-1,1)
            cond_var = result.residual_variance.values.reshape(-1,1)**2
            var = result.variance.values.reshape(-1,1)

            print('shape of for {}'.format(forecastrobust.shape))
            print(' shape of con var {}'.format(cond_var.shape))
            print(' shape of var {}'.format(var.shape))

          #  upper_ci = forecastrobust - var 
           # lower_ci = forecastrobust + var

    
            print(forecastrobust)
#

            predictions[t+'_forecast_'] = forecastrobust 
            ax[cnt].plot(ts,forecastrobust ,marker ='o',c = 'r',label='Simulations: ARDL-GARCH ')
            ax[cnt].plot(ts,y_test,c='b',label='Actual ')
        #    ax[cnt].plot(ts,cond_var,c ='g',label= ' Var of residual')
            
           # ax[cnt].fill_between(ts, lower_ci,upper_ci, color ='orange',label="Confidence Intervals")

            ax[cnt].set_title('Forecast for ' + t)
            ax[cnt].set_xlabel('Time')
            ax[cnt].set_ylabel(t+' Returns')

            ax[cnt].legend(loc = 'upper left')

            residuals = model_a.resid
            q = ar.distribution.ppf([0.01, 0.05], model_a.params[-1:])

            value_at_risk = pd.DataFrame(-forecastrobust- np.sqrt(cond_var) * q[None, :],columns=["1%","5%"],index = ts)


            print('VAR over t {}'.format(value_at_risk))
#

            # other stats appear within model summary

            self.ShapiroNormal(residuals)

            #self.BreuschHetero(residuals,self.X_test.drop((self.targetcol,t),axis =1).values)
            cnt+= 1 
        
        predictions.to_csv(self.ppath + 'ARXGARCH'+'_forecast_'+filex,index = False)
        plt.show()


        


tsm = TimeSeriesModels()
tsm.SARIMAX()
#tsm.ARXGARCH()
