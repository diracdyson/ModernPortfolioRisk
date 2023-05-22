from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from YFpipeline import Pipe
import statsmodels.api as sm
import pandas as pd
import pickle5 as pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
import statsmodels.stats.api as sms
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from FracDiff import Frac
from statsmodels.stats.diagnostic import linear_reset as regreset
#from chow_test import chowtest




class Forecast():
    def __init__(self, d= 0.35 , perc = 0.75):
        self.targetcol = '1DayRet_'+'Close'
        self.tickers = ['TSLA', 'AAPL','MSFT','AMZN','NVDA','GOOGL','SPY']
        
        self.ppath = '/Users/teacher/Desktop/PortfolioML/Models/predss/' 
        self.mnames = ['OLS','GLS']
        p = Pipe()
        p.LogChange()
        p.PctChangeAndVol()
        self.X =  p.RemoveFeat()
        self.X = p.RemoveOtherFeat()
        self.copy= self.X.copy()        

        print(self.X.head())

       # self.X.loc[:,('20DayRet_Close', 'TSLA')] = np.log(self.X.loc[:,('20DayRet_Close', 'TSLA')]) - np.log(self.X.loc[:,('20DayRet_Close', 'TSLA')].shift(1))
      #  self.X = self.X.dropna()

        self.InvTransform = pd.DataFrame(index = np.arange(0,self.X.shape[0], 1))
        np.seterr(invalid='ignore')
        for c in self.X.drop('MacroData',axis =1).columns:
            f = Frac()
            col = self.X[c].values.reshape(-1,1)
            #print(col)
            d=f.FindOptFracd(col)
            print(d)
            currf = np.array(f.OGfrac(col,d)).reshape(-1,1)
            self.InvTransform[c] =  col - currf
            self.X[c] = currf
            #print('shape of frac diff {}'.format(np.array(f.OGfrac(self.X[c].values.reshape(-1,1),d)).shape))
            #print('orignal {}'.format(self.X[c].values.reshape(-1,1).shape))
            #print(np.array(f.OGfrac(self.X[c].values.reshape(-1,1),d)).reshape(-1,1).shape)
            

        #self.X= self.X.dropna()

        print(self.X.isnull().sum())
        
        # remove nonstaionary feat after trans
        self.DickeyFullerStation(self.X)

        print('first')

        print(self.X.head())

        
        self.X_c = self.X.copy()
        self.ppath = '/Users/teacher/Desktop/PortfolioML/Models/predss/' 
        self.X_test = self.X.iloc[int(perc* self.X.shape[0]):self.X.shape[0],:]
        self.X=self.X.iloc[0:int(perc* self.X.shape[0]),:]
        print(self.X.shape)
       # print(self.X.values)
    
    @staticmethod
    def SaveModel(model,model_file_name,path=None):
        pickle.dump(model,open(path+model_file_name+'.pkl',"wb"))

    @staticmethod  
    def LoadModel(model_file_name,path=None)->object: 
        model = pickle.load(open(path+model_file_name+'.pkl',"rb"))
        return model
    # return lower_ci, upper_ci for predictionsresult statsmodels docs
    @staticmethod
    def ols_quantile(model, X, q = 0.95):
  # m: OLS model.
  # X: X matrix.
  # q: Quantile.
  #
  # Set alpha based on q.
        a = q * 2
        if q > 0.5:
            a = 2 * (1 - q)
        
        predictions = model.get_prediction(X)
        frame = predictions.summary_frame(alpha=a)
        #if q > 0.5:
            
        return frame.obs_ci_lower, frame.obs_ci_upper
    

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
                X= X.drop(feature,axis= 1)
        
        return X


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

    
    def LR(self,filex ='.csv',alpha = 0.95):
        
        ts = np.arange(0,self.X_test.shape[0],1) 
        predictions = pd.DataFrame()
        predictions.index = ts
        
        fig, ax = plt.subplots(len(self.tickers), 1, figsize=(60,70))
        cnt= 0 
        powers =[2,3,4]
        
        for t in self.tickers:

           # print(self.X.loc[:,(self.targetcol,t)])

            m = sm.OLS(self.X.loc[:,(self.targetcol,t)].shift(1).fillna(self.X.loc[:,(self.targetcol,t)].mean()),sm.add_constant(self.X.drop((self.targetcol,t),axis =1)))

        #    c = chowtest(y=self.X.loc[:,(self.targetcol,t)], X=self.X.drop(t,axis =1,level=1 ),
        #    last_index_in_model_1=int(self.X.shape[0]/2),
         #   first_index_in_model_2=int(self.X.shape[0]/2) +1 ,
         #   significance_level=.05)
 
            fit = m.fit() 

            #print('Testing blrt'.format(regreset(fit,power = 1, test_type='fitted',use_f= True).pvalue))
         #   ramseyreset=np.zeros(len(powers))
           # for cnt, p in enumerate(powers):
               
           #     ramseyreset[cnt]= (regreset(fit,power = p, test_type='fitted',use_f= True).pvalue)


           #     if ramseyreset[cnt] > (1- alpha ):

           #     
          #          print(' For power p {} the RAMSEY LIN REG RESET TEST YIELDS P VALUE {}'.format(p,ramseyreset[cnt]))

           #     else:
                    
            #       pass

            #  m = sm.OLS(self.X.loc[:,(self.targetcol,t)],sm.add_constant(self.X.drop(t,axis=1, level = 1)**2))

           #         fit = m.fit()

           #         break 

           #     else:
                    
           #         pass 
            
            pred = fit.predict(sm.add_constant(self.X_test.drop((self.targetcol,t),axis=1))).values.reshape(-1,1)  + self.InvTransform[(self.targetcol,t)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]].values.reshape(-1,1)
            
        
                # H_o: model has no omitted variables 
                
            y_test = self.X_test.loc[:,(self.targetcol,t)].shift(1).fillna(self.X_test.loc[:,(self.targetcol,t)].mean()).values.reshape(-1,1)  + self.InvTransform[(self.targetcol,t)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]].shift(1).fillna(self.InvTransform[(self.targetcol,t)].mean()).values.reshape(-1,1)
            y_testa = self.copy[(self.targetcol,t)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]]
            print( pred)
            print(y_test)
            print(self.copy[(self.targetcol,t)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]])
          #  print(y_test.shape[0])
          #  print(pred.shape[0])
           
          #  MSE = np.sqrt(np.sum((y_test - pred)**(2))/y_test.shape[0])

           # print('For ticker {} the MSE is {}'.format(t,MSE))

            lower_ci , upper_ci = self.ols_quantile(fit, sm.add_constant(self.X_test.drop((self.targetcol,t),axis = 1)))
           #print(lower_ci.shape)

            lower_ci= lower_ci.values.reshape(-1,1) + self.InvTransform[(self.targetcol,t)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]].values.reshape(-1,1)
            upper_ci =upper_ci.values.reshape(-1,1) +  self.InvTransform[(self.targetcol,t)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]].values.reshape(-1,1)

            lower_ci.shape=(self.X_test.shape[0],)
            upper_ci.shape=(self.X_test.shape[0],)
            

            ax[cnt].plot(ts,pred, c = 'r',label='statsmodels: Linear Reg Forecast')
            ax[cnt].plot(ts,y_test,marker='o',c='b',label='Actual frac ')
           # ax[cnt].plot(ts,y_testa,marker = 'o',c='g',label='Actual ')
           # print(lower_ci.shape)
            ax[cnt].fill_between(ts, lower_ci,upper_ci, color ='orange',label="Confidence Intervals")

            ax[cnt].set_title('Forecast for ' + t)
            ax[cnt].set_xlabel('Time')
            ax[cnt].set_ylabel(t+' Returns')

            ax[cnt].legend(loc = 'upper left')
            cnt+= 1
            

            residuals = fit.resid.values.reshape(-1,1)
            
            # Residuals anaysis
            self.ShapiroNormal(residuals)
            self.BreuschHetero(residuals,self.X.drop(t,axis =1,level =1  ).values)


            predictions[t+'_forecast_'] = pred

            print(fit.summary()) 
        
        predictions.index = ts
        predictions.to_csv(self.ppath+self.mnames[0]+'_forecast_'+filex,index = False)

        
        plt.show()



    def GLS(self,filex = '.csv'):

        ts = np.arange(0,self.X_test.shape[0],1)
        predictions = pd.DataFrame()
        predictions.index = ts
        
        fig, ax = plt.subplots(len(self.tickers), 1, figsize=(60,70))
        cnt= 0 
        for t in self.tickers:

           # print(self.X.loc[:,(self.targetcol,t)])
            

            m = sm.OLS(self.X.loc[:,(self.targetcol,t)],sm.add_constant(self.X.drop(t,axis =1,level=1 )))

            fit = m.fit()
            residuals = fit.resid
            
            res_fit = sm.OLS(np.asarray(residuals)[1:], sm.add_constant(np.asarray(residuals)[:-1])).fit()
            rho = res_fit.params[1]

             # estimation of the covariance structure of correlated residuals     
            order = toeplitz(np.arange(len(residuals.values.reshape(-1,1))))
            sigma = rho**order
           
        
            gls_model = sm.GLS(self.X.loc[:,(self.targetcol,t)],sm.add_constant(self.X.drop(t,axis =1,level=1 )), sigma=sigma)
            gls_fit = gls_model.fit()
            
            pred = gls_fit.predict(sm.add_constant(self.X_test.drop(t,axis = 1,level=1 ))).values.reshape(-1,1) + self.InvTransform[(self.targetcol,t)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]].values.reshape(-1,1)
            
            y_test = self.X_test.loc[:,(self.targetcol,t)].values.reshape(-1,1)  + self.InvTransform[(self.targetcol,t)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]].values.reshape(-1,1)
            
          #  print(y_test.shape[0])
         #   print(pred.shape[0])
           
          #  MSE = np.sqrt(np.sum((y_test - pred)**(2))/y_test.shape[0])

          #  print('For ticker {} the MSE is {}'.format(t,MSE))
            
            ts = np.arange(0,len(y_test),1)
            
            lower_ci , upper_ci = self.ols_quantile(gls_fit,sm.add_constant(self.X_test.drop(t,axis = 1,level=1)))



            lower_ci= lower_ci.values.reshape(-1,1) + self.InvTransform[(self.targetcol,t)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]].values.reshape(-1,1)
            upper_ci =upper_ci.values.reshape(-1,1) + self.InvTransform[(self.targetcol,t)][self.X.shape[0]:self.X.shape[0]+ self.X_test.shape[0]].values.reshape(-1,1)

            lower_ci.shape =(self.X_test.shape[0],)
            upper_ci.shape =(self.X_test.shape[0],)


            ax[cnt].plot(ts,pred,marker ='o',c = 'r',label='statsmodels: Generalized Least Squares Forecast')
            ax[cnt].plot(ts,y_test,c='b',label='Actual ')
            ax[cnt].fill_between(ts, lower_ci,upper_ci, color ='orange',label="Confidence Intervals")

            ax[cnt].set_title('Forecast for ' + t)
            ax[cnt].set_xlabel('Time')
            ax[cnt].set_ylabel(t+' Returns')

            ax[cnt].legend(loc = 'upper left')
            cnt+= 1 
            
            res_f= gls_fit.resid.values.reshape( -1,1)
            self.ShapiroNormal(res_f)
            self.BreuschHetero(residuals,self.X.drop(t,axis =1,level =1  ).values)
            
            print(gls_fit.summary())
            predictions[t+'_forecast_'] = pred
        
        predictions.to_csv(self.ppath+self.mnames[1]+'_forecast_'+filex,index = False)
       
        plt.show()



f = Forecast() 

f.LR()

#f.GLS()
