from YFpipeline import Pipe
import statsmodels.api as sm
import statsmodels.stats.api as sms
from scipy import stats
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

class CAPM():
    def __init__(self):
        p = Pipe()
        self.X = p.LogCAPM()
     #   p.PctChangeAndVol()
     #   self.X_1 = p.RemoveFeat() 
        
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



    def OLS(self):
       # self.X = self.X_1.iloc[0:int(0.9*self.X_1.shape[0]), :]
        self.y = self.X.loc[:,('1DayRet_'+'Close','SPY')]
        self.X = self.X.drop(('1DayRet_'+'Close','SPY'),axis =1)
    #    print(self.X.shape[1])
        m = sm.OLS(self.y,sm.add_constant(self.X))
     #   print(self.X.shape[1])
        f = m.fit()

        self.betas = f.params.values.reshape(-1,1).T[:,0:self.X.shape[1]]
        print(self.betas.shape)

       # print(f.summary())

        # Normality of res

        self.DickeyFullerStation(self.X)
        
        self.ShapiroNormal(f.wresid.values.reshape(-1,1))

        # hetero

        self.BreuschHetero(f.wresid.values.reshape(-1,1),self.X.values)


    def WLS(self,o=False):
        fig, ax = plt.subplots(figsize = ( 60, 70 ))
        if o == True:
            
            self.X = self.X_1.iloc[0:int(0.9*self.X_1.shape[0]), :]
            self.y = self.X.loc[:,('1DayRet_'+'Close','SPY')]
            self.X = self.X.drop(('1DayRet_'+'Close','SPY'),axis =1)

        m = sm.OLS(self.y,self.X)

        fit = m.fit()
        residuals = fit.wresid.values.reshape(-1,1)
        print(residuals)
        nsample = len(residuals)

        # this method burrowed from the documentation
        w = np.ones(nsample)
        resid1 = fit.wresid[w == 1.0]
        var1 = resid1.var(ddof=int(fit.df_model) + 1)
        resid2 = fit.wresid[w != 1.0]
        var2 = resid2.var(ddof=int(fit.df_model) + 1)

        w_est = w.copy()
        
        w_est[w != 1.0] = np.sqrt(var2) / np.sqrt(var1)
        res_fwls = sm.WLS(self.y, self.X, 1.0 / ((w_est ** 2))).fit()


        self.betas = res_fwls.params.values.reshape(-1,1).T
        
        print(res_fwls.summary())



        self.ShapiroNormal(residuals)

        # hetero

        self.BreuschHetero(residuals,self.X.values)


        ax.hist(residuals, bins = 50 ,density=True,color='r',label = 'Hisogram' )
        ax.set_title('GLS residuals')
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Freq')
        ax.legend()



    def GLS(self,o=False):
        fig, ax = plt.subplots(figsize = ( 60, 70 ))
        if o == True:
            self.y = self.X.loc[:,('1DayRet_'+'Close','SPY')]
            self.X = self.X.drop(('1DayRet_'+'Close','SPY'),axis =1)

        m = sm.OLS(self.y,sm.add_constant(self.X))

        fit = m.fit()
        residuals = fit.resid
        #print(residuals)
       # nsample = len(residuals)
   

##GLS Component    
        res_fit = sm.OLS(np.asarray(residuals)[1:], sm.add_constant(np.asarray(residuals)[:-1])).fit()
        rho = res_fit.params[1]


        order = toeplitz(np.arange(len(residuals.values.reshape(-1,1))))
        sigma = rho**order

        gls_model = sm.GLS(self.y, sm.add_constant(self.X), sigma=sigma)
        ls_results = gls_model.fit()


        self.betas = ls_results.params.values.reshape(-1,1)[0:self.X.shape[1]].T
        
        print(ls_results.summary())

        self.ShapiroNormal(residuals.values.reshape(-1,1))

        # hetero

        self.BreuschHetero(residuals.values.reshape(-1,1),self.X.values)


        ax.hist(residuals.values.reshape(-1,1), bins = 50 ,density=True,color='r',label = 'Histogram' )
        ax.set_title('GLS residuals')
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Freq')
        ax.legend()


        #plt.show()


    def FactorPortfolio(self):
        self.betas = self.betas /np.sum(self.betas)

        print(self.betas)
        total_return2=[]
        
        for c in range(0,self.X.shape[0]):
            total_return2.append(np.dot(self.betas,self.X.iloc[c,:]))
        
        total_return2=np.array(total_return2)
#print(total_return2.sum())
#print(total_return2.std())
        betasharperatio=(np.sum(total_return2))/(np.std(total_return2)* self.X.shape[0])
        
        spyfundsharperatio=np.sum(self.y)/(np.std(self.y) * self.X.shape[0])
        
        print('For a portfolio composed of {} stocks in SPY with weighting given by Beta {}'.format(6,betasharperatio))
        print('SPY Sharpe Ratio {}'.format(spyfundsharperatio))
        
        # ofc increasing tickers and exposure to economy will increase SHARPE and results confirmed as SPY currently has SHARPE ~ 0.6




c = CAPM()
#c.OLS()
#c.WLS()
c.GLS(o= True)
c.FactorPortfolio() 


        

