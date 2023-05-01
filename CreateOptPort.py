import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from numpy.linalg import norm
from YFpipeline import Pipe
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt
from copulas.univariate import BetaUnivariate

class Portfolio():
    
    def __init__(self, filex = '.csv'):
        
        self.targetcol = '1DayRet_'+'Close'
        self.tickers = ['TSLA', 'AAPL','MSFT','AMZN','NVDA','GOOGL','SPY']
        self.mnames = ['OLS','GLS','SARIMAX']
        #,'VAR']
        self.ppath = '/Users/teacher/Desktop/PortfolioML/Models/predss/' 

        p = Pipe()
        p.LogChange()
        p.PctChangeAndVol()
        self.X =  p.RemoveFeat()

        self.X = self.X.dropna()
       # self.X = p.PCASVD()
        # enforce stationairty of feat ('20DayRet_Close', 'TSLA')

        #self.X.loc[:,('20DayRet_Close', 'TSLA')] = self.X.loc[:,('20DayRet_Close', 'TSLA')].diff().fillna(0)
        #self.X.loc[:,('20DayRet_Close', 'TSLA')] = np.log(self.X.loc[:,('20DayRet_Close', 'TSLA')]) - np.log(self.X.loc[:,('20DayRet_Close', 'TSLA')].shift(1))

        #self.X = self.X.dropna()
        self.X_test  = self.X.iloc[int(0.7 * self.X.shape[0]): self.X.shape[0], :]
        self.X  = self.X.iloc[0:int(0.7 * self.X.shape[0]),:]
        
        self.OLSp = pd.read_csv(self.ppath  +self.mnames[0]+'_forecast_'+filex)
        self.GLSp = pd.read_csv(self.ppath + self.mnames[1] +'_forecast_'+filex)
        self.SARIMAXp = pd.read_csv(self.ppath + self.mnames[2] +'_forecast_'+filex)
      #  self.VARp = pd.read_csv(self.ppath + self.mnames[3] +'_forecast_'+filex)  

        self.mods = [self.OLSp,self.GLSp,self.SARIMAXp]
        #,self.VARp]
        
     #   print( self.OLSp.shape)
      #  print(self.GLSp.shape)

    @staticmethod
    def RegularizedCost(w,*args):
        ER, var,lam1= args
       # print('h')
        return -((ER.T @ w) - (w.T @ (var) @(w))/2 )
    


    def estimated_sharpe_ratio(self,returns):

        return returns.mean() / returns.std(ddof=1)


    def ann_estimated_sharpe_ratio(self,returns=None, periods=261, *, sr=None):

        if sr is None:
            sr = self.estimated_sharpe_ratio(returns)
        sr = sr * np.sqrt(periods)
        return sr


    def estimated_sharpe_ratio_stdev(self,returns=None, *, n=None, skew=None, kurtosis=None, sr=None):
        if type(returns) != pd.DataFrame:
            _returns = pd.DataFrame(returns)
        else:
            _returns = returns.copy()

        if n is None:
            n = len(_returns)
        if skew is None:
            skew = pd.Series(scipy_stats.skew(_returns), index=_returns.columns)
        if kurtosis is None:
            kurtosis = pd.Series(scipy_stats.kurtosis(_returns, fisher=False), index=_returns.columns)
        if sr is None:
            sr = self.estimated_sharpe_ratio(_returns)

        sr_std = np.sqrt((1 + (0.5 * sr ** 2) - (skew * sr) + (((kurtosis - 3) / 4) * sr ** 2)) / (n - 1))

        if type(returns) == pd.DataFrame:
            sr_std = pd.Series(sr_std, index=returns.columns)
        elif type(sr_std) not in (float, np.float64, pd.DataFrame):
            sr_std = sr_std.values[0]

        return sr_std


    def probabilistic_sharpe_ratio(self,returns=None, sr_benchmark=0.0, *, sr=None, sr_std=None):
   
        if sr is None:
            sr = self.estimated_sharpe_ratio(returns)
        if sr_std is None:
            sr_std = self.estimated_sharpe_ratio_stdev(returns, sr=sr)

        psr = scipy_stats.norm.cdf((sr - sr_benchmark) / sr_std)

        if type(returns) == pd.DataFrame:
            psr = pd.Series(psr, index=returns.columns)
        elif type(psr) not in (float, np.float64):
            psr = psr[0]

        return psr


    def min_track_record_length(self,returns=None, sr_benchmark=0.0, prob=0.95, *, n=None, sr=None, sr_std=None):
   
        if n is None:
            n = len(returns)
        if sr is None:
            sr = self.estimated_sharpe_ratio(returns)
        if sr_std is None:
            sr_std = self.estimated_sharpe_ratio_stdev(returns, sr=sr)

        min_trl = 1 + (sr_std ** 2 * (n - 1)) * (scipy_stats.norm.ppf(prob) / (sr - sr_benchmark)) ** 2

        if type(returns) == pd.DataFrame:
            min_trl = pd.Series(min_trl, index=returns.columns)
        elif type(min_trl) not in (float, np.float64):
            min_trl = min_trl[0]

        return min_trl


    def num_independent_trials(self,trials_returns=None, *, m=None, p=None):
        if m is None:
            m = trials_returns.shape[1]
        
        if p is None:
            corr_matrix = trials_returns.corr()
            p = corr_matrix.values[np.triu_indices_from(corr_matrix.values,1)].mean()
        
        n = p + (1 - p) * m
    
        n = int(n)+1  # round up
    
        return n

    def expected_maximum_sr(self,trials_returns=None, expected_mean_sr=0.0, *, independent_trials=None, trials_sr_std=None):

        emc = 0.5772156649 # Euler-Mascheroni constant
    
        if independent_trials is None:
            independent_trials = self.num_independent_trials(trials_returns)
    
        if trials_sr_std is None:
            srs = self.estimated_sharpe_ratio(trials_returns)
            trials_sr_std = srs.std()
    
        maxZ = (1 - emc) * scipy_stats.norm.ppf(1 - 1./independent_trials) + emc * scipy_stats.norm.ppf(1 - 1./(independent_trials * np.e))
        expected_max_sr = expected_mean_sr + (trials_sr_std * maxZ)
    
        return expected_max_sr


    def deflated_sharpe_ratio(self,trials_returns=None, returns_selected=None, expected_mean_sr=0.0, *, expected_max_sr=None):
  
        if expected_max_sr is None:
            expected_max_sr = self.expected_maximum_sr(trials_returns, expected_mean_sr)
        
        dsr = self.probabilistic_sharpe_ratio(returns=returns_selected, sr_benchmark=expected_max_sr)

        return dsr



  #out custom maximises 

  #Equality Constraints
    @staticmethod
    def h(w):
        return sum(w) - 1
    

    @staticmethod
    def ValueAtRisk(ER, var):
        return((ER - (1.96 * var))) 
 
    def ObtainWeights(self,pred,return_actual = False, lam1 =2):

        #print(0)

        opt_bounds = Bounds(0,1)

  #Constraints Dictionary
        
        
        

        if return_actual:
            
            ERa=[]
            for t in self.tickers:
                 ERa.append(np.sum(self.X_test.loc[:,(self.targetcol,t)].values.reshape(-1,1))/self.X_test.shape[0])

            

            vara  = self.X_test[self.targetcol].cov().values
            ERa = np.array(ERa).reshape(-1,1)
          #  print('era')
         #   print(ERa.shape) 
         #   print(vara.shape)

            

           # mupa = (ERa.T.dot(np.linalg.inv(vara)).dot( ERa))/( np.ones(ERa.shape[0]).dot((np.linalg.inv(vara)).dot(ERa) ))
         #   print(( np.identity(ERa.shape[0]).dot((np.linalg.inv(vara)).dot(ERa) )).shape)
          #  print((ERa.T.dot(np.linalg.inv(vara)).dot( ERa)).shape)
           # print(mupa)

           # mupa = np.sum(mupa)/len(self.tickers)


            cons = [
                    {'type' : 'eq', 'fun' : lambda w: self.h(w)}
                   # {'type' : 'eq', 'fun' : lambda w: np.dot(w.T,ERa) - mupa}
                    
            
            ]

            initcond = np.random.dirichlet([1]*ERa.shape[0])
            initcond2= np.ones(ERa.shape[0])/np.sum(np.ones(ERa.shape[0]))

            sola = minimize(self.RegularizedCost,
                 x0 = initcond,
                 args = (ERa, vara,lam1),
                 constraints = cons,
                 bounds = opt_bounds,
                 method = 'SLSQP',
                 
                 options = {'disp': False},
                 tol=10e-10)
            
            wa= sola.x
            resultdicta = {'Weights':wa,'Expected_Return':ERa, 'Risk':vara}

            return resultdicta
        
        else:

            ER= []

        
            for t in self.tickers:
            
                ER.append(np.sum(pred[t+'_forecast_'].values.reshape(-1,1))/self.X_test.shape[0])
        


            ER = np.array(ER).reshape(-1,1)
            
          #  print('er')
          #  print(ER.shape)
       
            var = pred.cov().values
          #  print(var)
           # print(var.shape)
          #  print(pred.columns)

          #  mup = (ER.T.dot(np.linalg.inv(var)).dot( ER))/( np.ones(ER.shape[0]).dot((np.linalg.inv(var)).dot(ER) ))
         #   print(( np.identity(ERa.shape[0]).dot((np.linalg.inv(vara)).dot(ERa) )).shape)
          #  print((ERa.T.dot(np.linalg.inv(vara)).dot( ERa)).shape)
           # print(mupa)

           # mup = np.sum(mup)/len(self.tickers)


            cons = [
                    {'type' : 'eq', 'fun' : lambda w: self.h(w)}
                   # {'type' : 'eq', 'fun' : lambda w: np.dot(w.T,ER) - mup}
                    
            
            ]
            initcond = np.random.dirichlet([1]*ER.shape[0])
            initcond2= np.ones(ER.shape[0])/np.sum(np.ones(ER.shape[0]))

  #Solver
            sol = minimize(self.RegularizedCost,
                 x0 =initcond,
                 args = (ER, var,lam1),
                 constraints = cons,
                 bounds = opt_bounds,
                 method = 'SLSQP',
                 options = {'disp': False},
                 tol=10e-5)
            

  #Predicted Results
            w = sol.x
         
            resultdict = {'Weights':w,'Expected_Return':ER, 'Risk':var}


            return resultdict
    

    def PortfolioSummary(self):

        wa, ERa, vara = self.ObtainWeights([],return_actual=True).values()

        ERPa = wa.dot(ERa)

        varpa = np.sqrt(wa.T.dot(vara).dot(wa))
        
        cnt=0
        labelcolor = ['r','b','g','y']
       

        fig, ax = plt.subplots(3, 1, figsize =(50,40))
        
        for ps in self.mods:
            
            w, ER, var = self.ObtainWeights(ps).values()

            ret = [ ]
            reta=[]
            equity = np.zeros(ps.shape[0])
            equity[0] = 100

            equitya = np.zeros(ps.shape[0])
            equitya[0] = 100


            for row in range(0,ps.shape[0]):

                ret.append(w.T.dot(ps.loc[row,:].values))
                reta.append(w.T.dot(self.X_test[self.targetcol].iloc[row,:].values))

            
            ret = np.array(ret)
            reta= np.array(reta)

            ret= ret.reshape( -1,1)
            reta= reta.reshape( -1,1)
            
            montecarlo = pd.DataFrame(index =self.X_test.index )
            montecarlo["OG"]= ret
            
            beta=BetaUnivariate()
            beta.fit(ret)
            
            simulations= 5
            
            for s in range(simulations):
                montecarlo[s]= beta.sample(len(ret))

          #  print(ps.iloc[0,:].values.reshape(-1,1).shape)
#
        #    print(self.X_test[self.targetcol].iloc[0,:].shape)
    
           # print(ret)
          #  print(reta)

    
         #   print(' shape of returns {}'.format(ret.shape))

            ts = np.arange(0 ,ps.shape[0],1)

            ERPa = wa.dot(ERa)

            varpa = np.sqrt(wa.T.dot(vara).dot(wa))

            ERP = w.dot(ERa)/self.X.shape[0]

            varp = np.sqrt(w.T.dot(var).dot(w))
#
            sharpeP = np.round(ERP/varp,3)

            valueatrisk = np.percentile(ret,5)

            print('Optimal portfolio has the weights {} for stocks {}'.format(w.round(3),self.tickers))
 
            print('For model '+self.mnames[cnt]+' the sharpe ratio {} has value at risk {}'.format(sharpeP,valueatrisk))

            print(ret)
            print(self.mnames[cnt])
#
            


            for t in range(1,len(ret)):
                equity[t] = (equity[t-1] * np.exp(ret[t]))
                equitya[t] = (equitya[t-1] * np.exp(reta[t]))


            #equity = np.array(equity)
            #equity = equity.reshape(-1,1)

            
           # print(ps.shape[0])
            equity = np.array( equity)

            print(equity)
            print(equitya)


        
            #print(equity)

            ax[0].plot(ts,equity,c = labelcolor[cnt], marker ='o',label = 'Forecast from ' + self.mnames[cnt])
            #ax[0].plot(ts,equitya,c =labelcolor[cnt], label = self.mnames[cnt] +' weight adjusted Actual Equity')
            ax[1].plot(ts,montecarlo)

         #   montecarlo.plot(ax=ax[1])
            


            ax[0].set_xlabel(' Time(Days)')
            
            ax[0].set_ylabel(' Portfolio equity')

            ax[0].set_title(' Simulation of Equity')



            ax[1].set_xlabel(' Time(Days)')
            
            ax[1].set_ylabel(' Returns')

            ax[1].set_title(' Monte Carlo Returns')






            ax[1].set_xlabel(' Time(Days)')
            
            ax[1].set_ylabel(' Returns')

            ax[1].set_title(' Returns')
            
            
            
            

            cnt+=1

        

     #   ax[2].plot(ts,ret,label='forecast')
      #  ax[2].plot(ts,reta,label='actual')

        ax[0].legend()
        ax[2].legend()
        
        plt.show()
        #print(wa)

      #  print(self.tickers)

        sharpePa = np.round((ERPa)/(varpa),3)

        sharpeSPY = np.sum(self.X_test.loc[:,(self.targetcol,'SPY')])/(np.std(self.X_test.loc[:,(self.targetcol,'SPY')]) * self.X_test.shape[0])

      #  print('Optimal portfolio has the weights {} for stocks {}'.format(wa.round(3),self.tickers))

        print('The sharpe ratio including actual data {}'.format(sharpePa))

        print('The market sharpe ratio {}'.format(sharpeSPY))
 

          #  print( var)


p = Portfolio()

p.PortfolioSummary()



        
#p = Portfolio()


        

