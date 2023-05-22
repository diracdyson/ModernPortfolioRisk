import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from numpy.linalg import norm
from YFpipeline import Pipe
from scipy import stats as scipy_stats
import matplotlib.pyplot as plt
from copulas.univariate import BetaUnivariate
from scipy.linalg import cholesky
from scipy.stats import kurtosis,moment
from sklearn.covariance import LedoitWolf

class Portfolio():
    
    def __init__(self, filex = '.csv',perc =0.75):
        
        self.targetcol = '1DayRet_'+'Close'
        self.tickers = ['TSLA', 'AAPL','MSFT','AMZN','NVDA','GOOGL','SPY']
        self.mnames = ['SARIMAX','XGB']
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
        self.X_test  = self.X.iloc[int(perc * self.X.shape[0]): self.X.shape[0], :]
        self.X  = self.X.iloc[0:int(perc * self.X.shape[0]),:]
        
     #   self.OLSp = pd.read_csv(self.ppath  +self.mnames[0]+'_forecast_'+filex)
      #  self.GLSp = pd.read_csv(self.ppath + self.mnames[1] +'_forecast_'+filex)
        self.SARIMAXp = pd.read_csv(self.ppath + self.mnames[0] +'_forecast_'+filex)
        self.XGBp = pd.read_csv(self.ppath  +self.mnames[1]+'_forecast_'+filex)
        
      #  self.VARp = pd.read_csv(self.ppath + self.mnames[3] +'_forecast_'+filex)  

        #self.mods = #[self.OLSp,self.GLSp,self.SARIMAXp,self.XGBp]
        self.mods=  [self.SARIMAXp,self.XGBp]

        
        #,self.VARp]
        
     #   print( self.OLSp.shape)
      #  print(self.GLSp.shape)

    
    
#--------------------------------------------------------------------------------------------------- 

    def estimated_sharpe_ratio(self,returns):

        return returns.mean() / returns.std(ddof=1)

#--------------------------------------------------------------------------------------------------- 
    def ann_estimated_sharpe_ratio(self,returns=None, periods=261,*,sr=None):
    
        if sr is None:
            sr = self.estimated_sharpe_ratio(returns)
        sr = sr * np.sqrt(periods)
        return sr

#--------------------------------------------------------------------------------------------------- 
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

#--------------------------------------------------------------------------------------------------- 
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

#--------------------------------------------------------------------------------------------------- 
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


#--------------------------------------------------------------------------------------------------- 
    def num_independent_trials(self,trials_returns=None, *, m=None, p=None):
        if m is None:
            m = trials_returns.shape[1]
        
        if p is None:
            corr_matrix = trials_returns.corr()
            p = corr_matrix.values[np.triu_indices_from(corr_matrix.values,1)].mean()
        
        n = p + (1 - p) * m
    
        n = int(n)+1  # round up
    
        return n

#--------------------------------------------------------------------------------------------------- 
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

#--------------------------------------------------------------------------------------------------- 

    def deflated_sharpe_ratio(self,trials_returns=None, returns_selected=None, expected_mean_sr=0.0, *, expected_max_sr=None):
  
        if expected_max_sr is None:
            expected_max_sr = self.expected_maximum_sr(trials_returns, expected_mean_sr)
        
        dsr = self.probabilistic_sharpe_ratio(returns=returns_selected, sr_benchmark=expected_max_sr)

        return dsr


  #out custom maximises 

  #Equality Constraints

  #--------------------------------------------------------------------------------------------------- 
    @staticmethod
    def h(w):
        return sum(w) - 1
#---------------------------------------------------------------------------------------------------  
    @staticmethod
    def h2(w):
        return sum(w)
#---------------------------------------------------------------------------------------------------  
    @staticmethod
    def var(w,varm,gamma =0.05):
        return -((w.T @ (varm) @(w)) - gamma)
    
#--------------------------------------------------------------------------------------------------- 
    @staticmethod
    def ValueAtRisk(ER, var):
        return((ER - (1.96 * var))) 
#--------------------------------------------------------------------------------------------------- 
    @staticmethod
    def RegularizedCost(w,*args):
        ER, var,kur,sk,lam1= args
       # print('h')
        return -((ER.T @ w) - (w.T @ (var) @(w))/2  - (sk.T @ w)/3 - (kur.T @ w)/4 )
       # return -(ER.T@w )
#--------------------------------------------------------------------------------------------------- 
    def ObtainWeights(self,pred,return_actual = False, lam1 =2):

        #print(0)

        opt_bounds = Bounds(0,1)

        if return_actual:
            
            ERa=[]
            for t in self.tickers:
                 ERa.append(np.sum(self.X_test.loc[:,(self.targetcol,t)].values.reshape(-1,1))/self.X_test.shape[0])

            vara  = self.X_test[self.targetcol].cov().values
            ld=LedoitWolf().fit(vara)
            vara= ld.covariance_
            vara = cholesky(vara)
            ERa = np.array(ERa).reshape(-1,1)
            kura = kurtosis(self.X_test[self.targetcol].values,axis=0).reshape(-1,1)
            ska = moment(self.X_test[self.targetcol].values,moment =3, axis=0).reshape(-1,1)

            print('shape of kura {}'.format(kura.shape))
            print('shape of ska {}'.format(ska.shape))

          #  print('era')
         #   print(ERa.shape) 
         #   print(vara.shape)

            

           # mupa = (ERa.T.dot(np.linalg.inv(vara)).dot( ERa))/( np.ones(ERa.shape[0]).dot((np.linalg.inv(vara)).dot(ERa) ))
         #   print(( np.identity(ERa.shape[0]).dot((np.linalg.inv(vara)).dot(ERa) )).shape)
          #  print((ERa.T.dot(np.linalg.inv(vara)).dot( ERa)).shape)
           # print(mupa)

           # mupa = np.sum(mupa)/len(self.tickers)

  #Constraints Dictionary

            cons = [
                    {'type' : 'eq', 'fun' : lambda w: self.h(w)},
                    {'type':'ineq', 'fun': lambda w: w -0.02},
                  #  {'type' : 'ineq', 'fun' : lambda w: self.var(w,vara)}
                    
            
            ]

            initcond = np.random.dirichlet([1]*ERa.shape[0])
            initcond2= np.ones(ERa.shape[0])/np.sum(np.ones(ERa.shape[0]))

            sola = minimize(self.RegularizedCost,
                 x0 = initcond2,
                 args = (ERa, vara,kura,ska,lam1),
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
            ldv=LedoitWolf().fit(var)
            var= ldv.covariance_
            kur = kurtosis(pred,axis=0).reshape(-1,1)
            sk = moment(pred,moment = 3,axis=0).reshape(-1,1)
            print('shape of kur {}'.format(kur.shape))
            print('shape of sk {}'.format(sk.shape))


          #  print(var)
           # print(var.shape)
          #  print(pred.columns)

          #  mup = (ER.T.dot(np.linalg.inv(var)).dot( ER))/( np.ones(ER.shape[0]).dot((np.linalg.inv(var)).dot(ER) ))
         #   print(( np.identity(ERa.shape[0]).dot((np.linalg.inv(vara)).dot(ERa) )).shape)
          #  print((ERa.T.dot(np.linalg.inv(vara)).dot( ERa)).shape)
           # print(mupa)

           # mup = np.sum(mup)/len(self.tickers)


            cons = [
            {'type' : 'eq', 'fun' : lambda w: self.h(w)},
            {'type':'ineq', 'fun': lambda w: w - 0.02},
            #{'type' : 'ineq', 'fun' : lambda w: self.var(w,var)}
                    
            
            ]
            initcond = np.random.dirichlet([1]*ER.shape[0])
            initcond2= np.ones(ER.shape[0])/np.sum(np.ones(ER.shape[0]))

  #Solver
            sol = minimize(self.RegularizedCost,
            x0 =initcond2,
            args = (ER, var,kur,sk,lam1),
            constraints = cons,
            bounds = opt_bounds,
            method = 'SLSQP',
            options = {'disp': False},
            tol=10e-10)
            


            

            #Predicted Results
            w = sol.x
         
            resultdict = {'Weights':w,'Expected_Return':ER, 'Risk':var}


            return resultdict
    
#--------------------------------------------------------------------------------------------------- 
    def PortfolioSummary(self):

        wa, ERa, vara = self.ObtainWeights([],return_actual=True).values()

        ERPa = wa.dot(ERa)

        varpa = np.sqrt(wa.T.dot(vara).dot(wa))
        
        cnt=0
        labelcolor = ['r','b','g','y']
        simulations= 200
        montecarlo =[]
       

       
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
            

            ret= ret.reshape( -1,1)

            cumret = (ret+1).cumsum()
            
            
         #   montecarlo["OG"]= ret
            
            beta=BetaUnivariate()
            beta.fit(ret)
            
            
          #  curr_cols = []
            for s in range(0,simulations):
                montecarlo.append(beta.sample(ret.shape[0]))

         #   print(ret)
           # print(montecarlo)
               # curr_cols.append(s)
#
        #  exp_max_sr_annualized = self.ann_estimated_sharpe_ratio(sr=exp_max_sr)
          #  print('ann jawn {}'.format(exp_max_sr_annualized))


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

            
            #equity = returns.add(1).cumprod().sub(1)

           # for t in range(1,len(ret)):
                #equity[t] = (equity[t-1] * (ret[t]))
               # equitya[t] = (equitya[t-1] * (reta[t]))


          #  equity = np.array(equity)
          #  equity = equity.reshape(-1,1)

            
           # print(ps.shape[0])
           # equity = np.array( equity)
          #  equitya = equitya.reshape(-1,1)

        #    print(equity)
        #    print(equitya)

            #print(equity)
            ax[0].plot(ts,ret,c = labelcolor[cnt], marker ='o',label = 'Forecast from ' + self.mnames[cnt])
            ax[0].plot(ts,reta,c = labelcolor[cnt], marker ='o',label = ' Actual Data weights from' + self.mnames[cnt])
           
            #ax[1].plot(ts,ret,label='OG forecast used for training copula')
            #ax[1].plot(ts,bestmontecarlo)

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
        ax[1].legend()
     #   ax[2].legend()
        #print(wa)

      #  print(self.tickers)

        montecarlo= np.asarray( montecarlo).reshape( ret.shape[0],simulations *len(self.mnames))
      #  assert montecarlo.shape == (ret.shape[0],simulations *len(self.mnames))

        montecarlo= pd.DataFrame(montecarlo,index =self.X_test.index )
    
        ann_best_srs = self.ann_estimated_sharpe_ratio(montecarlo).sort_values(ascending=False)

        # print(ann_best_srs)
#
        self.probabilistic_sharpe_ratio(returns=montecarlo, sr_benchmark=0).sort_values(ascending=False)
            
        curr_probs = self.probabilistic_sharpe_ratio(returns=montecarlo, sr_benchmark=0).sort_values(ascending=False)
        print(' curr probs {}'.format(curr_probs))

        best_psr_pf_name = self.probabilistic_sharpe_ratio(returns=montecarlo, sr_benchmark=0).sort_values(ascending=False).index[0]
        bestmontecarlo = montecarlo[best_psr_pf_name]


        independent_trials = self.num_independent_trials(trials_returns=montecarlo)
        print(' Independent trials {}'.format(independent_trials))
#
        exp_max_sr = self.expected_maximum_sr(trials_returns=montecarlo, independent_trials=independent_trials)
        print('Max sharpe {}'.format(exp_max_sr))


        dsr = self.deflated_sharpe_ratio(returns_selected=bestmontecarlo, expected_max_sr=exp_max_sr)

        #   dsr2 = self.deflated_sharpe_ratio(trials_returns=montecarlo, returns_selected=bestmontecarlo)
        print('deflated jawn {}'.format(dsr))

        sharpePa = np.round((ERPa)/(varpa),3)

        sharpeSPY = np.sum(self.X_test.loc[:,(self.targetcol,'SPY')])/(np.std(self.X_test.loc[:,(self.targetcol,'SPY')]) * np.sqrt(self.X_test.shape[0]))

      #  print('Optimal portfolio has the weights {} for stocks {}'.format(wa.round(3),self.tickers))

        print('The sharpe ratio including actual data {}'.format(sharpePa))

        print('The market sharpe ratio {}'.format(sharpeSPY))

        plt.show()
 

        #  print( var)


p = Portfolio()

p.PortfolioSummary()




#p = Portfolio()


        

