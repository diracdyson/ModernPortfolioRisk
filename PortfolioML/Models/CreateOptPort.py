import pandas as pd
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from numpy.linalg import norm
from YFpipeline import Pipe

class Portfolio():
    
    def __init__(self, filex = '.csv'):
        
        self.targetcol = '1DayRet_'+'Close'
        self.tickers = ['TSLA', 'AAPL','MSFT','AMZN','NVDA','GOOGL','SPY']
        self.mnames = ['OLS','GLS']
        self.ppath = '/Users/teacher/Desktop/PortfolioML/Models/predss/' 

        p = Pipe()
        p.LogChange()
        p.PctChangeAndVol()
        self.X =  p.RemoveFeat()
       # self.X = p.PCASVD()
        # enforce stationairty of feat ('20DayRet_Close', 'TSLA')

        self.X.loc[:,('20DayRet_Close', 'TSLA')] = self.X.loc[:,('20DayRet_Close', 'TSLA')].diff().fillna(0)

        self.X  = self.X.iloc[0:int(0.9 * self.X.shape[0]),:]
        self.X_test  = self.X.iloc[0:self.X.shape[0]- int(0.9 * self.X.shape[0]), :]
        self.OLSp = pd.read_csv(self.ppath  +self.mnames[0]+'_forecast_'+filex)
        self.GLSp = pd.read_csv(self.ppath + self.mnames[1] +'_forecast_'+filex)

        self.mods = [self.OLSp,self.GLSp]
        
     #   print( self.OLSp.shape)
      #  print(self.GLSp.shape)

    @staticmethod
    def RegularizedCost(w,*args):
        ER, var, lam1, lam2 = args
        return -(ER.T.dot(w) - lam1*(w.T.dot(var).dot(w)) + lam2*norm(w, ord=1))
  #out custom maximises 

  #Equality Constraints
    @staticmethod
    def h(w):
        return sum(w) - 1

    def ObtainWeights(self,pred,return_actual = False, lam1 =0.5, lam2 =2):

        opt_bounds = Bounds(0, 1)

  #Constraints Dictionary
        cons = ({
            'type' : 'eq',
            'fun' : lambda w: self.h(w)
            })
        
        tickers = ['TSLA', 'AAPL','MSFT','AMZN','NVDA','GOOGL','SPY']


        if return_actual:
            
            ERa=[]
            for t in tickers:
                 ERa.append(np.sum(self.X_test.loc[:,(self.targetcol,t)].values.reshape(-1,1))/self.X_test.shape[0])

            vara  = self.X_test[self.targetcol].cov().values
            ERa = np.array(ERa).reshape(len(tickers),1)


            sola = minimize(self.RegularizedCost,
                 x0 = np.random.rand(ERa.shape[0]),
                 args = (ERa, vara,lam1,lam2),
                 constraints = cons,
                 bounds = opt_bounds,
                 options = {'disp': False},
                 tol=10e-5)
            
            wa= sola.x
            resultdicta = {'Weights':wa,'Expected_Return':ERa, 'Risk':vara}

            return resultdicta
        
        else:

            ER= []

        
            for t in tickers:
            
                ER.append(np.sum(pred[t+'_forecast_'].values.reshape(-1,1))/pred.shape[0])
        


            ER = np.array(ER).reshape(len(tickers),1)
       
            var = pred.cov().values
       

  #Solver
            sol = minimize(self.RegularizedCost,
                 x0 = np.random.rand (ER.shape[0]),
                 args = (ER, var,lam1,lam2),
                 constraints = cons,
                 bounds = opt_bounds,
                 options = {'disp': False},
                 tol=10e-5)


  #Predicted Results
            w = sol.x
        
            resultdict = {'Weights':w,'Expected_Return':ER, 'Risk':var}


            return resultdict
    

    def PortfolioSummary(self):
        
        cnt=0
        for ps in self.mods:
            
            w, ER, var = self.ObtainWeights(ps).values()

            ERP = w.dot(ER)

            varp = np.sqrt(w.T.dot(var).dot(w))

            sharpeP = np.round(ERP/varp,3)

            print('Optimal portfolio has the weights {} for stocks {}'.format(w.round(3),self.tickers))

            print('For model '+self.mnames[cnt]+' the sharpe ratio {}'.format(sharpeP))
            
            cnt+=1

        wa, ERa, vara = self.ObtainWeights([],return_actual=True).values()

        ERPa = wa.dot(ERa)

        varpa = np.sqrt(wa.T.dot(vara).dot(wa))

        sharpePa = np.round(ERPa/varpa,3)

        print('Optimal portfolio has the weights {} for stocks {}'.format(wa.round(3),self.tickers))

        print('The actual sharpe ratio {}'.format(sharpePa))
          

          #  print( var)




p = Portfolio()

p.PortfolioSummary()










    



        
p = Portfolio()
        