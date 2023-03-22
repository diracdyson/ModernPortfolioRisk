import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.stats import boxcox
import seaborn as sns
# data + feature engineering/ selection pipeline 

class Pipe():
    def __init__(self,interval = '1d'):
        # consider the top 10 holdings of the SPY
        # include the SPY for portfolio diversity tech growth
        self.period =3
        period = str(self.period)+'y'
        
        self.tickers = ['TSLA', 'AAPL','MSFT','AMZN','NVDA','GOOGL','SPY']
        self.feat = ['High','Low','Close','Volume']
        self.X = yf.download(tickers =self.tickers,period = period, interval = interval)[self.feat]
        
     #   print('Null values {}'.format(self.X.isnull().sum()))
        
     #   print(self.X.columns)
     #   print(self.X.columns)
      #  print(self.X.shape)
       # sns.heatmap(self.X.corr(),cmap = 'rainbow')
       # plt.show()
        pass

    # the CAPM fits the SPY(
    def LogCAPM(self):
            
        for t in self.tickers:
                #log returns
                #self.X.loc[:,('Daily Return' + self.feat[u],t)] = self.X.loc[:,(self.feat[u],t)].pct_change()
            self.X.loc[:,('1DayRet_'+'Close',t)] = np.log(self.X.loc[:,('Close',t)]) - np.log(self.X.loc[:,('Close',t)].shift(1))


        self.X = self.X.dropna()
        self.X = self.X.drop(self.feat,axis =1 )

        return self.X

    def LogChange(self):
        for u in range(0,len(self.feat)):
            
            for t in self.tickers:
                #log returns
                #self.X.loc[:,('Daily Return' + self.feat[u],t)] = self.X.loc[:,(self.feat[u],t)].pct_change()
                self.X.loc[:,('1DayRet_'+self.feat[u],t)] = np.log(self.X.loc[:,(self.feat[u],t)]) - np.log(self.X.loc[:,(self.feat[u],t)].shift(1))


        self.X = self.X.dropna()

        return self.X
    
    # the data must be positive
    def BoxCox(self):

        for feature in self.X.columns:
            self.X[feature], best_lambda = boxcox(self.X[feature])
            print(" for the feature {} the best_lambda is {}".format(feature,best_lambda))

        return self.X





    def NormalizeFeat(self) -> pd.DataFrame():
        col = self.X.columns
        self.X = pd.DataFrame(MinMaxScaler().fit_transform(self.X),columns = col)

       # print(self.X.values)
        print(self.X.shape)

        return self.X

    def PctChangeAndVol(self,bins =100) -> pd.DataFrame():
        print(self.X.shape)
        cnt = 0
        
        self.X  = self.X.dropna()
      #  print('shape b4  {}'.format(self.X.shape))
        fig, ax = plt.subplots(len(self.feat) * len(self.tickers)+1 ,1,figsize = (55,140))
        
        for r in range(0,len(self.feat)):
            for t in self.tickers:
                
                curr= self.X.loc[:,( self.feat[r], t)]
                ax[cnt].hist(curr, bins = bins,density=True,color='r',label = 'Density' )
                ax[cnt].set_title('Ticker: '+ t + 'for feat: ' +self.feat[r])
                ax[cnt].set_xlabel('Price')
                ax[cnt].set_ylabel('Freq')
                ax[cnt].legend()
                cnt +=1

       

        for t in self.tickers:

            #self.X.loc[:,('10DayRet_'+'Close',t)] = self.X.loc[:,('Close',t)].pct_change(20)

            mu_1 = np.sum(self.X.loc[:,('1DayRet_'+'Close',t)])/self.X.shape[0]
            self.X.loc[:, ('1DayVol_'+'Close', t)] = np.sqrt(252 * self.period ) * np.sqrt(((self.X.loc[:,('1DayRet_'+'Close',t)] - mu_1)**(2))/(self.X.shape[0] - 1 ))

            self.X.loc[:,('20DayRet_'+'Close',t)] = np.log(self.X.loc[:,('Close',t)]) - np.log(self.X.loc[:,('Close',t)].shift(20))
                
            # add 20 day rolling vol to feat data
          #  roller = self.X['Close'][t].rolling(20)
            # log annualized vol 
            mu_20 = np.sum(self.X.loc[:,('20DayRet_'+'Close',t)])/self.X.shape[0]
           
            self.X.loc[:, ('20DayVol_'+'Close', t)] = np.sqrt(252 * self.period ) * np.sqrt(((self.X.loc[:,('20DayRet_'+'Close',t)] - mu_20) **(2))/(self.X.shape[0] - 1) )

           # self.X.loc[:, ('20DayVol_'+'Close', t)] = np.log(self.X.loc[:,('20DayVol_'+'Close',t)]) - np.log(self.X.loc[:,('20DayVol_'+'Close',t)].shift(20))
            
              # Normalize 
            
          #  rolling_year_ret = self.X["20DayRet_"+'Close'][t].rolling(252)

           # self.X.loc[:, ("20DayRet_"+'Close', t)] = (rolling_year_ret.mean().shift(1) - self.X["20DayRet_"+'Close'][t]) / rolling_year_ret.std(ddof=0).shift(1)


          #  rolling_year_vol = self.X['20DayVol_'+'Close'][t].rolling(252)
          #  self.X.loc[:, ('20DayVol_'+'Close', t)] = (rolling_year_vol.mean().shift(1) - self.X['20DayVol_'+'Close'][t]) / rolling_year_ret.std(ddof=0).shift(1)
            
            



        # declare this after
        self.X= self.X.replace(np.inf,np.nan)

        self.X = self.X.dropna()
       # print(self.X.values)
      #  print('shape af  {}'.format(self.X.shape))
      #  print(self.X.columns)

        #plt.show()
       
        
        return self.X
    
    def RemoveFeat(self) -> pd.DataFrame():
        print(self.X.shape)
        self.X = self.X.drop( self.feat, axis=1 )
        print(self.X.shape)

        return self.X
    


    def PCASVD(self,expvar = 0.90)-> pd.DataFrame():
        pca = PCA(n_components= expvar , random_state=0,svd_solver='auto')
        col = self.X
        self.X = pd.DataFrame(pca.fit_transform(self.X))
        #print(" the number of components {}".format(len(pca.explained_variance_ratio_)))
        print(self.X.shape)
        print(self.X.values )
        return self.X
    

    

    


#p = Pipe()
#p.PctChangeAndVol()
#p.RemoveFeat()
#p.NormalizeFeat()
#p.PCASVD()



