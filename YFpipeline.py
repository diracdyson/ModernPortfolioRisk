import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.stats import boxcox
import seaborn as sns
from fredapi import Fred 
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from ta import add_all_ta_features

#import QuantConnect
# data + feature engineering/ selection pipeline 

class Pipe():
    def __init__(self,interval = '1d',periodi= 10 ):
        # consider the top 10 holdings of the SPY
        # include the SPY for portfolio diversity tech growth
        self.period =periodi
        period = str(self.period)+'y'
        self.targetcol = '1DayRet_'+'Close'
        self.targetcollist =['1DayRet_'+'Close','20DayRet_'+'Close']
        
        self.tickers = ['TSLA', 'AAPL','MSFT','AMZN','NVDA','GOOGL','SPY']
        self.feat = ['High','Low','Close','Volume','Open']
        self.X = yf.download(tickers =self.tickers,period = period, interval = interval)[self.feat]

       # self.X.index = pd.DatetimeIndex(self.X.index).to_period('D')
        print(self.X.shape)
        #self.X= self.X.dropna()

        codes={'Corporate': 'BAA10Y','10Y':'DGS10','1Y': 'DGS1'}
       # assets={'Gold': '^XAU','US 10y bonds':'^TNX','US 5y bonds': '^FVX','S&P500':'^GSPC','Commodities':'^SPGSCI','US 13wk bonds':'^IRX'}
        #assets=dict(sorted(assets.items(), key=lambda item: item[1]))
       
        key='bb358fb7479df683ef3c8fb6df7c3ebf'
       
        self.macro = self.downloadmacro( codes,key,self.X.index[0]).resample('MS').ffill()

        self.macro.columns=list(codes.keys())

       

        self.macro['empl'] = Fred(key).get_series('PAYEMS', observation_start = self.X.index[0]) # seasonally ajusted non-fram employment in thousands 
        self.macro['gdp'] = Fred(key).get_series('GDPC1', observation_start = self.X.index[0])

       
        
        
        self.macro['gdp'] = self.macro['gdp'].interpolate(method='spline',order =2)

        rev = pd.DataFrame(self.macro.gdp.values.reshape(-1,1)[::-1])
        rev = rev.interpolate(method='spline',order =1)
        rev= rev.values.reshape(-1,1)[::-1]

        self.macro['gdp'] = rev


        
        # Seasonally adjusted real GDP # seasonally ajusted non-fram employment in thousands
        self.macro['unemp'] =Fred(key).get_series('UNRATE', observation_start = self.X.index[0])

        self.macro['Time']=((self.macro['10Y']-self.macro['1Y'])/((self.macro['10Y']).expanding().mean()))*100
        self.macro['Confidence']=((self.macro['10Y']-self.macro['Corporate'])/((self.macro['10Y']).expanding().mean()))*100

        self.macro['1Y'] = self.macro['1Y'].interpolate(method='spline',order =2)

        rev2 = pd.DataFrame(self.macro["1Y"].values.reshape(-1,1)[::-1])
        rev2 = rev2.interpolate(method='spline',order =1)
        rev2= rev2.values.reshape(-1,1)[::-1]

        self.macro['1Y'] = rev2


        self.macro['unemp'] = self.macro['unemp'].interpolate(method='spline',order =2)

        rev3 = pd.DataFrame(self.macro['unemp'].values.reshape(-1,1)[::-1])
        rev3 = rev3.interpolate(method='spline',order =1)
        rev3= rev3.values.reshape(-1,1)[::-1]

        self.macro['unemp'] = rev3

        self.macro['empl'] = self.macro['empl'].interpolate(method='spline',order =2)

        rev4 = pd.DataFrame(self.macro['empl'].values.reshape(-1,1)[::-1])
        rev4 = rev4.interpolate(method='spline',order =1)
        rev4= rev4.values.reshape(-1,1)[::-1]

        self.macro['empl'] = rev4


        self.macro = self.NormalizeMacro()
        print(self.macro.shape)

        
        #for c in self.macro.columns:
        #    self.macro[c]=(np.log(self.macro[c]) - np.log(self.macro[c].shift(1)))

       # self.macro=self.macro.dropna(),


        for o in range(0,len(self.macro.columns)): 
            self.X[('MacroData',self.macro.columns[o])] = self.macro.iloc[:,o]

        macro = self.X['MacroData']
        o1=3 
        o2=3
        o3=4
        o4=1
        o5=3
        o6=1
        o7=2
        o8=1

        for o,col in enumerate(self.macro.columns):
            
            if (col=='gdp'):

                fig, ax = plt.subplots(2,1)
    #
                ax[0].plot(np.arange(0,macro[col].dropna().shape[0],1), macro[col].dropna(),label=col,c='b')
                

                interp1 = macro[col][0:int(macro.shape[0]/2)+1].interpolate(method='spline',order =o1).copy()

                macro[col][0:int(macro.shape[0]/2)+1] = interp1 

                interp2 = macro[col][int(macro.shape[0]/2): macro.shape[0]].interpolate(method='spline',order =o1).copy()
                print('for {} it has null {}'.format(col,macro[col].isnull().sum()))
                macro[col][int(macro.shape[0]/2): macro.shape[0]] = interp2 
                ax[1].plot(macro.index, macro[col],label=col+'spline interp',c='r')
                

                rev = pd.DataFrame(macro[col].values.reshape(-1,1)[::-1])
                rev = rev.interpolate(method='spline',order =o2)
                rev= rev.values.reshape(-1,1)[::-1]

                macro[col] = rev

                ax[0].legend()
                ax[1].legend()


            elif col =='Corporate':
                
                fig, ax = plt.subplots(2,1)
    #
                ax[0].plot(np.arange(0,macro[col].dropna().shape[0],1), macro[col].dropna(),label=col,c='b')
                

                interp1 = macro[col][0:int(macro.shape[0]/2)+1].interpolate(method='spline',order =o3).copy()

                macro[col][0:int(macro.shape[0]/2)+1] = interp1 

                interp2 = macro[col][int(macro.shape[0]/2): macro.shape[0]].interpolate(method='spline',order =o3).copy()
                print('for {} it has null {}'.format(col,macro[col].isnull().sum()))
                macro[col][int(macro.shape[0]/2): macro.shape[0]] = interp2 
                
                

                rev = pd.DataFrame(macro[col].values.reshape(-1,1)[::-1])
                rev = rev.interpolate(method='spline',order =o4)
                rev= rev.values.reshape(-1,1)[::-1]
                macro[col] = rev

                ax[1].plot(macro.index, macro[col],label=col+'spline interp',c='r')


                ax[0].legend()
                ax[1].legend()

            elif col =='10Y':
                
                fig, ax = plt.subplots(2,1)
    #
                ax[0].plot(np.arange(0,macro[col].dropna().shape[0],1), macro[col].dropna(),label=col,c='b')

            
                interp1= macro[col][0: int(macro.shape[0]*(2/3))+1].interpolate(method='spline',order =o5).copy()
                macro[col][0: int(macro.shape[0]*(2/3))+1] = interp1 
                
                interp2 = macro[col][int(macro.shape[0]*(2/3)): int(macro.shape[0])].interpolate(method='spline',order =o5).copy()
                macro[col][int(macro.shape[0]*(2/3)): int(macro.shape[0])] = interp2
                print('for {} it has null {}'.format(col,macro[col].isnull().sum()))
                ax[1].plot(macro.index, macro[col],label=col+'spline interp',c='r')
                
                rev = pd.DataFrame(macro[col].values.reshape(-1,1)[::-1])
                rev = rev.interpolate(method='spline',order =o6)
                rev= rev.values.reshape(-1,1)[::-1]

                macro[col] = rev

                ax[0].legend()
                ax[1].legend()
        
            else:
            
                fig, ax = plt.subplots(2,1)

                ax[0].plot(np.arange(0,macro[col].dropna().shape[0],1), macro[col].dropna(),label=col,c='b')
                    
                macro[col] = macro[col].interpolate(method='spline',order =o7)

                ax[1].plot(macro.index, macro[col],label=col+'spline interp',c='r')

                print('for {} it has null {}'.format(col,macro[col].isnull().sum()))
                    
                rev = pd.DataFrame(macro[col].values.reshape(-1,1)[::-1])
                rev = rev.interpolate(method='spline',order =o8)
                rev= rev.values.reshape(-1,1)[::-1]

                macro[col] = rev

                ax[0].legend()
                ax[1].legend()


        for o in range(0,len(self.macro.columns)): 
            self.X[('MacroData',self.macro.columns[o])] = macro.iloc[:,o]
        
        print(self.X[('MacroData',self.macro.columns[0])])

        print(self.X.isnull().sum())


        copX = self.X.copy()
        X = copX
        ogcols = self.X.columns
        for t in self.tickers:
            X= copX
            add = add_all_ta_features(X, open=("Open",t), high=("High",t),low=("Low",t), close=("Close",t), volume=("Volume",t), fillna=True)
            add = add.drop(ogcols,axis =1)

            #add = add.dropna()

            #add = pd.DataFrame(StandardScaler().fit_transform(add.values),columns= add.columns)
            #print(add.columns)

            
            
            for n, a in enumerate(add.columns):
                
                self.X.loc[:,(a[0],t)] = add.iloc[:,n]

            for cols in self.X.columns:
                #print(cols[1])
                if (cols[1] != t) or (cols[1] not in self.macro.columns):
                    pass
                else:
                    self.X = self.X.drop(cols,axis = 1)

            #print(add.columns)
        print(self.X.columns)

        print(self.X.head())
        print(self.X.shape)
        print(self.X.isnull().sum())



        #self.X = self.X.dropna()
        

            
           # for a in add.columns:
            
                #self.X.loc[:,(a,t)] = add[a]

        #self.X.drop(add.columns,axis = 1)
        #print(self.X.drop(ogcols,axis =1 ).columns)
    

        
        #print(self.X.head()) 
        
    
       # plt.show()
        pass

    @staticmethod
    def downloadmacro(codes,key,date2)-> pd.DataFrame():
        key='bb358fb7479df683ef3c8fb6df7c3ebf'
        fred = Fred(api_key=key)
        data={}
        for i in codes.values():
            data[i]=fred.get_series(i,observation_start=date2)
        macro=pd.DataFrame.from_dict(data)
    
        return macro

    def NormalizeMacro(self) -> pd.DataFrame():
        self.macro=pd.DataFrame(StandardScaler().fit_transform(self.macro.values),columns = self.macro.columns,index = self.macro.index)

        self.macro = self.macro.dropna()
        for c in self.macro:
            if adfuller(self.macro[c].values.reshape(-1,1))[1] <= 0.05:
                #self.macro[c] = np.log(self.macro[c]) - np.log(self.macro[c].shift(1))
                self.macro[c] = self.macro[c].diff()
        
        self.macro = self.macro.dropna()

        return self.macro

    def GetX(self) -> pd.DataFrame():
        return self.X

    # the CAPM fits the SPY(
    def LogCAPM(self) -> pd.DataFrame():
            
        for t in self.tickers:
                #log returns
                #self.X.loc[:,('Daily Return' + self.feat[u],t)] = self.X.loc[:,(self.feat[u],t)].pct_change()
            self.X.loc[:,('1DayRet_'+'Close',t)] = np.log(self.X.loc[:,('Close',t)]) - np.log(self.X.loc[:,('Close',t)].shift(1))


        self.X = self.X.dropna()
        self.X = self.X.drop(self.feat,axis =1 )

        return self.X

    def LogChange(self)-> pd.DataFrame():
        
        for u in range(0,len(self.feat)):
            
            for t in self.tickers:
                #log returns
                #self.X.loc[:,('Daily Return' + self.feat[u],t)] = self.X.loc[:,(self.feat[u],t)].pct_change()
                self.X.loc[:,('1DayRet_'+self.feat[u],t)] = (self.X.loc[:,(self.feat[u],t)]) - (self.X.loc[:,(self.feat[u],t)].shift(1))


        self.X = self.X.dropna()

        return self.X
    
    # the data must be positive
    def BoxCox(self)-> pd.DataFrame():

        for feature in self.X.columns:
            self.X[feature], best_lambda = boxcox(self.X[feature])
            print(" for the feature {} the best_lambda is {}".format(feature,best_lambda))

        return self.X

    def NormalizeFeat(self) -> pd.DataFrame():
        col = self.X.columns
        self.X = pd.DataFrame(MinMaxScaler().fit_transform(self.X),columns = col)

       # print(self.X.values)
    #    print(self.X.shape)

        return self.X

    def PctChangeAndVol(self,bins =100) -> pd.DataFrame():
       # print(self.X.shape)
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

            self.X.loc[:,('20DayRet_'+'Close',t)] = (self.X.loc[:,('Close',t)]) - (self.X.loc[:,('Close',t)].shift(20))
                
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
        for t in self.tickers:
            for f in self.feat:
        
                self.X = self.X.drop((f,t), axis=1 )
        

        return self.X
    
    def RemoveOtherFeat(self) -> pd.DataFrame():
        
      

        for ts in self.tickers:
            self.X[('1DayRet_Spread',ts)] = self.X.loc[:,('1DayRet_High',ts)] - self.X.loc[:,('1DayRet_Low',ts)] 
            
            self.X = self.X.drop(('1DayRet_High',ts), axis=1 )
            self.X = self.X.drop(('1DayRet_Low',ts), axis=1 )
            

        

      #  for f in ['1DayRet_',' 2'

        return self.X
    

    @staticmethod
    def PCASVD(X,expvar = 0.90)-> pd.DataFrame():
        pca = PCA(n_components= expvar , random_state=0,svd_solver='auto')
       
        X = pd.DataFrame(pca.fit_transform(X))
        print(" the number of components {}".format(len(pca.explained_variance_ratio_)))
        
        return X
    
    

    

    

    


p = Pipe()
#p.PctChangeAndVol()
#p.RemoveFeat()
#p.NormalizeFeat()
#p.PCASVD()



