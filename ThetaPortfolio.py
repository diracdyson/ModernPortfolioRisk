from statsmodels.tsa.forecasting.theta import ThetaModel
from YFpipeline import Pipe
import pandas as pd 
import numpy as np


class TimeSeriesModels():
    def __init__(self):
        p = Pipe()
        p.LogChange()
        p.PctChangeAndVol()
        self.X =  p.RemoveFeat()

        self.X.loc[:,('20DayRet_Close', 'TSLA')] = np.log(self.X.loc[:,('20DayRet_Close', 'TSLA')]) - np.log(self.X.loc[:,('20DayRet_Close', 'TSLA')].shift(1))
        self.X = self.X.dropna()

    def Theta(self,filex = '.csv'):


        tm = ThetaModel()
        