import os
import pandas as pd
import numpy as np

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.io as pio
import plotly.graph_objs as go 
from plotly.subplots import make_subplots
import pandas_datareader.data as web
import yfinance as yf

init_notebook_mode(connected=True)

#define the StockDetails Class
class StockDetails(object):

    def __init__(self,*args):
        if len(args)==3:
            self.name, self.start_date, self.end_date=args
            from pandas_datareader import data as pdr
            import certifi
            certifi.where()
            self.df = pdr.get_data_yahoo(self.name, start=self.start_date, end=self.end_date)
        elif len(args)==1:
            print(args[0])
            self.df = pd.read_csv(args[0])
            print(self.df)

        #resp = urlrq.urlopen('https://example.com/bar/baz.html', cafile=certifi.where())

        #self.df = web.DataReader(self.name, data_source='yahoo', start=self.start_date, end=self.end_date)

        #self.df = yf.download(self.name,start=self.start_date, end=self.end_date)
        
    def RSI(self,n=14):
        close = self.df['adjClose']
        delta=close.diff()
        delta=delta[1:]
        pricesUp=delta.copy()
        pricesDown=delta.copy()
        pricesUp[pricesUp<0] =0
        pricesDown[pricesDown>0] =0
        rollUp = pricesUp.rolling(n).mean()
        rollDown = pricesDown.abs().rolling(n).mean()
        rs = rollUp/rollDown
        rsi = 100 - (100/(1+rs))

        return rsi 

    def addRSI_DF(self):
        self.df['RSI']= self.RSI().fillna(0)

        return self.df

    #KDJ
    def stochastic(self, fig, k=14, d=3, isShow=False ):
        df = self.df.copy()
        low_min = df['adjLow'].rolling(window=k).min()
        high_max = df['adjHigh'].rolling(window=k).max()
        df['RSV'] = 100*(df['adjClose']-low_min)/(high_max-low_min)
        df['stoch_k'] = df['RSV'].ewm(alpha=1/3).mean()
        df['stoch_d'] = df['stoch_k'].ewm(alpha=1/3).mean()
        df['stoch_j'] = 3*df['stoch_k'] - 2*df['stoch_d']

        if isShow:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.date.tail(num_days),
                                    y=df['stoch_k'].tail(num_days), name='K stochastic'))
            fig.add_trace(go.Scatter(x=df.date.tail(num_days),
                                    y=df['stoch_d'].tail(num_days), name='D stochastic'))
            fig.add_trace(go.Scatter(x=df.date.tail(num_days),
                                    y=df['stoch_j'].tail(num_days), name='J stochastic'))
            fig.show()

        return df

    def AddMA(self):
        self.df['EMA_9'] = self.df['adjClose'].ewm(5).mean().shift()
        self.df['SMA_50'] = self.df['adjClose'].rolling(50).mean().shift()
        self.df['SMA_100'] = self.df['adjClose'].rolling(100).mean().shift()
        self.df['SMA_200'] = self.df['adjClose'].rolling(200).mean().shift()

        return self.df

    def add_candlestick_info(self):
        self.df['close-open'] = (self.df['adjClose'] - self.df['adjOpen'])
        self.df['hi_lo'] = (self.df['adjHigh'] - self.df['adjLow'])
        self.df['up_shadow'] = (self.df['adjHigh'] - self.df[['adjOpen','adjClose']].max(axis=1))
        self.df['lower_shadow'] = (self.df[['adjOpen','adjClose']].min(axis=1) - self.df['adjLow'])
        self.df['is_grow'] = (self.df['close-open'].apply(lambda x: 1 if x>=0 else 0))


    def get_MACD(self, fig, isShow=False):
        self.df['EMA_12'] = pd.Series(self.df['adjClose'].ewm(
            span=12, min_periods=12).mean())
        self.df['EMA_26'] = pd.Series(self.df['adjClose'].ewm(
            span=26, min_periods=26).mean())
        self.MACD = pd.Series(self.df['EMA_12'] - self.df['EMA_26'])
        self.MACD_signal = pd.Series(self.MACD.ewm(span=9, min_periods=9).mean())
        
        if isShow:
            fig = make_subplots(rows=2, cols=1)
            fig.add_trace(go.Scatter(x=self.df.Date, y=self.df.Close,
                                     name='adjClose'), row=1, col=1)
            fig.add_trace(go.Scatter(x=self.df.Date, y=self.df['EMA_12'],
                                     name='EMA 12'), row=1, col=1)
            fig.add_trace(go.Scatter(x=self.df.Date, y=self.df['EMA_26'],
                                     name='EMA 26'), row=1, col=1)
            fig.add_trace(go.Scatter(x=self.df.Date, y=self.MACD,
                                     name='MACD'), row=2, col=1)
            fig.add_trace(go.Scatter(x=self.df.Date, y=self.MACD_signal,
                                     name='Signal line'), row=2, col=1)

            fig.show()

        return self.MACD, self.MACD_signal
    
    def show_close_MACD_KDJ(self,fig,df):    
       fig = make_subplots(rows=3, cols=1)
       fig.add_trace(go.Scatter(x=df.date, y=df.adjClose,
                                name='adjClose'), row=1, col=1)
       fig.add_trace(go.Scatter(x=df.date, y=df['EMA_12'],
                                name='EMA 12'), row=1, col=1)
       fig.add_trace(go.Scatter(x=df.date, y=df['EMA_26'],
                                name='EMA 26'), row=1, col=1)
       fig.add_trace(go.Scatter(x=df.date, y=self.MACD,
                                name='MACD'), row=2, col=1)
       fig.add_trace(go.Scatter(x=df.date, y=self.MACD_signal,
                                name='Signal line'), row=2, col=1)
       fig.add_trace(go.Scatter(x=df.date.tail(num_days),
                                y=df['stoch_k'].tail(num_days), name='K stochastic'), row=3, col=1)
       fig.add_trace(go.Scatter(x=df.date.tail(num_days),
                                y=df['stoch_d'].tail(num_days), name='D stochastic'), row=3, col=1)
       fig.add_trace(go.Scatter(x=df.date.tail(num_days),
                                y=df['stoch_j'].tail(num_days), name='J stochastic'), row=3, col=1)

       fig.show()

    def generate_stat(self):
        self.AddMA()
        self.add_candlestick_info()
        self.addRSI_DF()
        fig = go.Figure()
        self.get_MACD(fig)
        self.df = self.stochastic(fig)



pio.renderers.default = "browser"
layout = go.Layout(paper_bgcolor='rgba(0,0,0,0)',
                   plot_bgcolor='rgba(250,250,250,0.8)')
fig = go.Figure(layout=layout)
templated_fig = pio.to_templated(fig)
pio.templates['my_template'] = templated_fig.layout.template
pio.templates.default = 'my_template'

def plot_scatter(x, y, title):
    fig = go.Figure(go.Scatter(x=x, y=y, name=title))
    fig.update_layout(title_text=title)
    fig.show()


if __name__ == '__main__':
#    STOCK_NAME = 'DIS'
#    start_date = '2017-01-01'
#    end_date = '2020-12-28'
#
#    stock = StockDetails(STOCK_NAME, start_date, end_date)
    stock = StockDetails('AAPL.csv')

    df = stock.df
    df.head()

    df.reset_index(level=0, inplace=True)
    df.describe()

    #OHLC Chart
    # open, high, low, close prices
    fig = go.Figure([go.Ohlc(x=df.date,
                            open=df.adjOpen,
                            high=df.adjHigh,
                            low=df.adjLow,
                            close=df.adjClose)])
    #fig.update(layout_xaxis_rangeslider_visible=False)
    fig.show()

    fig = go.Figure(go.Bar(x=df.date, y=df.adjVolume,
                        name='Volume', marker_color='red'))
    fig.show()

    #MA
    #df = stock.AddMA()
    stock.generate_stat()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.date, y=df.EMA_9, name='EMA 9'))
    fig.add_trace(go.Scatter(x=df.date, y=df.SMA_50, name='SMA 50'))
    fig.add_trace(go.Scatter(x=df.date, y=df.SMA_100, name='SMA 100'))
    fig.add_trace(go.Scatter(x=df.date, y=df.SMA_200, name='SMA 200'))
    fig.add_trace(go.Scatter(x=df.date, y=df.adjClose, name='adjClose',
                            line_color='dimgray', opacity=0.3))
    fig.show()


    # get the RSI
    num_days = df.size
    df = stock.addRSI_DF()
    fig = go.Figure(go.Scatter(x=df.date.tail(num_days), y=df.RSI.tail(num_days)))
    #fig.show()

    stock.get_MACD(fig, False)

    df = stock.stochastic(fig, k=14, d=3, isShow=True)

    stock.show_close_MACD_KDJ(fig,df)



