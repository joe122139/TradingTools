import requests
import pandas as pd
import io
import json 
from os.path import exists

def get_data(path,id='MSFT',outfile='.csv'):
    if exists(path+id+outfile):
        print('%s%s exist'%(id,outfile))
        return

    headers = {
        'Content-Type': 'application/json'
    }
    urlData = requests.get("https://api.tiingo.com/tiingo/daily/%s/prices?startDate=1970-01-01&format=csv&token=d6ec472d3ec4168b05cf46989eb50633aeed290c"%(id), headers=headers).content
    rawData = pd.read_csv(io.StringIO(urlData.decode('utf-8')))
    rawData.to_csv(id+outfile, index=False)


get_data(r'./data/','AAPL','.csv')
get_data(r'./data/','MSFT','.csv')





