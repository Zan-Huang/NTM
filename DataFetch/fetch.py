import requests
import json
from datetime import timedelta, date

API_KEY = 'J6VF09SPJX6ORROI'

def date_iterator(start, end):
    for i in range(int((end - start).days)):
        yield start + timedelta(i)


class parse_function(object):
    def __init__(self, stock_symbol, API_KEY, start_day, end_day):
        self.stock_symbol = stock_symbol
        self.API_KEY = API_KEY
        self.start_day = start_day
        self.end_day = end_day

    def json_download(self):
        file = requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s'%(self.stock_symbol, self.API_KEY))

        file_json = file.json()
        #self.parsed_file = file_json.loads(file)
        self.parsed_file = file_json

    def json_collect(self):
        openval = []
        highval = []
        lowval = []
        closeval = []
        volumeval = []

        for days in date_iterator(self.start_day, self.end_day):
            try:
                openval.append(self.parsed_file['Time Series (Daily)'][days.strftime("%Y-%m-%d")]["1. open"])
                highval.append(self.parsed_file['Time Series (Daily)'][days.strftime("%Y-%m-%d")]["2. high"])
                lowval.append(self.parsed_file['Time Series (Daily)'][days.strftime("%Y-%m-%d")]["3. low"])
                closeval.append(self.parsed_file['Time Series (Daily)'][days.strftime("%Y-%m-%d")]["4. close"])
                volumeval.append(self.parsed_file['Time Series (Daily)'][days.strftime("%Y-%m-%d")]["5. volume"])
            except KeyError:
                continue

        return openval, highval, lowval, closeval, volumeval

def fetch(stock_symbol):
    test_parse = parse_function(stock_symbol, API_KEY, date(2003, 2, 7), date(2003, 2, 8))
    test_parse.json_download()
    print(test_parse.json_collect())
    print("----Data Fetching Tool----")
