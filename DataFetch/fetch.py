import requests
import json
from datetime import timedelta, date

API_KEY = 'J1X3N3PAL24DFCO4'

def date_iterator(start, end):
    for i in range(int((end_date - start_date).days)):
        yield start_date + timedelta(i)


class parse_function(object):
    def __init__(self, stock_symbol, API_KEY, start_day, end_day):
        self.stock_symbol = stock_symbol
        self.API_KEY = API_KEY
        self.time_series = time_series
        self.start_day = start_day
        self.end_day = end_day

    def json_download():
        file = requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey==%s'%(self.stock_symbol, self.API_KEY))

        file_json = file.json()
        parsed_file = file_json.loads(file_json)
        open_val = []
        high_val = []
        low_val = []
        close_val = []
        volume_val = []

        for days in daterange(self.start_day, self.end_day):
            openval.append(parsed_file['Time Series (Daily)'][days.strftime("%Y-%m-%d")]["1. open"])

def fetch():
    print("----Data Fetching Tool----")


if __name__ == '__fetch__':
    fetch()
