import fetch
import numpy as np
import json
import pandas as pd
from datetime import timedelta, date

csv_file = "companylist.csv"

csv_read = pd.read_csv(csv_file)
#print(csv_read)

stock_symbols = csv_read.iloc[:,0]

stock_symbol_list = stock_symbols.values.tolist()
#print(stock_symbol_list)

data_list = []

for i in range(len(stock_symbol_list)):
    iter_parser = fetch.parse_function(stock_symbol_list[i], 'J1X3N3PAL24DFCO4', date(2003, 2, 7), date(2019, 2, 8))
    download_stage = iter_parser.json_download()
    collect_stage = iter_parser.json_collect()
    data_list.append(collect_stage)

print(data_list)
