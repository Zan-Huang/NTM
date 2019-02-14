import fetch
import numpy as np
import json
import pandas as pd
from datetime import timedelta, date
from tqdm import tqdm
import pickle
import time

csv_file = "companylist.csv"

csv_read = pd.read_csv(csv_file)
#print(csv_read)

stock_symbols = csv_read.iloc[:,0]

stock_symbol_list = stock_symbols.values.tolist()
#print(stock_symbol_list)

data_list = []

print(len(stock_symbol_list))

for i in tqdm(range(len(stock_symbol_list))):
    iter_parser = fetch.parse_function(stock_symbol_list[i], 'J6VF09SPJX6ORROI', date(2003, 2, 7), date(2019, 2, 8))
    download_stage = iter_parser.json_download()
    collect_stage = iter_parser.json_collect()
    if not collect_stage:
        print("Null List detected %s" % stock_symbol_list[i])
        throw("Null List detected %s" % stock_symbol_list[i])

    data_list.append(collect_stage)
    time.sleep(1.75)

print(data_list)

with open('datafile', 'wb') as f:
    pickle.dump(datalist, fp)
