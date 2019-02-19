import fetch
import numpy as np
import json
import pandas as pd
from datetime import timedelta, date
from tqdm import tqdm
import pickle
import time

csv_file = "constituents_csv.csv"

csv_read = pd.read_csv(csv_file)
#print(csv_read)

stock_symbols = csv_read.iloc[0:6,0]

stock_symbol_list = stock_symbols.values.tolist()
#print(stock_symbol_list)

data_list = []

print(len(stock_symbol_list))

for i in tqdm(range(len(stock_symbol_list))):
    iter_parser = fetch.parse_function(stock_symbol_list[i], 'J6VF09SPJX6ORROI', date(2012, 2, 2), date(2016, 2, 3))
    download_stage = iter_parser.json_download()
    collect_stage = iter_parser.json_collect()
    for elements in collect_stage:
        if not elements:
            raise ValueError("Null List detected %s" % stock_symbol_list[i])
    if not collect_stage:
        raise ValueError("Null List detected %s" % stock_symbol_list[i])

    data_list.append(collect_stage)
    time.sleep(0)

numpy_data = np.asarray(data_list)
print(numpy_data)
print(len(data_list[0][0]))
print(numpy_data.shape)

np.save('data.npy', numpy_data)
