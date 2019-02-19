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

stock_symbols = csv_read.iloc[:,0]

stock_symbol_list = stock_symbols.values.tolist()
#print(stock_symbol_list)

data_list = []

print(len(stock_symbol_list))
numpy_data = []

for i in tqdm(range(len(stock_symbol_list))):
    iter_parser = fetch.parse_function(stock_symbol_list[i], 'J6VF09SPJX6ORROI', date(2018, 2, 2), date(2019, 2, 4))
    download_stage = iter_parser.json_download()
    collect_stage = iter_parser.json_collect()
    for elements in collect_stage:
        if not elements:
            raise ValueError("Null List detected %s" % stock_symbol_list[i])
    if not collect_stage:
        raise ValueError("Null List detected %s" % stock_symbol_list[i])

    collect_stage_np = np.asarray(collect_stage)
    if(i > 0 and collect_stage_np.shape != numpy_data[i-1].shape):
        zeros = np.zeros(5, collect_stage_np.shape[2] - numpy_data[i-1].shape[2])
        np.stack(zeros, collect_stage_np)
    #ata_list.append(collect_stage)
    numpy_data.append(collect_stage_np)
    time.sleep(0)

final_data = np.stack(numpy_data)
#numpy_data = np.asarray(data_list)
#print(numpy_data)
#print(len(data_list[0][0]))
print(final_data.shape)

np.save('data.npy', numpy_data)
