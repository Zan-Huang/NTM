import fetch
import numpy as np
import json
import pandas as pd

csv_file = "companylist.csv"

csv_read = pd.read_csv(csv_file)
print(csv_read)

stock_symbols = csv_read.iloc[:,0]

stock_symbol_list = stock_symbols.values.tolist()
print(stock_symbol_list)
