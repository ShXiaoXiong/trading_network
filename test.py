import csv
import pandas as pd

with open('train_1.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    column1 = [row[2]for row in reader]
closing_prices=column1[1:]
