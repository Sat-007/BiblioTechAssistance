import pandas as pd
import csv
import numpy as np
import random


df=pd.read_csv(r"C:\Users\kssan\OneDrive\Desktop\Final_proj_scrape\data\Scribd.csv")


no_rows=len(df.index)
idx = 0
l=[]
   
for b_id in range(0,no_rows):  
    b_id=b_id+1
    l.append(b_id)
df.insert(loc=idx, column='book_id', value=l)
df.to_csv(r"C:\Users\kssan\OneDrive\Desktop\Final_proj_scrape\data\books_scribd.csv", index=False)