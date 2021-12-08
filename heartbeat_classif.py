import os
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
import numpy as np

# path to the dataset directory
mit_bih_dir = '/home/simon/deep learning with python/data/mit-bih-arrhythmia-database-1.0.0'

# patient list, 100 to 234
record_names = np.arange(100,235) 

dataframe = pd.DataFrame()

for record_name in record_names:
    fname = os.path.join(mit_bih_dir, str(record_name))
    
    try: # some numbers don't exist in the records
         annotation = wfdb.rdann(fname, 'atr')
    except:
        continue
    
    labels = annotation.symbol

    label, count = np.unique(labels, return_counts=True)
    k_patient = pd.DataFrame({'label':label,'count':count,'patient':record_name})
    dataframe = pd.concat([dataframe,k_patient],axis=0)

print(dataframe)

