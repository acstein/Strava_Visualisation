"""
Package imports
"""
import numpy as np
import pandas as pd
import gzip
import shutil

activity_list = pd.read_csv('export_27187487/activities.csv')
run_list = activity_list[activity_list['Activity Type']=='Run'].copy() # List of only 'Run' activities
run_list['Relative Filename'] = run_list['Filename'].apply(lambda x: r'export_27187487/' + x if isinstance(x, str) else x)

fit_gz = run_list.iloc[200]['Relative Filename']

with gzip.open(fit_gz, 'rb') as f_in:
    file = f_in.read()
    # with open(fit_gz[:-2], 'wb') as f_out:
    #     shutil.copyfileobj(f_in, f_out)
   
print(file)
