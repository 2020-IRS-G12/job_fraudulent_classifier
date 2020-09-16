import pandas as pd

import numpy as np
data = pd.read_csv("/Users/taoxiyan/Downloads/job_classification/fraudulent_data.csv")
ls = ['company_profile','description','requirements','benefits']
for name in ls:
    qianzui = '/Users/taoxiyan/Downloads/job_classification/'
    path = qianzui+name+'.txt'
    company = open(path,'w',encoding='utf-8')
    for i in data[name]:
        company.write(str(i).strip())
        company.write('\n')