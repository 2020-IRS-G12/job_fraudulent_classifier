import pandas as pd
data1=pd.read_csv('/Users/taoxiyan/Downloads/job_classification/real_data.csv')
data2=pd.read_csv('/Users/taoxiyan/Downloads/job_classification/text2.csv')
res=pd.concat([data1,data2],axis=0)
res.sample(frac=1)
res.to_csv("/Users/taoxiyan/Downloads/job_classification/balanced_data.csv")