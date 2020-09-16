import pandas as pd
import numpy as np

data=pd.read_csv('/Users/taoxiyan/Downloads/job_classification/fraudulent_data.csv')
newdf = pd.DataFrame(np.repeat(data.values,17,axis=0))
newdf.columns = data.columns
print(newdf)

newdf.to_csv('/Users/taoxiyan/Downloads/job_classification/text17.csv')