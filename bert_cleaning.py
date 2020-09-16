import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# !pip install transformers
# !pip install tensorflow==2.1.0
# !pip install simpletransformers
# !pip install tokenizers==0.8.1.rc1
# !export CUDA_HOME=/usr/local/cuda-10.1
# !git clone https://github.com/NVIDIA/apex
# %cd apex
# !pip install -v --no-cache-dir ./
import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

import pandas as pd

df = pd.read_csv('/home/weihua/bert-nmt/cleaning_data_17.csv')
df.head()

train = pd.DataFrame(df)
test = pd.DataFrame(df)

from tqdm import tqdm
import numpy as np 
import pandas as pd

# train = pd.DataFrame(data)

for i in tqdm(range(train.index.max())):
    if train.loc[i,'label']==1:
        train.drop([i],inplace=True)
print(len(train))

from tqdm import tqdm
import numpy as np 
import pandas as pd

# train = pd.DataFrame(data)

for i in tqdm(range(test.index.max())):
    if test.loc[i,'label']==0:
        test.drop([i],inplace=True)
print(len(test))

train.drop(['Unnamed: 0','label'], axis=1, inplace=True)
train.head()

test.drop(['Unnamed: 0','label'], axis=1, inplace=True)
test.head()

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(df['text'], df.fraudulent, test_size = 0.20, stratify=df.fraudulent, random_state=777)

# X_train = train['text']
# y_train = train.fraudulent
# X_test = test.text
# y_test = test.fraudulent
train_df = pd.DataFrame()
test_df = pd.DataFrame()
train_df[0] = train['text'] 
train_df[1] = train['fraudulent']
test_df[0] = test['text']
test_df[1] = test['fraudulent']
#train_df = pd.DataFrame(df_train)
#test_df = pd.DataFrame(df_test)
# train_df = train.DataFrame({0: X_train, 1: y_train})
# test_df = test.DataFrame({0: X_test, 1: y_test})
#train_df = train
#test_df = test

from simpletransformers.classification import ClassificationModel


model = ClassificationModel('bert', 'bert-base-uncased', num_labels=2, args={'fp16': False,'overwrite_output_dir': True, "train_batch_size": 64, "save_steps": 10000, "save_model_every_epoch":False,
                                                                           'num_train_epochs': 1}, use_cuda=True)

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(test_df)

import numpy as np
preds = [np.argmax(tuple(m)) for m in model_outputs]

from sklearn.metrics import f1_score

print(f1_score(test_df[1], preds, average='micro'))
print(f1_score(test_df[1], preds, average='macro'))

from sklearn.metrics import classification_report
print(classification_report(test_df[1], preds))
