import numpy as np 
import pandas as pd
from pandas import DataFrame
data = pd.read_csv("/Users/taoxiyan/Downloads/job_classification/text17.csv")
#ls = ['job_id','title','location','department','salary_range','telecommuting','has_company_logo','has_questions','employment_type','required_experience','required_education','industry','function','fraudulent']
company = open('/Users/taoxiyan/Downloads/job_classification/data16/company_profile.txt').readlines()
description = open('/Users/taoxiyan/Downloads/job_classification/data16/description.txt').readlines()
requirement = open('/Users/taoxiyan/Downloads/job_classification/data16/requirements.txt').readlines()
benifit = open('/Users/taoxiyan/Downloads/job_classification/data16/benefits.txt').readlines()
# print(len(company))
# print(len(data['job_id']))

df = pd.DataFrame()
df['job_id'] = list(data['job_id'])
df['title'] = list(data['title'])
df['location'] = list(data['location'])
df['department'] = list(data['department'])
df['salary_range'] = list(data['salary_range'])
df['company_profile'] = company
df['description'] = description
df['requirements'] = requirement
df['benefits'] = benifit
df['telecommuting'] = list(data['telecommuting'])
df['has_company_logo'] = list(data['has_company_logo'])
df['has_questions'] = list(data['has_questions'])
df['employment_type'] = list(data['employment_type'])
df['required_experience'] = list(data['required_experience'])
df['required_education'] = list(data['required_education'])
df['industry'] = list(data['industry'])
df['function'] = list(data['function'])
df['fraudulent'] = list(data['fraudulent'])
df['label'] = list(data['label'])
# df.set_index('job_id', inplace=True)
# df['job_id'] = list(data['job_id'])*10
# df['title'] = list(data['title'])*10
# df['location'] = list(data['location'])*10
# df['department'] = list(data['department'])*10
# df['salary_range'] = list(data['salary_range'])*10
# df['company_profile'] = company
# df['description'] = description
# df['requirements'] = requirement
# df['benefits'] = benifit
# df['telecommuting'] = list(data['telecommuting'])*10
# df['has_company_logo'] = list(data['has_company_logo'])*10
# df['has_questions'] = list(data['has_questions'])*10
# df['employment_type'] = list(data['employment_type'])*10
# df['required_experience'] = list(data['required_experience'])*10
# df['required_education'] = list(data['required_education'])*10
# df['industry'] = list(data['industry'])*10
# df['function'] = list(data['function'])*10
# df['fraudulent'] = list(data['fraudulent'])*10
# df.set_index('job_id', inplace=True)
df.to_csv('/Users/taoxiyan/Downloads/job_classification/text17_new.csv')