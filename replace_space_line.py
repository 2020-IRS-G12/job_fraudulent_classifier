new_data = open('/Users/taoxiyan/Downloads/job_classification/description_new.txt','w',encoding='utf-8')
with open('/Users/taoxiyan/Downloads/job_classification/description.txt') as data:
    for i in data:
        if i.strip()=='':
            new_data.write('nan\n')
        else:
            new_data.write(i)