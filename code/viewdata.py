import pandas as pd
import numpy as np
import openslides

tt=pd.read_csv('/home/cad429/code/panda/data/train.csv')
test=pd.read_csv('/home/cad429/code/panda/data/test.csv')
sample=pd.read_csv('/home/cad429/code/panda/data/sample_submission.csv')
print(tt.head())
print("test csv")
print(test.head())
print("sample")
print(sample.head())

print(tt['gleason_score'].unique())
print(tt['isup_grade'].unique())

print(tt[tt['gleason_score'] == '3+4']['isup_grade'].unique())

print(tt[tt['gleason_score'] == '4+3']['isup_grade'].unique())
df=pd.DataFrame(tt)


tt[(tt['isup_grade'] == 2) & (tt['gleason_score'] == '4+3')]
tt['gleason_score'] = tt['gleason_score'].apply(lambda x: "0+0" if x=="negative" else x)
tt.head()
print(tt['gleason_score'].unique())



