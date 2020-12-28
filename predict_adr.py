import pandas as pd
import numpy as np
# trainRow = pd.read_csv('train.csv')
# trainRow = trainRow[trainRow.is_canceled != 1].drop(['is_canceled'], axis=1)
# trainRow['adr_days'] = trainRow.adr*(trainRow.stays_in_weekend_nights + trainRow.stays_in_week_nights)
# month_dict = {  'January':1,
#                         'February':2,
#                         'March':3,
#                         'April':4,
#                         'May':5,
#                         'June':6,
#                         'July':7,
#                         'August':8,
#                         'September':9,
#                         'October':10,
#                         'November':11,
#                         'December':12	}
# trainRow =  trainRow.replace({'arrival_date_month':month_dict})
# trainRow = trainRow.rename(columns={"arrival_date_year":"year", "arrival_date_month":"month", "arrival_date_day_of_month":"day"})
# trainRow["date"] = pd.to_datetime(trainRow[["year","month","day"]])
# # print(trainRow)
# df_tmp = trainRow[['adr_days','date']]
# print(df_tmp)
# d = df_tmp.groupby(["date"]).sum().sort_values("date")
# print(d)
# trainLabel = pd.read_csv('train_label.csv')
# print(trainLabel)
# d['label'] = trainLabel['label'].values
# print(d)