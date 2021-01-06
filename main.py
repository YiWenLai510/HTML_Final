import pandas as pd
import numpy as np
# from libsvm.svmutil import *
import datetime
import predict_canceled
import predict_adr

if __name__ == '__main__':
    # return numpy array
    test_cancel,df_obj = predict_canceled.predict_canceled()
    test_adr = predict_adr.predict_adr(df_obj)
    # # calculate adr
    testRow = pd.read_csv('test.csv')
    blank_label = pd.read_csv("test_nolabel.csv")
    testRow['is_canceled']= test_cancel
    testRow['adr'] = test_adr

    print("Final data =================================")
    print(testRow)

    testRow = testRow[testRow.is_canceled != 1].drop(['is_canceled'], axis=1)

    print("cancel remove ==============================")
    print(testRow)

    testRow['adr_days'] = testRow.adr * (testRow.stays_in_weekend_nights + testRow.stays_in_week_nights)
    month_dict = {'January': 1,
                  'February': 2,
                  'March': 3,
                  'April': 4,
                  'May': 5,
                  'June': 6,
                  'July': 7,
                  'August': 8,
                  'September': 9,
                  'October': 10,
                  'November': 11,
                  'December': 12}
    testRow = testRow.replace({'arrival_date_month': month_dict})
    testRow = testRow.rename(columns={"arrival_date_year": "year", "arrival_date_month": "month", "arrival_date_day_of_month": "day"})
    testRow["date"] = pd.to_datetime(testRow[["year", "month", "day"]])
    # print(testRow)
    df_tmp = testRow[['adr_days', 'date']]

    d = df_tmp.groupby(["date"]).sum().sort_values("date")
    print(df_tmp)

    blank_label["label"] = (d["adr_days"].values // 10000).astype(int)
    blank_label.loc[blank_label.label > 9, ["label"]] = 9
    blank_label.loc[blank_label.label < 0, ["label"]] = 0
    print(blank_label)

    blank_label.to_csv('result.csv',index=False)