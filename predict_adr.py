import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer, PolynomialFeatures
# from time import sleep
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
# from joblib import dump, load
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

def predict_adr(df):

    drop = ["year", "month", "day", "is_canceled","reserved_room_type", "stays_in_weekend_nights", "stays_in_week_nights","date"]
    toEncode = ["country", "market_segment", "distribution_channel", "assigned_room_type", "agent", "company",
                "customer_type","hotel", "meal", "deposit_type"]
    df.read_drop_encode(drop, toEncode, [])
    df.drop_encode()

    adrLabel, adrFeature, testf = df.getTrainTest_adr_pd()
    print(adrLabel.shape)
    print(adrFeature.shape)
    print(testf.shape)
    # print(adrFeature)

    # pca = PCA(n_components=85).fit(adrFeature)
    # adrFeature = pca.transform(adrFeature)

    # model = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(adrFeature, adrLabel)
    # 1.539474

    # model = LinearRegression().fit(adrFeature, adrLabel)
    # 0.618421

    # model = SVR().fit(adrFeature, adrLabel)
    # 0.776316

    model = KNeighborsRegressor(n_neighbors=10).fit(adrFeature, adrLabel)
    # 0.486842

    # model = MLPRegressor(max_iter=500).fit(adrFeature, adrLabel)
    # 0.486842

    # print(model.score(adrFeature, adrLabel))
    # _predict = model.predict(adrFeature)

    # print((_predict-adrLabel).abs().mean())
    # dump(model, 'model.joblib')

    # testf = test.drop(columns=["adr", "days", "date"], axis=1)
    #test_pca = pca.transform(testf)

    return model.predict(testf)
    # testf["profit"] = testf["adr"] * testf["days"]
    # print(testf["profit"])
    # test = test.sort_values("date").groupby(["date"]).sum()
    # print(test)
    #
    # label = pd.read_csv("test_nolabel.csv")
    # label["label"] = (test["profit"].values // 10000).astype(int)
    # label.loc[label.label > 9, ["label"]] = 9
    # label.loc[label.label < 0, ["label"]] = 0
    # print(label)
    #
    # label.to_csv("label.csv", index=False)

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