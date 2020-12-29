import pandas as pd
from sklearn.preprocessing import LabelBinarizer, PolynomialFeatures
# from time import sleep
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
# from joblib import dump, load
import numpy as np
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, BaggingRegressor

if __name__ == "__main__":
    feature = pd.read_csv("train.csv").fillna(value = 0).rename(columns={"arrival_date_year":"year", "arrival_date_month":"month", "arrival_date_day_of_month":"day"})
    testFeature = pd.read_csv("test.csv").fillna(value = 0).rename(columns={"arrival_date_year":"year", "arrival_date_month":"month", "arrival_date_day_of_month":"day"})
    testCanceled = np.loadtxt("testCanceled.txt").transpose()
    # print(testCanceled)
    testFeature["is_canceled"] = testCanceled
    # print(testFeature)
    feature = pd.concat([feature,testFeature], ignore_index=True)
    print(feature)
    feature = feature[feature["is_canceled"] == 0]
    # feature["hotel"].replace({"Resort Hotel":1,"City Hotel":0}, inplace=True)
    feature["month"].replace({'January':1,
                               'February':2,
                               'March':3,
                               'April':4,
                               'May':5,
                               'June':6,
                               'July':7,
                               'August':8,
                               'September':9,
                               'October':10,
                               'November':11,
                               'December':12}, inplace=True)
    # feature["meal"].replace({"Undefined":1,"SC":1,"BB":2,"HB":3,"FB":4}, inplace=True)
    # feature["deposit_type"].replace({"No Deposit":1,"Non Refund":2,"Refundable":3}, inplace=True)
    feature["desired_room"] = (feature["reserved_room_type"] == feature["assigned_room_type"])
    feature["days"] = feature["stays_in_weekend_nights"] + feature["stays_in_week_nights"]
    feature["date"] = pd.to_datetime(feature[["year","month","day"]])
    feature.drop(columns=["year","month","day","ID","is_canceled","reservation_status","reservation_status_date",
                          "reserved_room_type","stays_in_weekend_nights","stays_in_week_nights"], axis=1, inplace=True)
    # print(len(feature["market_segment"]))
    toEncode = ["country","market_segment","distribution_channel","assigned_room_type","agent","company","customer_type"] + ["hotel","meal","deposit_type"]
    for f in toEncode:
        encoder = LabelBinarizer()
        binaryFeature = pd.DataFrame(encoder.fit_transform(feature[f].astype(str)))
        # sleep(10)
        feature = pd.concat([feature.reset_index(drop=True), binaryFeature.reset_index(drop=True)], axis=1).drop([f], axis=1)
    # print(feature)

    train = feature[feature["date"] < pd.to_datetime("2017-04-1")]
    test = feature[feature["date"] > pd.to_datetime("2017-03-31")]
    
    adrLabel = train["adr"]
    # print(adrLabel)
    adrFeature = train.drop(columns=["adr","days","date"], axis=1)
    # print(adrFeature)

    # pca = PCA(n_components=85).fit(adrFeature) 
    # adrFeature = pca.transform(adrFeature)

    # model = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(adrFeature, adrLabel)
    # 1.539474

    # model = LinearRegression().fit(adrFeature, adrLabel)
    # 0.618421

    # model = SVR().fit(adrFeature, adrLabel)
    # 0.776316

    # model = KNeighborsRegressor(n_neighbors=10).fit(adrFeature, adrLabel)
    # 0.486842

    model = MLPRegressor(max_iter=500).fit(adrFeature, adrLabel)
    0.486842

    # model = AdaBoostRegressor(n_estimators=100).fit(adrFeature, adrLabel)
    # 0.736842

    # model = RandomForestRegressor().fit(adrFeature, adrLabel)
    # 0.631579

    # model = BaggingRegressor(base_estimator=MLPRegressor(max_iter=1500)).fit(adrFeature, adrLabel)
    # 0.644737

    # print(model.score(adrFeature, adrLabel))
    # _predict = model.predict(adrFeature)

    # print((_predict-adrLabel).abs().mean())
    # dump(model, 'model.joblib')

    testf = test.drop(columns=["adr","days","date"], axis=1)
    # testf = pca.transform(testf)
    test["adr"] = model.predict(testf)
    test["profit"] = test["adr"] * test["days"]
    print(test["profit"])
    test = test.sort_values("date").groupby(["date"]).sum()
    print(test)

    label = pd.read_csv("test_nolabel.csv")
    label["label"] = (test["profit"].values // 10000).astype(int)
    label.loc[label.label > 9, ["label"]] = 9
    label.loc[label.label < 0, ["label"]] = 0
    print(label)

    label.to_csv("label.csv",index=False)
