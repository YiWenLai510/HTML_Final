import pandas as pd
from sklearn.preprocessing import LabelBinarizer, PolynomialFeatures
# from time import sleep
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
# from joblib import dump, load

if __name__ == "__main__":
    feature = pd.read_csv("train.csv").fillna(value = 0).rename(columns={"arrival_date_year":"year", "arrival_date_month":"month", "arrival_date_day_of_month":"day"})
    # tfeature = pd.read_csv("train.csv").fillna(value = 0).rename(columns={"arrival_date_year":"year", "arrival_date_month":"month", "arrival_date_day_of_month":"day"})
    feature = feature[feature["is_canceled"] == 0]
    # 58771
    feature["hotel"].replace({"Resort Hotel":2,"City Hotel":1}, inplace=True)
    # feature["month"].replace({'January':1,
    #                            'February':2,
    #                            'March':3,
    #                            'April':4,
    #                            'May':5,
    #                            'June':6,
    #                            'July':7,
    #                            'August':8,
    #                            'September':9,
    #                            'October':10,
    #                            'November':11,
    #                            'December':12}, inplace=True)
    feature["meal"].replace({"Undefined":1,"SC":1,"BB":2,"HB":3,"FB":4}, inplace=True)
    feature["deposit_type"].replace({"No Deposit":1,"Non Refund":2,"Refundable":3}, inplace=True)
    feature["desired_room"] = (feature["reserved_room_type"] == feature["assigned_room_type"])
    feature["days"] = feature["stays_in_weekend_nights"] + feature["stays_in_week_nights"]
    # feature["date"] = pd.to_datetime(feature[["year","month","day"]])
    feature.drop(columns=["year","month","day","ID","is_canceled","reservation_status","reservation_status_date",
                          "reserved_room_type","stays_in_weekend_nights","stays_in_week_nights"], axis=1, inplace=True)
    # print(len(feature["market_segment"]))
    for f in ["country","market_segment","distribution_channel",
              "assigned_room_type","agent","company","customer_type"]:
        encoder = LabelBinarizer()
        binaryFeature = pd.DataFrame(encoder.fit_transform(feature[f].astype(str)))
        print()
        # sleep(10)
        feature = pd.concat([feature.reset_index(drop=True), binaryFeature.reset_index(drop=True)], axis=1).drop([f], axis=1)
    # print(feature)
    
    adrLabel = feature["adr"] * feature["days"]
    # print(adrLabel)
    adrFeature = feature.drop(columns=["adr","days"], axis=1)
    # print(adrFeature)

    pca = PCA(n_components=85).fit(adrFeature) 
    adrFeature = pca.transform(adrFeature)

    model = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(adrFeature, adrLabel)
    _predict = model.predict(adrFeature)

    # 80 2

    # model = LinearRegression().fit(adrFeature, adrLabel)
    # print(model.score(adrFeature, adrLabel))
    # _predict = model.predict(adrFeature)

    print((_predict-adrLabel).abs().mean())
    # dump(model, 'model.joblib')

