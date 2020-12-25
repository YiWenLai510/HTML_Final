import pandas as pd
from sklearn.preprocessing import LabelBinarizer, PolynomialFeatures
# from time import sleep
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

if __name__ == "__main__":
    feature = pd.read_csv("train.csv").fillna(value = 0).rename(columns={"arrival_date_year":"year", "arrival_date_month":"month", "arrival_date_day_of_month":"day"})
    feature = feature[feature["is_canceled"] == 0]
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
    # feature["date"] = pd.to_datetime(feature[["year","month","day"]])
    feature.drop(columns=["year","month","day","ID","is_canceled","reservation_status","reservation_status_date",
                          "reserved_room_type","reservation_status","reservation_status_date"], axis=1, inplace=True)
    # print(len(feature["market_segment"]))
    for f in ["country","market_segment","distribution_channel",
              "assigned_room_type","agent","company","customer_type"]:
        encoder = LabelBinarizer()
        binaryFeature = pd.DataFrame(encoder.fit_transform(feature[f].astype(str)))
        # sleep(10)
        feature = pd.concat([feature.reset_index(drop=True), binaryFeature.reset_index(drop=True)], axis=1).drop([f], axis=1)
    # print(feature)
    
    adrLabel = feature["adr"]
    print(adrLabel)
    adrFeature = feature.drop(columns=["adr"], axis=1)
    print(adrFeature)

    poly_model = make_pipeline(PolynomialFeatures(5), LinearRegression())
    labelfit = poly_model.fit(adrFeature, adrLabel)
    print((labelfit-adrLabel).abs().mean())

