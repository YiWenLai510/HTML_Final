import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer, PolynomialFeatures
# from time import sleep
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from joblib import dump, load
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

def predict_adr(df):

    drop = [ "reserved_room_type", "stays_in_weekend_nights", "stays_in_week_nights","date"]
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
    dump(model, 'sklearn_adr_KNR')
    #model = load('sklearn_adr_KNR')
    # testf = test.drop(columns=["adr", "days", "date"], axis=1)
    #test_pca = pca.transform(testf)

    return model.predict(testf)

