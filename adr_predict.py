import pandas as pd
import numpy as np
from liblinearutil import *

class DataReader(object):
    def __init__(self, trainFile, testFile, notinclude, drop_adr, notoneHot, normalization):
        self.trainRow = pd.read_csv(trainFile)
        self.testRow = pd.read_csv(testFile)
        self.train_label_adr = self.trainRow['adr']

        self.not_included = notinclude
        self.drop_adr = drop_adr
        self.not_onehot = notoneHot
        self.normalization = normalization

        self.train_size = len(self.trainRow.index) # nums of train data

        # one hot
        self.wholeData_adr = self.combined()
        self.train_data = self.wholeData_adr[:self.train_size]
        self.test_data = self.wholeData_adr[self.train_size:]


    def combined(self):
        # remove tags in train but not in test
        tmp_train = self.trainRow.drop(columns = self.not_included) # contain ID
        tmp_test = self.testRow.drop(columns = ['ID'])
        # combine train and test
        tmp_combined = pd.concat([tmp_train,tmp_test])
        # remove unwanted columns
        tmp_combined_adr = tmp_combined.drop(columns=self.drop_adr)
        # z-score normalization
        for col_name in self.normalization:
            mu = tmp_combined_adr[col_name].mean()
            std = tmp_combined_adr[col_name].std()
            tmp_combined_adr[col_name] = (tmp_combined_adr[col_name] - mu) / std
        # columns be encode
        all_columns_adr = list(tmp_combined_adr.columns.values)        
        encode_col_adr = []
        for c in all_columns_adr:
            if c not in self.not_onehot:
                encode_col_adr.append(c)
        # one hot encoder
        final_adr = pd.get_dummies(tmp_combined_adr.astype(str),columns=encode_col_adr)
        return final_adr

    def train_svm(self):
        x = np.array(self.train_data).astype(float)
        y = np.array(self.train_label_adr).astype(float)
        # replace nan with 0
        x[np.isnan(x)] = 0
        # cross valid 5
        index_list = np.arange(x.shape[0])
        np.random.shuffle(index_list)
        unit_len = int(x.shape[0]/5)
        acc = []
        models = []
        for i in range(5): # 5 fold cross validation
            part_y = []
            part_x = []
            valid_y = []
            valid_x = []
            for j in range(x.shape[0]):
                if i*unit_len <= j < ((i+1)*unit_len):
                    valid_y.append(y[index_list[j]])
                    valid_x.append(x[index_list[j]].tolist())
                else:
                    part_y.append(y[index_list[j]])
                    part_x.append(x[index_list[j]].tolist())
            print(i)
            model = train(np.array(part_y), np.array(part_x),'-s 11 -c 0.125')
            save_model('adr_%s.model'%(str(i)), model)
            p_label, p_acc, p_val = predict(np.array(valid_y), np.array(valid_x), model)
            models.append(model)
            acc.append(p_acc)
        mse = [p_acc[1] for p_acc in acc]
        print(mse)
        print('min mse ', min(mse))
        best_model_id = mse.index(min(mse))
        print(best_model_id)
        # check accuracy for training set
        m_label, m_acc, m_val = predict(y, x, models[best_model_id])
        return models[best_model_id]

    def predict_svm(self, model):
        x = np.array(self.test_data).astype(float)
        x[np.isnan(x)] = 0
        m_label, m_acc, m_val = predict([], x, model)
        return m_label

    def output_csv(self, label):
        print(self.testRow.shape)
        self.testRow['adr'] = label
        print(self.testRow.shape)
        self.testRow.to_csv('./test_with_adr.csv')
        return self.testRow


if __name__ == "__main__":
    # make train & test same dimension
    notinclude = ['ID', 'is_canceled', 'adr', 'reservation_status', 'reservation_status_date']
    # should be preserved not be encode
    notoneHot_adr = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children',\
                    'babies', 'days_in_waiting_list', 'required_car_parking_spaces', 'previous_bookings_not_canceled',\
                    'previous_cancellations', 'total_of_special_requests', 'booking_changes']
    # redunancy info for predict adr
    drop_adr = ['agent',  'arrival_date_week_number', 'arrival_date_day_of_month']
    # column that want to normalization
    normalization_adr = ['lead_time']

    df = DataReader('train.csv', 'test.csv', notinclude, drop_adr, notoneHot_adr, normalization_adr)

    # train a new model
    model_adr = df.train_svm()
    save_model('best.model', model_adr)

    # # load from exist file
    # model_adr = load_model('best.model')

    label = df.predict_svm(model_adr)
    test_with_adr = df.output_csv(label)