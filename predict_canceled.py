import pandas as pd
import numpy as np
from libsvm.svmutil import *
class DataReader(object):    
    def __init__(self,trainFile,testFile,notinclude,drop_canceled,notoneHot):
        self.trainRow = pd.read_csv(trainFile)
        self.train_label = self.trainRow['is_canceled']
        self.testRow = pd.read_csv(testFile)

        self.not_included = notinclude
        self.drop_canceled = drop_canceled
        self.not_onehot = notoneHot

        self.trainStartIndex = 0
        self.testStartIndex = len(self.trainRow.index) # check
        self.trainEndIndex = (self.testStartIndex) - 1

        self.wholeData = self.combined()
        # print(self.wholeData)
    def combined(self):
        # remove tags in train but not in test
        tmp_train = self.trainRow.drop(columns = self.not_included) # contain ID
        tmp_test = self.testRow.drop(columns = ['ID'])
        # combine train and test
        tmp_combined = pd.concat([tmp_train,tmp_test])
        # remove unwanted columns
        tmp_combined = tmp_combined.drop(columns=self.drop_canceled)
        # columns be encode
        encode_col = []
        all_columns = list(tmp_combined.columns.values)
        for c in all_columns:
            if c not in self.not_onehot:
                encode_col.append(c)
        # one hot encoder

        return pd.get_dummies(tmp_combined.astype(str),columns=encode_col)
    def add_column_to_test(self,predict_result):
        if len(self.testRow.index) == len(predict_result):
            self.testRow['is_canceled'] = predict_result
            print(self.testRow)
        else:
            print('Error !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    def getTrainTest_svm(self):
        df_to_numpy = self.wholeData.to_numpy()
        return np.array(self.train_label).astype(int) ,(df_to_numpy[:self.trainEndIndex + 1]).astype(float),(df_to_numpy[self.testStartIndex:]).astype(float)

def processData(train,test,notinclude,drop_canceled,notoneHot):
    df = DataReader(train,test,notinclude,drop_canceled,notoneHot)
    train_y,train_x,test_x  = df.getTrainTest_svm()
    return train_y,train_x,test_x,df

def train_svm(y,x):
    print('train')
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
        # print(len(part_y),len(part_x),len(valid_y),len(valid_x))
        model = svm_train(np.array(part_y), np.array(part_x),'-h 0')
        svm_save_model('is_canceled_%s.model'%(str(i)), model)
        p_label, p_acc, p_val = svm_predict(np.array(valid_y), np.array(valid_x), model)
        models.append(model)
        acc.append(p_acc)
    print('Max acc ',max(acc))

    # check accuracy for training set
    m_label,m_acc,m_val = svm_predict(y,x,models[acc.index(max(acc))])

    return  models[acc.index(max(acc))]
def predict_canceled_forTest(model,x,df):
    label, acc, val = svm_predict([],x,model)
    df.add_column_to_test(label)
    
if __name__ == '__main__':
    # make train & test same dimension
    notinclude = ['ID', 'is_canceled' ,'adr' ,'reservation_status' ,'reservation_status_date']
    # should be preserved not be encode
    notoneHot =  ['lead_time','stays_in_weekend_nights','stays_in_week_nights','adults','children','babies']
    # redunancy info for predict canceled
    drop_canceled = ['agent', 'arrival_date_month', 'country', 'arrival_date_week_number', 'arrival_date_day_of_month']# 'agent',

    train_y, train_x, test_x, df_Obj = processData('train.csv', 'test.csv', notinclude, drop_canceled, notoneHot)
    model = train_svm(train_y, train_x)
    predict_canceled_forTest(model,test_x,df_Obj)