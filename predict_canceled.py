import pandas as pd
import numpy as np
from datetime import timedelta
from libsvm.svmutil import *
from data import DataReader
from sklearn import svm
from sklearn.metrics import accuracy_score
# def processData(train,test,notinclude,drop_canceled,notoneHot):
#     df = DataReader(train,test,notinclude,drop_canceled,notoneHot)
#     train_y,train_x,test_x  = df.getTrainTest_cancel_np()
#     return train_y,train_x,test_x,df
def sklearn(y,x,test_x):
    clf = svm.SVC()
    clf.fit(x, y)
    y_pred = clf.predict(x)
    print(accuracy_score(y, y_pred))
def train_svm(y,x):
    print('train')
    # cross valid 5
    index_list = np.arange(x.shape[0])
    np.random.shuffle(index_list)
    unit_len = int(x.shape[0]/5)
    acc = []
    models = []
    for i in range(5): # 5 fold cross validation
        part_y, part_x, valid_y, valid_x = [],[],[],[]
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
        # svm_save_model('is_canceled_%s.model'%(str(i)), model)
        p_label, p_acc, p_val = svm_predict(np.array(valid_y), np.array(valid_x), model)
        models.append(model)
        acc.append(p_acc)
    print('Max acc ',max(acc))
    # check accuracy for training set
    svm_save_model('is_canceled_.model', models[acc.index(max(acc))])
    m_label,m_acc,m_val = svm_predict(y,x,models[acc.index(max(acc))])
    return  models[acc.index(max(acc))]

def predict_canceled_forTest(model,x,df):
    label, acc, val = svm_predict([],x,model)
    with open('Test_is_canceled_Label.txt','w') as f:
        f.write(' '.join(str(ele) for ele in label))
    df.add_column_to_test(label)

def predict_canceled():
    # make train & test same dimension
    notinclude = ['ID', 'is_canceled', 'adr' , 'reservation_status' , 'reservation_status_date']
    # should be preserved not be encode
    notoneHot = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 'days', 'total_of_special_requests']
    # redunancy info for predict canceled
    drop_canceled = ['agent', 'country', 'arrival_date_week_number',"year", "month", "day", "date"]

    df = DataReader('train.csv', 'test.csv', notinclude, drop_canceled, notoneHot, [])
    df.drop_encode()
    train_y, train_x, test_x = df.getTrainTest_cancel_np()
    #sklearn(train_y,train_x,test_x)
    # model = train_svm(train_y, train_x)
    # predict_canceled_forTest(model,test_x,df_Obj)
    # model = svm_load_model('is_canceled_0.model')
    # predict_canceled_forTest(model, test_x, df_Obj)

    test_cancel_label = []
    p = []
    with open ('Test_is_canceled_Label.txt', 'r') as f:
        line = f.readline()
        # print(line)
        test_cancel_label = line.split(' ')

        for i in test_cancel_label:
            if i == '0.0':
                p.append(0)
            elif i == '1.0':
                p.append(1)
        df.add_column_to_test(np.array(p))

    return np.array(p),df