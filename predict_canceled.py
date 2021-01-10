import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import timedelta
# from libsvm.svmutil import *
from data import DataReader
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load
from sklearn.model_selection import GridSearchCV

# def processData(train,test,notinclude,drop_canceled,notoneHot):
#     df = DataReader(train,test,notinclude,drop_canceled,notoneHot)
#     train_y,train_x,test_x  = df.getTrainTest_cancel_np()
#     return train_y,train_x,test_x,df
def sklearn(y,x,test_x):
    # clf = svm.SVC()

    clf = AdaBoostClassifier(n_estimators=120, random_state=0)
    # clf = GradientBoostingClassifier(n_estimators=125, random_state=0)
    # 0.8365253302159924  656 columns
    # 0.8547595896472233 1168 columns
    # clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME",n_estimators=200)
    # 0.843004009570528 1168 columns
    clf.fit(x, y)
    dump(clf, "sklearn_ada")
    y_pred = clf.predict(x)
    print(clf.score(x,y))
    print(accuracy_score(y, y_pred))
def train(y,x):
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
        print(i,'==========')
        # print(len(part_y),len(part_x),len(valid_y),len(valid_x))

        # libsvm part
        # model = svm_train(np.array(part_y), np.array(part_x),'-h 0')
        # # svm_save_model('is_canceled_%s.model'%(str(i)), model)
        # p_label, p_acc, p_val = svm_predict(np.array(valid_y), np.array(valid_x), model)
        # models.append(model)
        # acc.append(p_acc)

        clf = AdaBoostClassifier(n_estimators=100, random_state=0)
        clf.fit(np.array(part_x), np.array(part_y))
        dump(clf,"sklearn_ada_%s"%(str(i)))
        print('train score ',clf.score(np.array(part_x),np.array(part_y)))
        y_pred = clf.predict(np.array(valid_x))
        print('valid score ',accuracy_score(np.array(valid_y), y_pred))



    # print('Max acc ',max(acc))
    # check accuracy for training set
    # svm_save_model('is_canceled_.model', models[acc.index(max(acc))])
    # m_label,m_acc,m_val = svm_predict(y,x,models[acc.index(max(acc))])
    return  models[acc.index(max(acc))]

def lightgbm(y, x, test_x):
    index_list = np.arange(x.shape[0])
    np.random.shuffle(index_list)
    unit_len = int(x.shape[0]/5)

    train_data = lgb.Dataset(x[index_list[unit_len:]], label=y[index_list[unit_len:]])
    validation_data = lgb.Dataset(x[index_list[:unit_len]], label=y[index_list[:unit_len]])
    params = {
    'boosting_type': 'gbdt', 
    'objective': 'binary', 
    'n_estimators': 3000,
    'learning_rate': 0.5, 
    'num_leaves': 60,
    'max_depth': 10,
    'reg_lambda': 0,
    }
    params['metric'] = 'auc'
    bst = lgb.train(params, train_data, valid_sets=[validation_data], early_stopping_rounds=1000)

    # # GredSearch CV
    # hyper_space = {'n_estimators': [1000, 1500, 2000, 2500],
    #                 'max_depth':  [4, 5, 8, -1],
    #                 'num_leaves': [15, 31, 63, 127],
    #                 'subsample': [0.6, 0.7, 0.8, 1.0],
    #                 'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
    #                 'learning_rate' : [0.01,0.02,0.03]
    #                 }
    # gbm = GridSearchCV()

    # lgb.cv(param, train_data, num_round, nfold=5)
    bst.save_model('model_gbm.txt')

    y_pred = bst.predict(test_x, num_iteration=bst.best_iteration)
    y_pred = y_pred.round(0)
    y_pred = y_pred.astype(int)
    return y_pred

# def predict_canceled_forTest(model,x,df):
#     label, acc, val = svm_predict([],x,model)
#     with open('Test_is_canceled_Label.txt','w') as f:
#         f.write(' '.join(str(ele) for ele in label))
#     df.add_column_to_test(label)

def predict_canceled():
    # make train & test same dimension
    notinclude = ['ID', 'is_canceled', 'adr' , 'reservation_status' , 'reservation_status_date']
    # should be preserved not be encode
    notoneHot = ['lead_time',  'adults', 'children', 'babies', 'days', 'total_of_special_requests']
    # redunancy info for predict canceled
    drop_canceled = ['agent', 'country', 'stays_in_weekend_nights', 'stays_in_week_nights', 'date']

    df = DataReader('train.csv', 'test.csv', notinclude, drop_canceled, notoneHot, [])
    df.drop_encode()
    train_y, train_x, test_x = df.getTrainTest_cancel_np()
    
    sklearn(train_y,train_x,test_x)
    # model = train(train_y, train_x)
    # predict_canceled_forTest(model,test_x,df_Obj)
    # model = svm_load_model('is_canceled_0.model')
    # predict_canceled_forTest(model, test_x, df_Obj)
    clf = load('sklearn_ada')
    cancel_predict = clf.predict(test_x)
    # test_cancel_label = []
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
    #     df.add_column_to_test(np.array(p))
    # return np.array(p),df

    # cancel_predict = lightgbm(train_y,train_x,test_x)
    # err= []
    # for i in range(len(cancel_predict)):
    #     if cancel_predict[i] != p[i]:
    #         err.append(i)
    # print(len(err))
    return  cancel_predict, df
'''0 ==========
train score  0.8548446568794811
valid score  0.8513055828690047
1 ==========
train score  0.8530829634687607
valid score  0.8468808041079428
2 ==========
train score  0.8546534653465346
valid score  0.851633344258713
3 ==========
train score  0.8528508023216115
valid score  0.8576423030700316
4 ==========
train score  0.8535336292249914
valid score  0.8564951382060526'''