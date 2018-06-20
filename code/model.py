import pandas as pd
from collections import Counter
import datetime
import numpy as np
import warnings
import lightgbm as lgb
from sklearn import model_selection
from sklearn.metrics import f1_score
from sklearn.grid_search import GridSearchCV

warnings.filterwarnings('ignore')

today = '0620'
# =============================================================================
LGBM = lgb.LGBMClassifier(  max_depth=3,
                            n_estimators = 120,
                            learning_rate =0.05,
                            objective = 'binary',
                            subsample=0.7,
                            colsample_bytree=0.74,
                            num_leaves=8
)
# =============================================================================
# LGBM = lgb.LGBMClassifier(  max_depth=6,
#                                 n_estimators = 280,  #350,500    , tree num = class_num * n_iter
#                                 learning_rate =0.05, #0.05,0.025
#                                 objective = 'binary',
#                                 subsample=0.7,
#                                 # colsample_bytree=0.74,
#                                 num_leaves=25,
#                                 boosting_type='dart',
#                                feature_fraction=0.5,
#                                     lambda_l1=1,
#                                     lambda_l2=0.5,
#                                 )

def get_data(path):
    # data
    data1_in = pd.read_csv(path + 'data1.csv')
    data2_in = pd.read_csv(path + 'data2.csv')
    data3_in = pd.read_csv(path + 'data3.csv')

    drop_cols = ['login_sum', 'login_max', 'loginvar', 'loginmean', 'login_3_cnt', 'login_week_cnt', 'device_map',
                 'page_sum','page_0_sigle','page_1_sigle','page_2_sigle','page_3_sigle','page_4_sigle',
                 'action_type_sum','action_type_0_sigle','action_type_1_sigle','action_type_2_sigle',
                 'action_type_3_sigle','action_type_4_sigle','action_type_5_sigle']

    f_r = open(r"F:\kuaishou\data\result_importance_2.txt",'r')
    a = f_r.readlines()
    save_cols = []
    for i in a:
        bb = i.split()
        save_cols.append(bb[0])
    save_cols.append('user_id')
    save_cols.append('label')

    # select_cols = ['login_week_cnt', 'login_day_max', 'act_last_cnt', 'device_type',
    #                'login_last_cnt', 'act_day_max', 'register_type', 'login_day_std',
    #                'action_type_0', 'act_week_cnt', 'act_day_std', 'login_week_arg_cnt',
    #                'act_sum', 'login_cnt', 'page_1', 'page_0', 'action_type_1', 'page_2',
    #                'act_day_min', 'action_type_3', 'act_week_arg_cnt', 'page_3',
    #                'register_day', 'action_type_2', 'page_4', 'act_max', 'act_cnt',
    #                'video_last_cnt', 'act_arg', 'login_sum', 'login_day_min',
    #                'video_day_max', 'video_sum', 'video_arg', 'video_week_cnt']

    def mapDeviceType(thread_value=0.5):
        con_data = pd.concat([data1_in, data2_in])
        index = con_data['label'].groupby(con_data["device_type"]).mean().index
        values = con_data['label'].groupby(con_data["device_type"]).mean().get_values()
        return index[values > thread_value]

    good_index = mapDeviceType()

    data1_in['device_map'] = data1_in['device_type'].apply(lambda x: int(x in good_index))
    data2_in['device_map'] = data1_in['device_type'].apply(lambda x: int(x in good_index))
    data3_in['device_map'] = data1_in['device_type'].apply(lambda x: int(x in good_index))

# =============================================================================
#     data1 = data1_in[[c for c in data1_in.columns if c not in drop_cols]]
#     data2 = data2_in[[c for c in data2_in.columns if c not in drop_cols]]
#     data3 = data3_in[[c for c in data3_in.columns if c not in drop_cols]]
# =============================================================================
    data1 = data1_in[[c for c in data1_in.columns if c  in save_cols]]
    data2 = data2_in[[c for c in data2_in.columns if c  in save_cols]]
    data3 = data3_in[[c for c in data3_in.columns if c  in save_cols]]

    return data1,data2,data3

def buildModelAndPredict(path,isOnLine=True, isTest=False, yuzhi=0.4, model=LGBM):

    data1, data2, data3 = get_data(path)
    print(data1.columns.size-2)

    if (isOnLine):
        # yuzhi=0.4
        train = pd.concat([data1, data2])
        test = data3.copy()
        train.pop('user_id')
        label = train.pop('label')
        model.fit(train, label)
        user_list = test.pop('user_id')
        print(len(user_list))
        user_df = pd.DataFrame(user_list)
        user_df['pre_act'] = model.predict_proba(test)[:, 1]
        user = user_df.sort_values(by='pre_act', ascending=False).head(yuzhi)
        user_pre = user.user_id
        return user_pre

    else:
        # best yuzhi 0.6
        train = data1.copy()
        test = data2.copy()
        # train pop user_id and get label
        train.pop('user_id')
        train_df_label = train.pop('label')
        train_df = train

        # test get user_id and pop label
        real_user = test[test.label == 1]['user_id']
        user_list = test.pop('user_id')
        test.pop('label')
        test_df = test

        user_df = pd.DataFrame(user_list)
        # train the model and predict
        model.fit(train_df, train_df_label)
        user_df_ = model.predict_proba(test_df)
        user_df['pre_act'] = model.predict_proba(test_df)[:, 1]

        user = user_df.sort_values(by='pre_act', ascending=False).head(yuzhi)
        user_pre = user.user_id

        # calculate the F1 score
        if (isTest):
            for i in np.arange(0.3, 0.8, 0.01):
                user_pre = user_df[user_df.pre_act > i]['user_id']
                sroceF1(user_pre, real_user)
                print(i)
        else:
            # user_pre = user_df[user_df.pre_act > 0.4]['user_id']
            # print(len(user_pre), len(real_user))
            F1_s = sroceF1(user_pre, real_user)
        # return None
        return model.feature_importances_ , train.columns, F1_s


def sroceF1(pred, real):
    M = set(pred)
    N = set(real)
    Precision = len(M.intersection(N))/len(M)
    Recall = len(M.intersection(N))/len(N)
    F1 = 2*Precision*Recall/(Precision+Recall)

    print("Precision=",Precision,"| Recall=",Recall)
    print("F1=",F1)
    return F1

def cv_LGBM():
    data1, data2, data3 = get_data(path)

    # train = pd.concat([data1, data2])
    # train.pop('user_id')
    # label = train.pop('label')
    # X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train, label, test_size=0.3,
    #                                                                     random_state=1017)
    X_train = data1.copy()
    X_train.pop('user_id')
    Y_train = X_train.pop('label')

    X_test = data2.copy()
    X_test.pop('user_id')
    Y_test = X_test.pop('label')


    param_val1 = {'n_estimators':list(range(200,501,10))}

    param_val2 = {'max_depth':[6,7,8],#list(range(3,9,1)),
                  'num_leaves':list(range(20, 101, 5))}#list(range(8,201,4))}

    param_val3 = {'max_bin':list(range(5,50,5)),
                  'min_data_in_leaf':list(range(10,200,5))}

    param_val4 = {'feature_fraction':[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]}

    param_val5 =  {'lambda_l1':[0.05,0.1,0.5,1,10,100,1000],
                   'lambda_l2':[0.05,0.1,0.5,1,10,100,1000]}

    param_val = {
#                 'num_leaves': list(range(20, 100, 5)),
#                  'max_depth':[6,7,8],
                'n_estimators':list(range(200,501,10)),
                #  'feature_fraction': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
                 # 'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
#                  'learning_rate': [0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
#                  'max_bin':list(range(5,51,5)),
#                 'lambda_l1':[0.05,0.1,0.5,1,10,100,1000],
#                 'lambda_l2':[0.05,0.1,0.5,1,10,100,1000],
                 }

    LGBM_cv = lgb.LGBMClassifier(
                                max_depth=6,
                                n_estimators = 370,  #350,500    , tree num = class_num * n_iter
                                learning_rate =0.1, #0.05,0.025
                                objective = 'binary',
                                subsample=0.7,
                                num_leaves=20,
                                boosting_type='dart',
                               feature_fraction=0.5,
                                lambda_l1=100,
                                lambda_l2=1000,
                                max_bin=15,
                                min_data_in_leaf=165,
                                )
    cv1 = GridSearchCV(estimator=LGBM_cv,
                       param_grid= param_val5, scoring='f1',iid=False,n_jobs=6,verbose=1)
    cv1.fit(X_train,Y_train)
    print(str(cv1.grid_scores_),str(cv1.best_params_),str(cv1.best_score_))
    print(str(cv1.best_params_),str(cv1.best_score_))
#    print(str(cv1.best_score_)+'\n')

def error_num(file_name):
    compare = pd.read_csv(file_name, header=None)
    good_f1 = pd.read_csv(path + '24038.csv',header=None)
    all_right= pd.read_csv(path + '19580.csv',header=None)
    compare = set(compare[0].values)
    good_f1 = set(good_f1[0].values)
    all_right = set(all_right[0].values)
    pd_f1 = len(good_f1.intersection(compare))
    pd_r = len(all_right.intersection(compare))
    print('Compare with all_right:'+str(pd_r))
    print('Compare with good_f1:' + str(pd_f1))
    print('The error user number:'+str(len(compare)-len(all_right)))
    error_ = len(compare)-len(all_right)
    return error_


if __name__ == '__main__':

    # path = r"F:\downloads_Python\Test\kesci\data\\"
    path = r'F:\kuaishou\data\\'

    LGBM_new = lgb.LGBMClassifier(
                                max_depth=6,
                                n_estimators = 370,  #350,500    , tree num = class_num * n_iter
                                learning_rate =0.1, #0.05,0.025
                                objective = 'binary',
                                subsample=0.7,
                                num_leaves=20,
                                boosting_type='dart',
                               feature_fraction=0.5,
                                lambda_l1=100,
                                lambda_l2=1000,
                                max_bin=15,
                                min_data_in_leaf=165,
                                )
    yuzhi = 24000
    feature_importances, columns_name, F1 = buildModelAndPredict(path, isOnLine=False, isTest=False, yuzhi=yuzhi, model=LGBM_new)
    user_pre = buildModelAndPredict(path,isOnLine=True, isTest=False, yuzhi=yuzhi, model=LGBM_new)  # 0.38
    print(len(user_pre))
    Output_name = 'result_' + today + '_' + str(F1)[2:] + '_' + str(len(user_pre)) + '_' + str(yuzhi)[2:] + '.csv'
    user_pre.to_csv(path + 'output/'+Output_name, index=False)
    feature_importances = 100 * (feature_importances / max(feature_importances))
    columns_name = list(columns_name)
    feature_importances = feature_importances.tolist()

    f3_name = r"F:\kuaishou\data\result_importance.txt"
    f3 = open(f3_name, 'w')
    ind = []
    for i, v in enumerate(feature_importances):
        if float(v) > 0:
            ind.append(i)
            f3.write(columns_name[i] + '\t' + str(feature_importances[i]) +'\n')
    print(len(ind))

    # cv_LGBM()

    # file_name = r"C:\Users\Administrator\Desktop\result0609_select_col.csv"
    # error_num(file_name)