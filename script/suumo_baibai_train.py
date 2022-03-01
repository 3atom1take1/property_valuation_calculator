#!/usr/bin/env python
# cofing: utf-8
import sys
import pandas as pd
import re
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import glob
import os
import statistics
import yaml

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import shap
import pickle
from sklearn.model_selection import KFold
from mlxtend.regressor import StackingCVRegressor

diff_jst_from_utc = 0
start_time = dt.datetime.now() + dt.timedelta(hours=diff_jst_from_utc)
excution_date = dt.datetime.today().strftime('%Y%m%d')
now_time = (dt.datetime.now() +
            dt.timedelta(hours=diff_jst_from_utc)).strftime('%Y%m%d_%H%M')

work_dir = os.getcwd()
yaml_file = work_dir + '/setting/train_config.yaml'

f = open(yaml_file, 'r')
settings = yaml.safe_load(f)
upper_price = settings['upper_price']
do_grid_search = settings['do_grid_search']

files = glob.glob(work_dir + '/scraping_raw/suumo_baibai*.csv')
input_file = files[0]

train_data_date = input_file[-12:-4]
log_dir = work_dir + '/log/{now_time}_train/'.format(
    now_time=now_time, train_data_date=train_data_date)

os.makedirs(log_dir, exist_ok=True)

log_file = log_dir + '/{}log.txt'.format(train_data_date)

f = open(log_file, 'w', encoding='UTF-8')
f.close()


def write_log(log_file, text):
    f = open(log_file, 'a', encoding='UTF-8')
    f.write(text)
    f.close()
    print(text)


text = 'processing_start_time:' + str(start_time.replace(microsecond=0)) + '\n'
write_log(log_file, text)

text = 'train_data_date:' + str(train_data_date) + '\n'
write_log(log_file, text)

text = 'do_grid_search:' + str(do_grid_search) + '\n'
write_log(log_file, text)


def preprocessing():
    # trainデータの読み込み
    df_raw = pd.read_csv(input_file)

    # 複数戸のデータを削除
    df_raw = df_raw[~df_raw['price'].str.contains('～')]

    # price
    temp_price = []
    for i in df_raw['price']:
        try:
            temp_oku_value = int(i.split('億')[0])
            try:
                temp_man_value = int(i.split('億')[1].split('万')[0])
            except:
                temp_man_value = 0
        except:
            temp_oku_value = 0
            temp_man_value = int(i.split('万')[0])
        temp_pl_value = temp_oku_value * 100000000 + temp_man_value * 10000
        temp_price.append(temp_pl_value)
    df_raw['price'] = temp_price

    # service_room
    df_raw['service_room'] = list(
        map(lambda x: 1 if x.count('+S') > 0 else 0, df_raw['floor_plan']))

    # floor_plan
    df_raw['floor_plan'] = df_raw['floor_plan'].str.replace(r'\+S（納戸）', '')
    df_raw['floor_plan'] = df_raw['floor_plan'].str.replace(r'\+2S（納戸）', '')
    df_raw['floor_plan'] = df_raw['floor_plan'].str.replace(r'\+3S（納戸）', '')
    df_raw['floor_plan'] = list(
        map(lambda x: 'ワンルーム' if x == '1ワンルーム' else x, df_raw['floor_plan']))

    # total_rooms
    df_raw['total_rooms'] = df_raw['total_rooms'].str.replace(r'戸', '')
    df_raw['total_rooms'] = df_raw['total_rooms'].str.replace(r'-', '')
    df_raw['total_rooms'] = list(
        map(lambda x: 0 if x == '' else x, df_raw['total_rooms']))
    df_raw['total_rooms'] = df_raw['total_rooms'].astype(int)

    # exclusive_area
    temp_exclusive_area = []
    for i in df_raw['exclusive_area']:
        temp_ea_value = i[0:3]
        temp_ea_value = re.sub('\.', '', temp_ea_value)
        temp_ea_value = re.sub('m', '', temp_ea_value)
        temp_exclusive_area.append(temp_ea_value)
    df_raw['exclusive_area'] = temp_exclusive_area
    df_raw['exclusive_area'] = df_raw['exclusive_area'].astype(int)

    # other_area
    temp_other_area = []
    for i in df_raw['other_area']:
        try:
            temp_oa_value = i.split('：')[1]
        except:
            temp_oa_value = 0
        try:
            temp_oa_value = temp_oa_value.split('.')[0]
        except:
            pass
        try:
            temp_oa_value = temp_oa_value.split('m')[0]
        except:
            pass
        temp_other_area.append(temp_oa_value)
    df_raw['other_area'] = temp_other_area
    df_raw['other_area'] = df_raw['other_area'].astype(int)

    # stories
    temp_stories = []
    temp_total_stories = []
    for i in df_raw['stories']:
        try:
            temp_st_value = i.split('/')[0]
            temp_st_value = temp_st_value.split('階')[0]
    #         temp_stttl_value = i.split('/')[1]　 # 建物階を入れる場合
    #         temp_stttl_value = temp_stttl_value.split('階')[0]
        except:
            pass
        try:
            temp_st_value = temp_st_value.split('-')[0]
        except:
            pass
        if temp_st_value == '':
            temp_st_value = 0
    #     if temp_stttl_value == '':
    #         temp_stttl_value = 0
        temp_stories.append(temp_st_value)
    #     temp_total_stories.append(temp_stttl_value)
    df_raw['stories'] = temp_stories
    # df_raw['total_stories'] = temp_total_stories
    df_raw['stories'] = df_raw['stories'].str.replace(r'B1', '0').str.replace(
        r'RC', '0').str.replace(r'B2', '0').str.replace(r'ＳＲＣ・軽量鉄骨造7', '0')
    df_raw['stories'] = df_raw['stories'].str.replace(r'00', '0').str.replace(
        r'b2', '0').str.replace(r'B3', '0').str.replace(r'3F', '3')
    df_raw['stories'] = df_raw['stories'].fillna(0)
    df_raw['stories'] = df_raw['stories'].astype(int)

    # age
    temp_completion = []
    for i in df_raw['completion']:
        try:
            temp_cp_value = i.split('年')[0]
        except:
            pass
        temp_completion.append(temp_cp_value)
    df_raw['completion'] = temp_completion
    df_raw['completion'] = df_raw['completion'].astype(int)
    today = datetime.date.today()
    this_year = today.year
    df_raw['age'] = this_year - df_raw['completion']
    df_raw = df_raw.drop(columns='completion')

    # other_area
    temp_adress = []
    for i in df_raw['adress']:
        try:
            temp_ad_value = i.split('\n')[0]
        except:
            pass
        temp_adress.append(temp_ad_value)
    df_raw['adress'] = temp_adress
    df_raw['adress'] = df_raw['adress'].str.replace('-', '').str.replace(
        '\d*', '').str.replace(r'東京都', '')

    # station
    temp_station = []
    temp_from_station = []
    for i in df_raw['access']:
        try:
            temp_st_value1 = i.split('「')[1]
            temp_st_value2 = temp_st_value1.split('」')[0]
            temp_frst_value1 = i.split('歩')[1]
            temp_frst_value2 = temp_frst_value1.split('分')[0]
        except:
            pass
        temp_station.append(temp_st_value2)
        temp_from_station.append(temp_frst_value2)
    df_raw['station'] = temp_station
    df_raw['from_station'] = temp_from_station
    df_raw['from_station'] = df_raw['from_station'].astype(int)
    df_raw = df_raw.drop(columns='access')

    # move_in_date
    temp_move_in_date = []

    for i in df_raw['move_in_date']:
        if i == '即入居可':
            temp_mid = 1
        else:
            temp_mid = 0
        temp_move_in_date.append(temp_mid)
    df_raw['move_in_date'] = temp_move_in_date

    # direction
    df_raw['direction'] = df_raw['direction'].str.replace('-', '不明')

    # reform
    temp_reform = []

    for i in df_raw['reform']:
        if i == '-':
            temp_rf = 0
        else:
            temp_rf = 1
        temp_reform.append(temp_rf)
    df_raw['reform'] = temp_reform

    # ownership
    temp_ownership = []

    for i in df_raw['ownership']:
        try:
            temp_os_value = i.split('（')[0]
        except:
            temp_os_value = i
        temp_ownership.append(temp_os_value)
    df_raw['ownership'] = temp_ownership
    df_raw['ownership'] = df_raw['ownership'].str.replace(
        '建物譲渡特約付定期借地権', '賃借権').str.replace('借地権', '賃借権').str.replace('一部', '')

    # use_district
    temp_use_district = []

    for i in df_raw['use_district']:
        try:
            temp_ud_value = i.split('、')[0]
        except:
            temp_ud_value = i
        temp_use_district.append(temp_ud_value)
    df_raw['use_district'] = temp_use_district
    df_raw['use_district'] = df_raw['use_district'].str.replace('-', '不明')

    df = df_raw[[
        'price', 'floor_plan', 'total_rooms', 'exclusive_area', 'other_area',
        'stories', 'adress', 'move_in_date', 'direction', 'reform',
        'ownership', 'use_district', 'service_room', 'age', 'station',
        'from_station'
    ]]

    df = pd.get_dummies(df)

    df.to_csv(work_dir +
              '/train_data/baibai_train_{}.csv'.format(train_data_date),
              index=False)

    text = 'train_data_shape:' + str(df.shape) + '\n'
    write_log(log_file, text)


# 市場価格と予測価格
def result_plot(y_train, y_train_pred, y_test, y_test_pred, model_type):
    sns.set()
    act_pred_price_file = log_dir + '/{}_act_pred_price.png'.format(model_type)
    plt.figure(figsize=(10, 7))
    plt.scatter(y_train,
                y_train_pred,
                c='red',
                marker='s',
                s=35,
                alpha=0.7,
                label='Train data')
    plt.scatter(y_test,
                y_test_pred,
                c='blue',
                marker='s',
                s=35,
                alpha=0.7,
                label='Test data')
    plt.xlabel('Market_price')
    plt.ylabel('{}_model_Pred_price'.format(model_type))
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, lw=2)
    plt.xlim([0, 150000000])
    plt.ylim([0, 150000000])
    plt.savefig(act_pred_price_file)

    # 市場価格と誤差
    act_diff_file = log_dir + '/{}_act_diff.png'.format(model_type)
    plt.figure(figsize=(10, 7))
    plt.scatter(y_train,
                y_train_pred - y_train,
                c='red',
                marker='s',
                s=35,
                alpha=0.7,
                label='Train data')
    plt.scatter(y_test,
                y_test_pred - y_test,
                c='blue',
                marker='s',
                s=35,
                alpha=0.7,
                label='Test data')
    plt.xlabel('Market_price')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, lw=2)
    plt.xlim([0, 100000000])
    plt.ylim([-50000000, 50000000])
    plt.savefig(act_diff_file)


# モデルの保存
def save_model(model, model_type):
    model_dir = work_dir + '/model/{}'.format(excution_date)
    os.makedirs(model_dir, exist_ok=True)
    model_filename = model_dir + '/{model_type}_{train_data_date}.pickle'.format(
        train_data_date=train_data_date, model_type=model_type)
    pickle.dump(model, open(model_filename, 'wb'))


def main():
    preprocessing()
    df_train_raw = pd.read_csv(
        work_dir + '/train_data/baibai_train_{}.csv'.format(train_data_date))
    df_train_raw = df_train_raw[df_train_raw['price'] < 100000000]

    # 目的変数の設定
    target_column = 'price'

    df_train, df_test = train_test_split(df_train_raw,
                                         test_size=0.2,
                                         random_state=0)

    X_train = df_train.drop(columns=target_column)
    y_train = df_train[target_column]
    X_test = df_test.drop(columns=target_column)
    y_test = df_test[target_column]

    # 線形回帰
    model_type = 'lr'
    lr_model = LinearRegression(n_jobs=-1)
    lr_model.fit(X_train, y_train)
    # 価格予測
    y_train_pred = lr_model.predict(X_train)
    y_test_pred = lr_model.predict(X_test)

    def mean_absolute_percentage_error(y_test, y_test_pred):
        return np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    df_coef = pd.DataFrame({
        "Name": X_train.columns,
        "Coefficients": lr_model.coef_
    }).sort_values(by='Coefficients')
    text = '[{} result]\ny_test_describe:\n'.format(model_type) + str(
        y_test.describe(
        )) + '\n' + 'train_RMSE:' + str(train_rmse) + '\ntest_RMSE:' + str(
            test_rmse) + '\ntrain_MAPE' + str(
                mean_absolute_percentage_error(
                    y_train, y_train_pred)) + '\ntest_MAPE' + str(
                        mean_absolute_percentage_error(y_test, y_test_pred)
                    ) + '\nintercept:' + str(lr_model.intercept_) + '\n' + str(
                        df_coef.sort_values(
                            'Coefficients',
                            ascending=False).head(10)) + '\n' + str(
                                df_coef.sort_values('Coefficients',
                                                    ascending=True).head(10))
    write_log(log_file, text)
    result_plot(y_train, y_train_pred, y_test, y_test_pred, model_type)
    save_model(lr_model, model_type)
    text = 'training {} done.'.format(model_type)
    write_log(log_file, text)

    # Lasso回帰
    model_type = 'lasso'
    lasso_model = LassoCV(alphas=[1, 0.1, 0.01, 0.001]).fit(X_train, y_train)
    # 価格予測
    y_train_pred = lasso_model.predict(X_train)
    y_test_pred = lasso_model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    df_coef = pd.DataFrame({
        "Name": X_train.columns,
        "Coefficients": lasso_model.coef_
    }).sort_values(by='Coefficients')
    text = '[{} result]\ny_test_describe:\n'.format(model_type) + str(
        y_test.describe()
    ) + '\n' + 'train_RMSE:' + str(train_rmse) + '\ntest_RMSE:' + str(
        test_rmse) + '\ntrain_MAPE' + str(
            mean_absolute_percentage_error(
                y_train, y_train_pred)) + '\ntest_MAPE' + str(
                    mean_absolute_percentage_error(y_test, y_test_pred)
                ) + '\nintercept:' + str(lasso_model.intercept_) + '\n' + str(
                    df_coef.sort_values(
                        'Coefficients',
                        ascending=False).head(10)) + '\n' + str(
                            df_coef.sort_values('Coefficients',
                                                ascending=True).head(10))
    write_log(log_file, text)
    result_plot(y_train, y_train_pred, y_test, y_test_pred, model_type)
    save_model(lasso_model, model_type)
    text = 'training {} done.'.format(model_type)
    write_log(log_file, text)

    # Ridge回帰
    model_type = 'ridge'
    ridge_model = RidgeCV(alphas=[10, 1, 0.1, 0.01, 0.001, 0.0005]).fit(
        X_train, y_train)
    # 価格予測
    y_train_pred = ridge_model.predict(X_train)
    y_test_pred = ridge_model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    df_coef = pd.DataFrame({
        "Name": X_train.columns,
        "Coefficients": ridge_model.coef_
    }).sort_values(by='Coefficients')
    text = '[{} result]\ny_test_describe:\n'.format(model_type) + str(
        y_test.describe()
    ) + '\n' + 'train_RMSE:' + str(train_rmse) + '\ntest_RMSE:' + str(
        test_rmse) + '\ntrain_MAPE' + str(
            mean_absolute_percentage_error(
                y_train, y_train_pred)) + '\ntest_MAPE' + str(
                    mean_absolute_percentage_error(y_test, y_test_pred)
                ) + '\nintercept:' + str(ridge_model.intercept_) + '\n' + str(
                    df_coef.sort_values(
                        'Coefficients',
                        ascending=False).head(10)) + '\n' + str(
                            df_coef.sort_values('Coefficients',
                                                ascending=True).head(10))
    write_log(log_file, text)
    result_plot(y_train, y_train_pred, y_test, y_test_pred, model_type)
    save_model(ridge_model, model_type)
    text = 'training {} done.'.format(model_type)
    write_log(log_file, text)

    # LightGBM
    model_type = 'lgb'

    def run_grid_search():
        params = {
            'learning_rate': [0.05, 0.1, 0.3, 0.7, 1],
            'max_depth': [6, 8, 9, 10],
            'min_data_in_leaf': [1, 3, 5, 7],
            'n_estimators': [300],
            'bagging_fraction': [0.8, 1],
            'feature_fraction': [0.6, 1.0],
        }
        gs = GridSearchCV(lgb.LGBMRegressor(),
                          params,
                          cv=5,
                          n_jobs=-1,
                          verbose=1)
        gs.fit(X_train, y_train)
        return gs

    if do_grid_search:
        lgb_model = run_grid_search()
        text = 'best_estimator:' + str(lgb_model.best_estimator_) + '\n'
        write_log(log_file, text)
        text = 'lgb_model.best_score_:' + str(lgb_model.best_score_) + '\n'
        write_log(log_file, text)
    else:
        lgb_model = lgb.LGBMRegressor(bagging_fraction=0.8,
                                      boosting_type='gbdt',
                                      class_weight=None,
                                      colsample_bytree=1.0,
                                      feature_fraction=0.6,
                                      importance_type='split',
                                      learning_rate=0.1,
                                      max_depth=10,
                                      min_child_samples=20,
                                      min_child_weight=0.001,
                                      min_data_in_leaf=7,
                                      min_split_gain=0.0,
                                      n_estimators=300,
                                      n_jobs=-1,
                                      num_leaves=31,
                                      objective=None,
                                      random_state=None,
                                      reg_alpha=0.0,
                                      reg_lambda=0.0,
                                      silent=True,
                                      subsample=1.0,
                                      subsample_for_bin=200000,
                                      subsample_freq=0)
        
        lgb_model.fit(X_train, y_train)

    # 価格予測
    y_train_pred = lgb_model.predict(X_train)
    y_test_pred = lgb_model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    text = '[{} result]\ny_test_describe:\n'.format(model_type) + str(
        y_test.describe(
        )) + '\n' + 'train_RMSE:' + str(train_rmse) + '\ntest_RMSE:' + str(
            test_rmse) + '\ntrain_MAPE' + str(
                mean_absolute_percentage_error(
                    y_train, y_train_pred)) + '\ntest_MAPE' + str(
                        mean_absolute_percentage_error(y_test, y_test_pred))
    write_log(log_file, text)
    result_plot(y_train, y_train_pred, y_test, y_test_pred, model_type)
    shap.initjs()
    explainer = shap.TreeExplainer(model=lgb_model,
                                   feature_perturbation='tree_path_dependent',
                                   model_output='margin')
    shap_values = explainer.shap_values(X_train)
    sns.set()
    shap.summary_plot(shap_values, X_train, plot_type='bar')
    shap_bar_file = log_dir + '/{}_shap_bar.png'.format(model_type)
    plt.savefig(shap_bar_file)

    shap.summary_plot(shap_values, X_train)
    shap_summary_file = log_dir + '/{}_shap_summary.png'.format(model_type)
    plt.savefig(shap_summary_file)

    save_model(lgb_model, model_type)
    text = 'training {} done.'.format(model_type)
    write_log(log_file, text)

    # Ensemble model
    model_type = 'ensemble'
    kfolds = KFold(n_splits = 5 , shuffle = True , random_state = 0)

    def cv_rmse(model, X=X_train):
        rmse = np.sqrt(-cross_val_score(model,X_train,y_train,
                                    scoring = "neg_mean_squared_error", cv=kfolds))
        return(rmse)

    stack_gen = StackingCVRegressor(regressors=(
                                        lr_model
                                        ,lasso_model
                                        ,ridge_model
                                        ,lgb_model
                                        ),
                                    meta_regressor=lgb_model,
                                    use_features_in_secondary=True)
    ensemble_model = stack_gen.fit(np.array(X_train), np.array(y_train))

    # 価格予測
    y_train_pred = ensemble_model.predict(X_train)
    y_test_pred = ensemble_model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    text = '[{} result]\ny_test_describe:\n'.format(model_type) + str(
        y_test.describe(
        )) + '\n' + 'train_RMSE:' + str(train_rmse) + '\ntest_RMSE:' + str(
            test_rmse) + '\ntrain_MAPE' + str(
                mean_absolute_percentage_error(
                    y_train, y_train_pred)) + '\ntest_MAPE' + str(
                        mean_absolute_percentage_error(y_test, y_test_pred))
    write_log(log_file, text)
    result_plot(y_train, y_train_pred, y_test, y_test_pred, model_type)

    save_model(ensemble_model, model_type)
    text = 'training {} done.'.format(model_type)
    write_log(log_file, text)

    end_time = dt.datetime.now() + dt.timedelta(hours=diff_jst_from_utc)
    text = 'end_time:{}\n'.format(end_time)
    write_log(log_file,text)

    processing_time = end_time - start_time
    text = 'processing_time:{}\n'.format(processing_time)
    write_log(log_file,text)

if __name__ == '__main__':
    main()
