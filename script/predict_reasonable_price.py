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

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LassoCV, LassoLarsCV
# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import RidgeCV
# from sklearn.model_selection import GridSearchCV
# import lightgbm as lgb
# import shap
import pickle
# from sklearn.model_selection import KFold
# from mlxtend.regressor import StackingCVRegressor

diff_jst_from_utc = 0
start_time = dt.datetime.now() + dt.timedelta(hours=diff_jst_from_utc)
excution_date = dt.datetime.today().strftime('%Y%m%d')
now_time = (dt.datetime.now() +
            dt.timedelta(hours=diff_jst_from_utc)).strftime('%Y%m%d_%H%M')

def write_log(log_file, text):
    f = open(log_file, 'a', encoding='UTF-8')
    f.write(text)
    f.close()
    print(text)

work_dir = os.getcwd()
log_dir = work_dir + '/log/{now_time}_predict/'.format(now_time=now_time)
os.makedirs(log_dir, exist_ok=True)

# trainデータの読み込み
files = glob.glob(work_dir + '/train_data/baibai_train*.csv')
input_file = files[-1]
train_data_date = input_file[-12:-4]

log_file = log_dir + '/{}log.txt'.format(train_data_date)
f = open(log_file, 'w', encoding='UTF-8')
f.close()

# with open(work_dir + '/setting/predict_suumo_baibai_config.yaml', 'r') as yml:
#     config = yaml.safe_load(yml)

# config.yamlの読み込み
# mansion_name = config['mansion_name']
# price = config['price']
# adress = config['adress']
# station = config['station']
# from_station = config['from_station']
# floor_plan = config['floor_plan']
# exclusive_area = config['exclusive_area']
# age = config['age']
# stories = config['stories']
# direction = config['direction']
# other_area = config['other_area']
# total_rooms = config['total_rooms']
# move_in_date = config['move_in_date']
# reform = config['reform']
# ownership = config['ownership']
# use_district = config['use_district']
# url = config['url']
# log_date = config['log_date']
# service_room = config['service_room']


text = 'processing_start_time:' + str(start_time.replace(microsecond=0)) + '\n'
write_log(log_file, text)

text = 'train_data_date:' + str(train_data_date) + '\n'
write_log(log_file, text)

# text = 'mansion_name:' + str(mansion_name) + '\n' + 'adress:' + str(adress) + '\n' + 'station:' + str(station) + '\n' + 'from_station:' + str(from_station) + '\n' + 'floor_plan:' + str(floor_plan) + '\n' + 'exclusive_area:' + str(exclusive_area) + '\n' + 'age:' + str(age) + '\n' + 'stories:' + str(stories) + '\n' + 'direction:' + str(direction) + '\n' + 'other_area:' + str(other_area) + '\n'
# write_log(log_file, text)

def main():
    # target_room = {}
    # target_room['mansion_name'] = [mansion_name]
    # target_room['price'] = [price]
    # target_room['floor_plan'] = [floor_plan]
    # target_room['total_rooms'] = [total_rooms]
    # target_room['exclusive_area'] = [exclusive_area]
    # target_room['other_area'] = [other_area]
    # target_room['stories'] = [stories]
    # target_room['adress'] = [adress]
    # target_room['move_in_date'] = [move_in_date]
    # target_room['direction'] = [direction]
    # target_room['reform'] = [reform]
    # target_room['ownership'] = [ownership]
    # target_room['use_district'] = [use_district]
    # target_room['service_room'] = [service_room]
    # target_room['age'] = [age]
    # target_room['station'] = [station]
    # target_room['from_station'] = [from_station]
    # target_room['url'] = [url]
    # target_room['log_date'] = [log_date]
    # df_target_room = pd.DataFrame(target_room)
    # df_target_room = df_target_room[['mansion_name','price', 'floor_plan', 'total_rooms', 'exclusive_area'
    #                                 ,'other_area','stories', 'adress', 'move_in_date', 'direction', 'reform'
    #                                 ,'ownership', 'use_district', 'service_room', 'age', 'station'
    #                                 ,'from_station','url','log_date']]
    # trainデータの読み込み
    df_raw = pd.read_csv(input_file)
    # df_raw = df_raw.append(df_target_room)
    text = 'shape:' + str(df_raw.shape) + '\n'
    write_log(log_file, text)
    df = df_raw[[
        'price', 'floor_plan', 'total_rooms', 'exclusive_area', 'other_area',
        'stories', 'adress', 'move_in_date', 'direction', 'reform',
        'ownership', 'use_district', 'service_room', 'age', 'station',
        'from_station'
    ]]
    df = pd.get_dummies(df)
    # 目的変数の設定
    target_column = 'price'
    X_test = df.drop(columns = target_column)
    y_test = df[target_column]

    #　モデルの読み込み
    files = glob.glob(work_dir + '/model/*/ensemble_*.pickle')
    model_file = files[-1]
    text = 'model_file:' + str(model_file) + '\n'
    write_log(log_file, text)
    model = pickle.load(open(model_file, 'rb'))

    # 価格予測
    y_test_pred = model.predict(X_test.values)
    df_raw['pred_price'] = y_test_pred
    df_raw['real_per_pred'] = df_raw['price'] / df_raw['pred_price'] - 1.0
    df_raw = df_raw[['mansion_name'
                    ,'url'
                    ,'price'
                    ,'pred_price'
                    ,'real_per_pred'
                    ,'station'
                    ,'from_station'
                    ,'adress'
                    ,'age'
                    ,'floor_plan'
                    ,'exclusive_area'
                    ,'stories'
                    ,'direction'
                    ,'total_rooms'
                    ,'other_area'
                    ,'reform'
                    ,'ownership'
                    ,'use_district'
                    ,'move_in_date'
                    ,'service_room'
                    ,'log_date']]
    df_raw = df_raw.query('10000000 <= price <= 40000000 & real_per_pred < -0.2 & exclusive_area < 50 & stories >= 2 & ownership =="所有権" &from_station < 10 ')
    df_raw = df_raw.sort_values('real_per_pred')
    df_raw.to_csv(work_dir + '/pred_output/pred_output_{}.csv'.format(excution_date),index = False)

    end_time = dt.datetime.now() + dt.timedelta(hours=diff_jst_from_utc)
    text = 'predicting done.\nend_time:{}\n'.format(end_time)
    write_log(log_file,text)

    processing_time = end_time - start_time
    text = 'processing_time:{}\n'.format(processing_time)
    write_log(log_file,text)

if __name__ == '__main__':
    main()