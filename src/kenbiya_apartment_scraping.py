#!/usr/bin/env python
# cofing: utf-8

# from retry import retry
import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime as dt
from tqdm import tqdm
import re
import math
# from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG
import yaml
import os
from retry import retry
import sys
import time

notebook = True
if notebook:
    work_dir = '/Users/satomitakei/property_valuation_calculator'
    with open(work_dir + '/setting/kenbiya_scraping_config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)
    area_name = 'apartment'
else:
    work_dir = os.getcwd()
    with open(work_dir + '/setting/kenbiya_scraping_config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)
    area_name = sys.argv[1]
if area_name == 'tokyo':
    base_url = config['base_url_tokyo']
elif area_name == 'apartment':
    base_url = config['base_url_apartment']
elif area_name == 'osaka':
    base_url = config['base_url_osaka']
elif area_name == 'fukuoka':
    base_url = config['base_url_fukuoka']

def write_log(log_file, text):
    f = open(log_file, 'a', encoding='UTF-8')
    f.write(text)
    f.close()
    print(text)

def extract_value(pattern,temp_property):
    match = re.search(pattern, temp_property)
    if match:
        return match.group(1)
def get_html(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")
    return soup

diff_jst_from_utc = 0
start_time = dt.datetime.now() + dt.timedelta(hours=diff_jst_from_utc)
now_time = (dt.datetime.now() +
            dt.timedelta(hours=diff_jst_from_utc)).strftime('%Y%m%d_%H%M')

log_dir = work_dir + f'/log/scraping'
os.makedirs(log_dir, exist_ok=True)
log_file = log_dir + f'/{now_time}_log.txt'
f = open(log_file, 'w', encoding='UTF-8')
f.close()

text = 'processing_start_time:' + str(start_time.replace(microsecond=0)) + '\n'
write_log(log_file, text)
excution_date = dt.datetime.today().strftime('%Y%m%d')

# file_name = 'suumo_baibai'
# excution_date = dt.datetime.today().strftime('%Y%m%d')

# @retry(tries=3, delay=10, backoff=2)
def main():
    all_data = []

    # 基本ページurl 
    page = '1'
    url = base_url.format(page = page)
    write_log(log_file,'base_url:'+url+'\n')
    # get html
    item = get_html(url)

    # extract all items
    total_rooms = int(re.sub(r"\D", "",item.find(True,"strong", class_="result_num").get_text()))
    max_page = math.floor(total_rooms/50)+ 1
    text = f"max_page:{max_page} \n"
    write_log(log_file,text)

    url_list = []
    all_data = []
    error_page = []
    # 物件URLの取得
    data= {}

    property_dict = {
        'property_name': '物件名',
        'price': '価格',
        'transportation': '交通',
        'address': '住所',
        'year_built': '築年月',
        'building_structure': '建物構造/階数',
        'land_area': '土地面積',
        'building_area': '建物面積',
        'use_area': '用途地域',
        'floor_plan': '間取り',
        'coverage_ratio': '建ぺい率',
        'floor_area_ratio': '容積率',
        'connecting_road': '摂動状況',
        'fire_rotection_law': '防火法',
        'land_raw': '国土法',
        'transaction_method': '取引態様',
        'delivery': '引渡',
        'current_condition': '現況',
        'cap_rate': '満室時利回り',
        'full_occupancy_incom': '満室時年収/月収',
        'property_name': '物件名',
        'land_rights': '土地権利',
        'last_update_date': '直前の更新日',
        'register_date': '情報公開日',
        'scheduled_update_date': '更新予定日',
        'management_id': '管理ID'
    }
    # for i in tqdm(range(max_page+1)): 
    for i in tqdm(range(3)): 
        all_data = []
        url = base_url.format(page = str(i))
        # get html
        item = get_html(url)
        for j in item.findAll(href=re.compile('/pp2/s/.+/.+/re')):
            room_url = 'https://www.kenbiya.com/'+j.get('href')
            write_log(log_file,'room_url:'+room_url+'\n')
            room_item = get_html(room_url)
            temp_property_data={}
            for k, l in property_dict.items():
                try:
                    temp_property_data[k] = room_item.find('dt', text=l).find_next('dd').get_text()
                except:
                    temp_property_data[k] = ''
            data = {}
            # 物件詳細のデータを収集
            # マンション名
            data["property_name"] = temp_property_data['property_name']

            # 価格
            pattern = r'\n(\d*)億'
            try:
                price_oku = int(extract_value(pattern,temp_property_data['price'])) * 100000000
            except:
                price_oku = 0
            try:
                price_man = int(temp_property_data['price'].replace(',', '').replace('万円', '')) * 10000
            except:
                try:
                    pattern = r'\n*億(\d*)'
                    price_man = int(extract_value(pattern,temp_property_data['price'].replace(',', '').replace('万円', ''))) * 10000
                except:
                    price_man = 0
            data["price"] = price_oku + price_man

            # 交通
            data["train_line"] = temp_property_data['transportation'].split()[0]
            data["station"] = temp_property_data['transportation'].split()[1]
            temp_minutes_from_station = temp_property_data['transportation'].split()[2]
            pattern = r'徒歩(\d.+)分'
            data["minutes_from_station"] = extract_value(pattern,temp_minutes_from_station)

            # 住所
            temp_address= temp_property_data['address']
            pattern = r'^(.+(都|道|府|県))'
            data["prefecture_name"] = extract_value(pattern,temp_address)
            pattern = r'^(.+(市|区))'
            data["city_name"] = extract_value(pattern,temp_address)
            data["town_name"] = temp_address.replace(str(data["prefecture_name"]),'').replace(str(data["city_name"]),'')

            # 築年数
            temp_year_built = temp_property_data['year_built']
            try:
                pattern = r'築(\d*)年'
                data["year_built"] = int(extract_value(pattern,temp_year_built))
            except:
                data["year_built"] = 0

            # 構造
            temp_building_structure= temp_property_data['building_structure']
            pattern = r'^(.+)造'
            data["structure"] = extract_value(pattern,temp_building_structure)

            pattern = r'^.+造(\d+)階'
            data["floor"] = extract_value(pattern,temp_building_structure)

            pattern = r'^.+(\d.+)階建'
            data["max_floor"] = extract_value(pattern,temp_building_structure)

            # 総戸数
            temp_total_rooms = temp_property_data['building_structure'].strip()
            pattern = r'総戸数(\d+)戸'
            data["total_rooms"] = extract_value(pattern,temp_total_rooms)

            # 土地面積
            temp_land_area= temp_property_data['land_area']
            pattern = r'^(\d.+)m²'
            data["land_area"] = extract_value(pattern,temp_land_area)

            # 建物面積
            temp_building_area= temp_property_data['building_area']
            pattern = r'^(\d.+)m²'
            data["building_area"] = extract_value(pattern,temp_building_area)

            # 用途地域
            data["use_area"] = temp_property_data['use_area']
            
            # 間取り
            data["floor_plan"] = temp_property_data['floor_plan'].split()[0]

            # 建ぺい/容積率
            try:
                data["coverage_ratio"] = temp_property_data['coverage_ratio'].split('/')[0].replace(' ％ ')
            except:
                data["coverage_ratio"] = None
            try:
                data["floor_area_ratio"] = temp_property_data['floor_area_ratio'].split('/')[0].replace(' ％ ')
            except:
                data["floor_area_ratio"] = None

            # 接道状況
            data["connecting_road"] = temp_property_data['connecting_road']

            # 防火法 / 国土法
            data["fire_rotection_law"] = temp_property_data['fire_rotection_law']
            data["land_raw"] = temp_property_data['land_raw']

            # 取引態様
            data["transaction_method"] = temp_property_data['transaction_method']

            # 引渡
            data["delivery"] = temp_property_data['delivery']

            # 現況
            data["current_condition"] = temp_property_data['current_condition']

            # 満室時利回り
            try:
                data["cap_rate"] = float(temp_property_data['cap_rate'].replace('％','')) / 100
            except:
                data["cap_rate"] = None

            # 満室時年収
            try:
                data["full_occupancy_incom"] = int(float(temp_property_data['full_occupancy_incom'].split()[0].replace('万円','')) * 10000)
            except:
                data["full_occupancy_incom"] = None

            # 土地権利
            data["land_rights"] = temp_property_data['land_rights']

            # 直前の更新日
            date_format = '%Y年%m月%d日'
            try:
                text = temp_property_data['last_update_date'].replace(' ','')
                data["last_update_date"] = dt.datetime.strptime(text, date_format).date()
            except:
                text = temp_property_data['register_date'].replace(' ','')
                data["last_update_date"] = dt.datetime.strptime(text, date_format).date()

            # 'scheduled_update_date': '更新予定日',
            text = temp_property_data['scheduled_update_date'].replace(' ','')
            data["scheduled_update_date"] = dt.datetime.strptime(text, date_format).date()

            # 'management_id': '管理ID'
            data["management_id"] = temp_property_data['management_id']
            
            # room_url:URL
            data["room_url"] = room_url
            write_log(log_file,'Done:'+data["property_name"])
            time.sleep(5)
            all_data.append(data)
        # except:
        #     time.sleep(3)
        #     write_log(log_file,'error')

        df = pd.DataFrame(all_data,index=None)
        df.to_csv(work_dir+f'/scraping_raw/kenbiya_aprtment_{excution_date}.csv',index = False)
        
    text = 'df_shape:{}\n'.format(df.shape)
    write_log(log_file,text)

    end_time = dt.datetime.now() + dt.timedelta(hours=diff_jst_from_utc)
    text = 'predicting done.\nend_time:{}\n'.format(end_time)
    write_log(log_file,text)

    processing_time = end_time - start_time
    text = 'processing_time:{}\n'.format(processing_time)
    write_log(log_file,text)
if __name__ == '__main__':
    main()