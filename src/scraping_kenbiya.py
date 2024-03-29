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
    area_name = 'tokyo'
else:
    work_dir = os.getcwd()
    with open(work_dir + '/setting/kenbiya_scraping_config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)
    area_name = sys.argv[1]
if area_name == 'tokyo':
    base_url = config['base_url_tokyo']
elif area_name == 'osaka':
    base_url = config['base_url_osaka']
elif area_name == 'fukuoka':
    base_url = config['base_url_fukuoka']

def write_log(log_file, text):
    f = open(log_file, 'a', encoding='UTF-8')
    # f.write(text)
    # f.close()
    print(text)

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

def get_html(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, "html.parser")
    return soup
# @retry(tries=3, delay=10, backoff=2)
# def main():
all_data = []

# 基本ページurl 
page = '1'
url = base_url.format(page = page)
write_log(log_file,'base_url:'+url)
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

def extract_value(pattern,temp_property):
    match = re.search(pattern, temp_property)
    if match:
        return match.group(1)
def main():
    # for i in tqdm(range(max_page+1)): 
    for i in tqdm(range(2)): 
        url = base_url.format(page = str(i))
        # get html
        item = get_html(url)
        for j in item.findAll(href=re.compile("pp1/s/.*/re_.*/")):
            room_url = 'https://www.kenbiya.com/'+j.get('href')
            room_item = get_html(room_url)
            property_dict = {
                'price': '価格',
                'transportation': '交通',
                'address': '住所',
                'year_built': '築年月',
                'building_structure': '建物構造/階数',
                'exclusive_area': '専有面積',
                'floor_plan': '間取り',
                'transaction_method': '取引態様',
                'delivery': '引渡',
                'current_condition': '現況',
                'cap_rate': '満室時利回り',
                'full_occupancy_incom': '満室時年収/月収',
                'property_name': '物件名',
                'land_rights': '土地権利',
                'management_fee_repair_reserve_fund': '管理費/修繕積立',
                'management_company': '管理会社',
                'management_method': '管理方式/管理人',
                'last_update_date': '直前の更新日',
                'scheduled_update_date': '更新予定日',
                'management_id': '管理ID'
            }
            temp_property_data={}
            for i, j in property_dict.items():
                dt_tag = room_item.find('dt', text=j)
                if dt_tag:
                    dd_tag = dt_tag.find_next('dd')
                    if dd_tag:
                        temp_property_data[i] = dd_tag.text.strip()
                    else:
                        temp_property_data[i] = ''
                else:
                    temp_property_data[i] = ''
            try:
                # 物件詳細のデータを収集
                # マンション名
                data["property_name"] = temp_property_data['property_name']

                # 価格
                data["price"] = int(temp_property_data['price'].replace(',', '').replace('万円', '')) * 10000

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
                data["town_name"] = temp_address.replace(data["prefecture_name"],'').replace(data["city_name"],'')

                # 築年数
                temp_year_built = temp_property_data['year_built']
                pattern = r'（築(\d.+)年）'
                data["year_built"] = int(extract_value(pattern,temp_year_built))

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

                # 専有面積
                temp_exclusive_area= temp_property_data['exclusive_area']
                pattern = r'^(\d.+)m²'
                data["exclusive_area"] = extract_value(pattern,temp_exclusive_area)

                # 間取り
                data["floor_plan"] = temp_property_data['floor_plan'].split()[0]

                # 方角
                data["direction"] = temp_property_data['floor_plan'].split()[1]

                # 取引態様
                data["transaction_method"] = temp_property_data['transaction_method']

                # 引渡
                data["delivery"] = temp_property_data['delivery']

                # 現況
                data["current_condition"] = temp_property_data['current_condition']

                # 満室時利回り
                data["cap_rate"] = float(temp_property_data['cap_rate'].replace('％','')) / 100

                # 満室時年収
                data["full_occupancy_incom"] = int(float(temp_property_data['full_occupancy_incom'].split()[0].replace('万円','')) * 10000)

                # 土地権利
                data["land_rights"] = temp_property_data['land_rights']

                # 'management_fee_repair_reserve_fund': '管理費/修繕積立',
                data["management_fee"] = int(temp_property_data['management_fee_repair_reserve_fund'].split('/')[0].replace('円','').replace(',',''))
                data["repair_reserve_fund"] = int(temp_property_data['management_fee_repair_reserve_fund'].split('/')[1].replace('円','').replace(',',''))

                # 管理会社
                data["management_company"] = temp_property_data['management_company']

                # 'management_method': '管理方式/管理人',
                data["management_method"] = re.findall('(.*)\r\n  \r\n*', temp_property_data['management_method'].split('/')[0])[0]
                data["management_person"] = re.findall('\r\n    (.*)', temp_property_data['management_method'].split('/')[1])[0]

                # 直前の更新日
                date_format = '%Y年%m月%d日'
                text = temp_property_data['last_update_date'].replace(' ','')
                data["last_update_date"] = datetime.strptime(text, date_format).date()


                # 'scheduled_update_date': '更新予定日',
                text = temp_property_data['scheduled_update_date'].replace(' ','')
                data["scheduled_update_date"] = datetime.strptime(text, date_format).date()

                # 'management_id': '管理ID'
                data["management_id"] = temp_property_data['management_id']
                write_log(log_file,'Done:'+data["property_name"])
                time.sleep(3)
            except:
                time.sleep(3)
                write_log(log_file,'error')
            
            #     
            #     data["exclusive_area"] = room_item.findAll("td", {"class": "w290 bdCell"})[4].getText().strip()
            #     data["other_area"] = room_item.findAll("td", {"class": "w290 bdCell"})[5].getText().strip()
            #     data["stories"] = room_item.findAll("td", {"class": "w290 bdCell"})[6].getText().strip()
            #     data["completion"] = room_item.findAll("td", {"class": "w290 bdCell"})[7].getText().strip()
            #     data["adress"] = room_item.findAll("td", {"class": "w290 bdCell"})[8].getText().strip()
            #     data["access"] = room_item.findAll("td", {"class": "w290 bdCell"})[9].getText().strip()

            #     # 物件詳細概要ページからのデータを取得
            #     room_detail_item = get_html(room_detail_url)
            #     data["move_in_date"] = room_detail_item.findAll("td", {"class": "w299 bdCell"})[12].getText().strip()
            #     data["direction"] = room_detail_item.findAll("td", {"class": "w299 bdCell"})[15].getText().strip()
            #     data["reform"] = room_detail_item.findAll("td", {"class": "w299 bdCell"})[16].getText().strip()
            #     data["ownership"] = room_detail_item.findAll("td", {"class": "w299 bdCell"})[23].getText().strip()
            #     data["use_district"] = room_detail_item.findAll("td", {"class": "w299 bdCell"})[24].getText().strip()
            #     data["url"] = room_url
            #     data["log_date"] = excution_date
            #     all_data.append(data)
            # except:
            #     error_page.append(room_url)

        df = pd.DataFrame(data)
        df.to_csv(work_dir+f'/scraping_raw/kenbiya_baibai_{excution_date}.csv',index = False)
        
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