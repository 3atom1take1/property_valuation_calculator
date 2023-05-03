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

notebook = False
if notebook:
    work_dir = '/Users/satomitakei/property_valuation_calculator'
    with open(work_dir + '/setting/homes_scraping_config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)
    area_name = 'saitama'
else:
    work_dir = os.getcwd()
    with open(work_dir + '/setting/homes_scraping_config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)
    area_name = sys.argv[1]

base_area = 'base_url_'+area_name
base_url = config[base_area]

def write_log(log_file, text):
    f = open(log_file, 'a', encoding='UTF-8')
    f.write(text)
    f.close()
    print(text)

diff_jst_from_utc = 0
start_time = dt.datetime.now() + dt.timedelta(hours=diff_jst_from_utc)
now_time = (dt.datetime.now() +
            dt.timedelta(hours=diff_jst_from_utc)).strftime('%Y%m%d_%H%M')

log_dir = work_dir + f'/log/scraping'
os.makedirs(log_dir, exist_ok=True)
log_file = log_dir + f'/{now_time}_{area_name}_log.txt'
f = open(log_file, 'w', encoding='UTF-8')
f.close()

text = 'processing_start_time:' + str(start_time.replace(microsecond=0)) + '\n'
write_log(log_file, text)
excution_date = dt.datetime.today().strftime('%Y%m%d')


def get_html(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    return soup

def extract_value(pattern,temp_property):
    match = re.search(pattern, temp_property)
    if match:
        return match.group(1)
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
    max_page = math.ceil(int(item.find('em', class_='prg-listNumber').get_text()) / 20 )    
    text = f'max_page:{max_page} \n'
    write_log(log_file,text)

    url_list = []
    error_page = []
    # 物件URLの取得
    data= {}
    detail_base_url = 'https://toushi.homes.co.jp'
    for i in tqdm(range(max_page+1)):
    # for i in tqdm(range(4)):
        # get html
        url = base_url.format(page = str(i+1))
        text = f'detail_list_page : {url}'
        write_log(log_file,text)
        item = get_html(url)
        detail_volume = len(item.find_all('a',{'class':'propertyList__link'}))
        for i in range(detail_volume):
            data = {}
            detail_url = detail_base_url+item.find_all('a',{'class':'propertyList__link'})[i].get('href')
            detail_item = get_html(detail_url)
            try:
                # 建物名
                data['property_name'] = detail_item.find('td', {'class': 'madori prg-nameTableItem'}).get_text()
                
                # url
                data['url'] = detail_url
                
                # 価格
                temp_price = detail_item.find('td', {'class': 'prg-priceTableItem'}).get_text().replace('\n','').replace(',','').replace('万円','').replace('億','')
                data['price'] = int(temp_price) * 10000
                
                # 満室時利回り
                data['cap_rate'] = float(detail_item.find('table', {'class': 'bukkenInfo__excerpt'}).select('strong')[1].get_text().replace('％',''))
                
                #満室時年収
                data['full_occupancy_incom'] = data['price'] * data['cap_rate'] / 100
                
                # 住所
                temp_address= detail_item.find('td', {'colspan': '3'}).get_text().replace('\n','').replace('  ','').replace('\r','').replace('地図を見る','')
                pattern = r'^(.+(都|道|府|県))'
                data['prefecture_name'] = extract_value(pattern,temp_address)
                pattern = r'^(.+(市|区|郡))'
                data['city_name'] = extract_value(pattern,temp_address).replace(data['prefecture_name'],'')
                data['town_name'] = temp_address.replace(str(data['prefecture_name']),'').replace(str(data['city_name']),'')
                        
                # 交通
                try:
                    temp_train = detail_item.find('td', {'class': 'prg-accessTableItem'}).get_text().strip().split('\n')[0].replace('\r','')
                    data['train_line'] = temp_train.split(' ')[0]
                    data['station'] = temp_train.split(' ')[1].replace('駅','')
                    temp_minutes_from_station = ' '.join(temp_train.split(' ')[2:])
                    # バス判定
                    if 'バス' in temp_train:    
                        bus_flag = 1
                        pattern = r'バス(\d*)分'
                        try:
                            bus_from_station = int(extract_value(pattern,temp_minutes_from_station))
                        except:
                            bus_from_station = None
                        pattern = r'徒歩(\d*)分'
                        data['minutes_from_station'] = int(extract_value(pattern,temp_minutes_from_station)) + bus_from_station
                        data['bus_flag'] = bus_flag
                    else:
                        bus_flag = 0
                        bus_from_station = 0
                        pattern = r'徒歩(\d*)分'
                        data['minutes_from_station'] = int(extract_value(pattern,temp_minutes_from_station)) + bus_from_station
                        data['bus_flag'] = bus_flag
                except:
                    data['train_line']=''
                    data['station']=''
                    data['bus_flag']=None
                    data['minutes_from_station']=None
                    
                # 築年
                temp_age = detail_item.find('td', {'class': 'period prg-periodTableItem'}).getText()
                if '新築' in temp_age:
                    data['age'] = 0    
                else:
                    pattern = r'築(\d*)年'
                    data['age'] = int(extract_value(pattern,temp_age))
                
                # 間取り
                data['floor_plan'] = detail_item.find('td', {'class': 'madori prg-madoriTableItem'}).getText().replace('\n','').replace('\r','').replace(' ','')
                
                # 総戸数
                temp_total_rooms = detail_item.find('td', {'class': 'prg-allNumberTableItem'}).get_text().replace('\n','').replace('\r','').replace(' ','')
                pattern = r'(\d*)戸'
                data['total_rooms'] = int(extract_value(pattern,temp_total_rooms))

                # 事務所戸数
                temp_total_offices = detail_item.find('td', {'class': 'prg-numberTableItem'}).get_text().replace('\n','').replace('\r','').replace(' ','')
                pattern = r'(\d*)戸'
                data['total_offices'] = int(extract_value(pattern,temp_total_offices))

                # 建物構造
                data['structure'] = detail_item.find('td', {'class': 'prg-structureTableItem'}).get_text().replace('\n','').replace('\r','').replace(' ','')

                # 所在階／階数
                temp_floors = detail_item.find('td', {'class': 'prg-floorsTableItem'}).get_text().replace('\n','').replace('\r','').replace(' ','')
                pattern = r'(\d*)階建'
                data['floors'] = int(extract_value(pattern,temp_floors))
                
                # 建ぺい率/容積率
                temp_building_coverage_ratio = detail_item.find('td', {'class': 'prg-floorAreaRatioTableItem'}).get_text().replace('\n','').replace('\r','').replace(' ','').replace('\xa0%','')
                data['building_coverage_ratio'] = temp_building_coverage_ratio.split('／')[0]
                data['floor_area_ratio'] = temp_building_coverage_ratio.split('／')[1]
                
                # 防火法／国土法
                temp_fire_protection_law = detail_item.find('td', {'class': 'prg-landLawTableItem'}).get_text().replace('\n','').replace('\r','').replace(' ','')
                data['fire_protection_law'] = temp_fire_protection_law.split('／')[0]
                data['land_low'] = temp_fire_protection_law.split('／')[1]
                
                # 土地権利
                data['land_rights'] = detail_item.find('td', {'class': 'prg-rightTableItem'}).get_text().replace('\n','').replace('\r','').replace(' ','')
                
                # 用途地域
                data['use_area'] = detail_item.find('td', {'class': 'prg-areaTableItem'}).get_text().replace('\n','').replace('\r','').replace(' ','')
                
                # 取引様態
                data['transaction_condition'] = detail_item.find('td', {'class': 'prg-aspectTableItem'}).get_text().replace('\n','').replace('\r','').replace(' ','')
                
                # 引渡し
                data['transfer'] = detail_item.find('td', {'class': 'prg-deliveryTableItem'}).get_text().replace('\n','').replace('\r','').replace(' ','')

                # 管理費／修繕積立
                temp_management_fee = detail_item.find('td', {'class': 'prg-managementCostTableItem'}).get_text().replace('\n','').replace('\r','').replace(' ','')
                data['management_fee'] = temp_management_fee.split('／')[0]
                data['repair_reserve_fund'] = temp_management_fee.split('／')[1]

                # 管理方式／管理人
                temp_management_method = detail_item.find('td', {'class': 'prg-managementWayTableItem'}).get_text().replace('\n','').replace('\r','').replace(' ','')
                data['management_method'] = temp_management_method.split('／')[0]
                data['manager'] = temp_management_method.split('／')[1]

                # 接道状況
                temp_connecting_road = detail_item.find('td', {'class': 'prg-setsudouTableItem'}).get_text().replace('\n','').replace('\r','').replace('                  ','')
                data['connecting_road'] = temp_connecting_road.split(' ')[0]
                data['connecting_road_other'] = ' '.join(temp_connecting_road.split(' ')[1:])

                # 地目
                data['classification'] = detail_item.find('td', {'class': 'prg-chimokuTableItem'}).get_text().replace('\n','').replace('\r','').replace(' ','')

                # 建築確認番号
                data['confirmation_number'] = detail_item.find('td', {'class': 'prg-buildConfirmTableItem'}).get_text().replace('\n','').replace('\r','').replace(' ','')

                # 管理ID
                data['management_id'] = detail_item.find('td', {'class': 'prg-buildConfirmTableItem'}).get_text().replace('\n','').replace('\r','').replace(' ','')

                # 現況
                temp_no_vacancy = detail_item.find('span', {'class': 'prg-situationTableItem'}).get_text().replace('\n','').replace('\r','').replace(' ','')
                if '満室' in temp_no_vacancy:
                    data['no_vacancy'] = 1
                else:
                    data['no_vacancy'] = 0

                # 建物面積
                data['building_area'] = float(detail_item.find('td', {'class': 'prg-houseAreaTableItem'}).get_text().replace('\n','').replace('\r','').replace(' ','').split('㎡')[0])

                # 土地面積
                data['land_area'] = float(detail_item.find('td', {'class': 'prg-landAreaTableItem'}).get_text().replace('\n','').replace('\r','').replace(' ','').split('㎡')[0])

                # 専有面積
                try:
                    data['min_exclusive_area'] = float(detail_item.find('td', {'class': 'prg-exclusiveAreaTableItem'}).get_text().replace('\n','').replace('\r','').replace('                  ','').split('㎡ 〜 ')[0])
                except:
                    data['min_exclusive_area'] = None

                try:
                    data['max_exclusive_area'] = float(detail_item.find('td', {'class': 'prg-exclusiveAreaTableItem'}).get_text().replace('\n','').replace('\r','').replace('                  ','').split('㎡ 〜 ')[1].replace('㎡',''))
                except:
                    data['max_exclusive_area'] = None
                write_log(log_file,'\ndetail_url : '+detail_url)
            except Exception as e:
                write_log(log_file,'\n error : '+detail_url+'\n'+str(type(e)))

            all_data.append(data)
            # pd.DataFrame(all_data).to_csv(work_dir+f'/temp/temp_homes_apart_{area_name}_{excution_date}.csv',index = False)
            time.sleep(1)
    save_file_name = work_dir+f'/scraping_raw/homes/apartment_{area_name}_{excution_date}.csv'
    df = pd.DataFrame(all_data)
    df.to_csv(save_file_name,index = False)

    write_log(log_file, f'\nsaved file : {save_file_name}')
    end_time = dt.datetime.now() + dt.timedelta(hours=diff_jst_from_utc)
    write_log(log_file,f'end_time : {end_time}\n'.format(end_time))

    processing_time = end_time - start_time
    write_log(log_file,f'processing_time : {processing_time}\n')
if __name__ == '__main__':
    main()