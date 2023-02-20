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
    f.write(text)
    f.close()
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


@retry(tries=3, delay=10, backoff=2)
def main():
    def get_html(url):
        r = requests.get(url)
        soup = BeautifulSoup(r.content, "html.parser")
        return soup

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
    # for i in tqdm(range(max_page+1)): 
    for i in tqdm(range(1)): 
        url = base_url.format(page = str(i))
        # get html
        item = get_html(url)
        text = item.findAll("h3", {"class": "property_unit-title"})
        write_log(log_file,text)
        for j in item.findAll("h3", {"class": "property_unit-title"}):
            room_url = 'https://www.kenbiya.com/'+j.find('a').get('href')
            write_log(log_file,'room_url'+room_url)
            try:
                # room_detail_url = room_url + 'bukkengaiyo/?fmlg=t001'
                # get html
                room_item = get_html(room_url)
                # 物件詳細のデータを収集
                data["mansion_name"] = room_item.find("h1", {"class": "inner ttl"}).getText().strip()
                write_log(log_file,data["mansion_name"])
            except:
                write_log(log_file,'error')
            #     data["price"] = room_item.find("p", {"class": "mt7 b"}).getText().strip()
            #     data["floor_plan"] = room_item.findAll("td", {"class": "w290 bdCell"})[1].getText().strip()
            #     data["total_rooms"] = room_item.findAll("td", {"class": "w290 bdCell"})[3].getText().strip()
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
