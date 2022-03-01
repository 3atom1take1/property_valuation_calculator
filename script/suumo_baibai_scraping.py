#!/usr/bin/env python
# cofing: utf-8

# from retry import retry
import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime as dt
from tqdm import tqdm
# from logging import getLogger, StreamHandler, Formatter, FileHandler, DEBUG
import yaml
import os
import retry

work_dir = os.getcwd()
with open(work_dir + '/setting/scraping_config.yaml', 'r') as yml:
    config = yaml.safe_load(yml)
base_url = config['baibai_base_url']

diff_jst_from_utc = 9
start_time = dt.datetime.now() + dt.timedelta(hours=diff_jst_from_utc)

file_name = 'suumo_baibai'
excution_date = dt.datetime.today().strftime('%Y%m%d')

# base_url = "https://suumo.jp/jj/bukken/ichiran/JJ010FJ001/?ar=030&bs=011&ta=13&jspIdFlg=patternShikugun&sc=13101&sc=13102&sc=13103&sc=13104&sc=13105&sc=13113&sc=13106&sc=13107&sc=13108&sc=13118&sc=13121&sc=13122&sc=13123&sc=13109&sc=13110&sc=13111&sc=13112&sc=13114&sc=13115&sc=13120&sc=13116&sc=13117&sc=13119&kb=1&kt=9999999&mb=0&mt=9999999&ekTjCd=&ekTjNm=&tj=0&cnb=0&cn=9999999&srch_navi=1&pn={}"

# def setup_logger(log_folder, modname=__name__):
#     logger = getLogger(modname)
#     logger.setLevel(DEBUG)

#     sh = StreamHandler()
#     sh.setLevel(DEBUG)
#     formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     sh.setFormatter(formatter)
#     logger.addHandler(sh)

#     fh = FileHandler(log_folder) #fh = file handler
#     fh.setLevel(DEBUG)
#     fh_formatter = Formatter('%(asctime)s - %(filename)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s')
#     fh.setFormatter(fh_formatter)
#     logger.addHandler(fh)
#     return logger

# # 保存するファイル名を指定
# log_folder = work_dir+'log/'+file_name+'_{}.log'.format(excution_date)

# # ログの初期設定を行う
# logger = setup_logger(log_folder)

# # ログを出力(debugレベル）
# logger.debug('processing start:',start_time)


# @retry(tries=3, delay=10, backoff=2)
def main():
    print('start_time:'+str(start_time))
    def get_html(url):
        r = requests.get(url)
        soup = BeautifulSoup(r.content, "html.parser")
        return soup

    all_data = []

    # 基本ページurl 
    page = 1
    url = base_url.format(page)

    # get html
    item = get_html(url)

    # extract all items
    max_page = int(item.find("ol",{"class": "pagination-parts"}).findAll('a')[4].getText().strip())
    print("max_page", max_page, "items", len(item))

    url_list = []
    all_data = []
    error_page = []
    # 物件URLの取得
    for i in tqdm(range(max_page+1)): 
        url = base_url.format(i)
        # get html
        item = get_html(url)
        for j in item.findAll("h2", {"class": "property_unit-title"}):
            try:
                room_url = 'https://suumo.jp'+j.find('a').get('href')
                data= {}
                room_detail_url = room_url + 'bukkengaiyo/?fmlg=t001'
                # get html
                room_item = get_html(room_url)

                # 物件詳細のデータを収集
                data["mansion_name"] = room_item.find("td", {"class": "w752 bdCell b"}).getText().strip()
                data["price"] = room_item.find("p", {"class": "mt7 b"}).getText().strip()
                data["floor_plan"] = room_item.findAll("td", {"class": "w290 bdCell"})[1].getText().strip()
                data["total_rooms"] = room_item.findAll("td", {"class": "w290 bdCell"})[3].getText().strip()
                data["exclusive_area"] = room_item.findAll("td", {"class": "w290 bdCell"})[4].getText().strip()
                data["other_area"] = room_item.findAll("td", {"class": "w290 bdCell"})[5].getText().strip()
                data["stories"] = room_item.findAll("td", {"class": "w290 bdCell"})[6].getText().strip()
                data["completion"] = room_item.findAll("td", {"class": "w290 bdCell"})[7].getText().strip()
                data["adress"] = room_item.findAll("td", {"class": "w290 bdCell"})[8].getText().strip()
                data["access"] = room_item.findAll("td", {"class": "w290 bdCell"})[9].getText().strip()

                # 物件詳細概要ページからのデータを取得
                room_detail_item = get_html(room_detail_url)
                data["move_in_date"] = room_detail_item.findAll("td", {"class": "w299 bdCell"})[12].getText().strip()
                data["direction"] = room_detail_item.findAll("td", {"class": "w299 bdCell"})[15].getText().strip()
                data["reform"] = room_detail_item.findAll("td", {"class": "w299 bdCell"})[16].getText().strip()
                data["ownership"] = room_detail_item.findAll("td", {"class": "w299 bdCell"})[23].getText().strip()
                data["use_district"] = room_detail_item.findAll("td", {"class": "w299 bdCell"})[24].getText().strip()
                data["url"] = room_url
                data["log_date"] = excution_date
                all_data.append(data)
            except:
                error_page.append(room_url)

    df = pd.DataFrame(all_data)
    df.to_csv(work_dir+'/scraping_raw/suumo_baibai_{excution_date}.csv'.format(excution_date=excution_date)
            ,index = False)
    print('record_volume:'+str(df.shape))
    end_time = dt.datetime.now() + dt.timedelta(hours=diff_jst_from_utc)
    print('end_time:'+str(end_time))
    print('processing_time:'+str(end_time - start_time))
    
    # logger.debug('success & error:', len(error_page),'/',df.shape[0]+len(error_page))
    # logger.debug('processing start:',start_time)
    # logger.debug('end time:',end_time)
    # logger.debug('processing time：',end_time - start_time)

if __name__ == '__main__':
    main()
