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
base_url = config['chintai_base_url']

diff_jst_from_utc = 9
start_time = dt.datetime.now() + dt.timedelta(hours=diff_jst_from_utc)

file_name = 'suumo_chintai'
excution_date = dt.datetime.today().strftime('%Y%m%d')

# base_url = "https://suumo.jp/jj/chintai/ichiran/FR301FC001/?ar=030&bs=040&ta=13&sc=13101&sc=13102&sc=13103&sc=13104&sc=13105&sc=13113&sc=13106&sc=13107&sc=13108&sc=13118&sc=13121&sc=13122&sc=13123&sc=13109&sc=13110&sc=13111&sc=13112&sc=13114&sc=13115&sc=13120&sc=13116&sc=13117&sc=13119&cb=0.0&ct=9999999&mb=0&mt=9999999&et=9999999&cn=9999999&shkr1=03&shkr2=03&shkr3=03&shkr4=03&sngz=&po1=25&pc=50&page={}"


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
# log_folder = work_dir+'/log/'+file_name+'_{}.log'.format(excution_date)

# # ログの初期設定を行う
# logger = setup_logger(log_folder)

# # ログを出力(debugレベル）
# logger.debug('processing start:',start_time)




# @retry(tries=3, delay=10, backoff=2)
def main():
    def get_html(url):
        r = requests.get(url)
        soup = BeautifulSoup(r.content, "html.parser")
        return soup

    # 基本ページurl 
    page = 1
    url = base_url.format(page)

    # get html
    item = get_html(url)

    # extract all items
    max_page = int(item.find("ol",{"class": "pagination-parts"}).findAll('a')[4].getText().strip())
    print("max_page", max_page, "items", len(item))
    all_data = []
    error_page = []
    for page in tqdm(range(1, max_page+1)):
        # define url 
        url = base_url.format(page)

        # get html
        soup = get_html(url)

        # extract all items
        items = soup.findAll("div", {"class": "cassetteitem"})
        print("page", page, "items", len(items))

        # process each item
        for item in items:
            stations = item.findAll("div", {"class": "cassetteitem_detail-text"})

            # process each station 
            for station in stations:
                try:
                    # define variable 
                    base_data = {}

                    # collect base information    
                    base_data["mansion_name"] = item.find("div", {"class": "cassetteitem_content-title"}).getText().strip()
                    base_data["type"] = item.find("div", {"class": "cassetteitem_content-label"}).getText().strip()
                    base_data["adress"] = item.find("li", {"class": "cassetteitem_detail-col1"}).getText().strip()
                    base_data["access"] = station.getText().strip()
                    base_data["age"] = item.find("li", {"class": "cassetteitem_detail-col3"}).findAll("div")[0].getText().strip()
                    base_data["structure"] = item.find("li", {"class": "cassetteitem_detail-col3"}).findAll("div")[1].getText().strip()

                    # process for each room
                    tbodys = item.find("table", {"class": "cassetteitem_other"}).findAll("tbody")

                    for tbody in tbodys:
                        data = base_data.copy()

                        data["stories"] = tbody.findAll("td")[2].getText().strip()

                        data["price"] = tbody.findAll("td")[3].findAll("li")[0].getText().strip()
                        data["maintenance_fee"] = tbody.findAll("td")[3].findAll("li")[1].getText().strip()

                        data["deposit"] = tbody.findAll("td")[4].findAll("li")[0].getText().strip()
                        data["key_money"] = tbody.findAll("td")[4].findAll("li")[1].getText().strip()

                        data["floor_plan"] = tbody.findAll("td")[5].findAll("li")[0].getText().strip()
                        data["exclusive_area"] = tbody.findAll("td")[5].findAll("li")[1].getText().strip()
                        url = "https://suumo.jp" + tbody.findAll("td")[8].find("a").get("href")

                        data["url"] = url
                        data["log_date"] = excution_date
                        all_data.append(data)
                except:
                    error_page.append(url)

    df = pd.DataFrame(all_data)
    df.head()
    df.to_csv(work_dir + '/scraping_raw/suumo_chintai_{excution_date}.csv'.format(excution_date=excution_date)
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