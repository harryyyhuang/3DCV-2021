import argparse
import csv

import requests
from selenium import webdriver
from bs4 import BeautifulSoup
from urllib.request import urlretrieve
import os
import time
import shutil



# Do scroll down
# def execute_scroll(times):
#     for i in range(times + 1):
#         driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#         time.sleep(0.8)
from selenium.webdriver.common.keys import Keys

parser = argparse.ArgumentParser(description='The Implementation using PyTorch')
parser.add_argument('--keyword', type=str, default="car", help='keyword you want to search')
parser.add_argument('--page', type=str, default="2", help='page you want to go')
parser.add_argument('--website', type=str, default="https://www.cgtrader.com/", help='website you want to go, only https://www.cgtrader.com/ support now')

opt = parser.parse_args()


# Web crawler
def main(opt):


    # login
    # URL = "https://accounts.google.com/o/oauth2/auth/oauthchooseaccount?access_type=offline&client_id=41826574703.apps.googleusercontent.com&redirect_uri=https%3A%2F%2Fwww.cgtrader.com%2Fusers%2Fauth%2Fgoogle_oauth2%2Fcallback&response_type=code&scope=email%20profile&state=6a855d113e5c693a56e66639d2855d5deb1d899847401cac&flowName=GeneralOAuthFlow"
    # driver = webdriver.Chrome()
    # driver.get(URL)

    # elem_user = driver.find_element_by_name("username")
    # elem_user.clear
    # elem_user.send_keys("user")
    # elem_pwd = driver.find_element_by_name("password")
    # elem_pwd.clear
    # elem_pwd.send_keys("password")
    # elem_pwd.send_keys(Keys.RETURN)

    # driver.find_element_by_id('Login').click()
    # agent = requests.get(URL, auth=('210509fssh', 'jk745ol992'))

    # return

    if opt.keyword == '':
        opt.keyword = "car"
    main_URL = "https://www.cgtrader.com/free-3d-models?keywords=" + opt.keyword + "&page=" + opt.page
    agent = requests.get(main_URL)
    agent = BeautifulSoup(agent.text, 'html.parser')
    print("agent in:", main_URL + "\n")
    # print("\n\n\n\n")

    items = agent.find_all('a', {"class": "content-box__link"})

    # Get keyword search locations URL
    download_pages = []
    for item in items:
        URL = item.get('href')
        # agent = requests.get(URL)
        agent_item = requests.get(URL)
        agent_item = BeautifulSoup(agent_item.text, 'html.parser')
        download_btn = agent_item.find_all('a', {"class": "btn btn-secondary btn-block"})[0]
        download_pages_url = download_btn.get('href')
        download_pages.append(download_pages_url)
        print("download page collect: " + download_pages_url)
        # if len(download_pages)> 10:
        #     break


    print("len(items):", len(items), "len(download_pages)", len(download_pages))
    # print("now agent in :",html)

    # download file 
    print("starting saving file......")
    save_folder = './download_3D_model'
    try:
        os.stat(save_folder)
    except:
        os.makedirs(save_folder)
    save_root = os.path.join(save_folder, 'car')

    count = 0
    
    if int(opt.page) > 2 or opt.keyword != "car":
        csvfile = open(os.path.join(save_root, 'output.csv'), 'a+', newline='', encoding='UTF-8')
        writer = csv.writer(csvfile)
    else:
        csvfile = open(os.path.join(save_root, 'output.csv'), 'w', newline='', encoding='UTF-8')
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'downloadLink'])

    # each model
    for download_page in download_pages:
        if download_page is None:
            continue
        id = str(download_page).split('/')[2]
        agent = requests.get(opt.website + download_page)
        agent = BeautifulSoup(agent.text, 'html.parser')
        download_queue = []
        candidates = agent.find_all('li')
        for candidate in candidates:
            download_links = candidate.find_all('a', {"data-id": id})
            if len(download_links) > 0:
                filename = list(candidate.stripped_strings)[0]
                if  str(filename).__contains__('.ply') or  str(filename).__contains__('.stl') or  str(filename).__contains__('.obj'):
                    priority = -1
                    if str(filename).__contains__('.ply'):
                        priority = 1
                    elif str(filename).__contains__('.stl'):
                        priority = 2
                    elif str(filename).__contains__('.obj'):
                        priority = 3
                    download_queue.append({'NAME':filename, "URL":download_links[0], "priority": priority})

        # save_path = os.path.join(save_root, id)
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)

        # print("start download...")
        already_save = False
        download_queue = sorted(download_queue, key = lambda obj: obj['priority'])
        for obj in download_queue:
           filename = obj['NAME']
           # response = requests.get(opt.website + obj["URL"].get('href'), allow_redirects=True, auth=('210509fssh', 'jk745ol992'))
           # open(os.path.join(save_path, filename), 'wb').write(response.content)
           writer.writerow([filename, os.path.join(opt.website, obj["URL"].get('href')[1:])])
           print(filename, os.path.join(opt.website, obj["URL"].get('href')[1:]))
           count += 1
           break
        # del response


    print("finish download image,", count, "file save")


if __name__ == "__main__":
    for i in range(1, 31):
        opt.page = str(i)
        main(opt)
