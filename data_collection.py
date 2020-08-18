# import packages
from bs4 import BeautifulSoup
from requests import get
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from decimal import Decimal
from re import sub
import re
import sqlite3
import urllib.request
from rightmove_webscraper import RightmoveData
import time

# rm = 'https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E87490&maxPrice=3000000&sortType=6&propertyTypes=&maxDaysSinceAdded=1&includeSSTC=true&mustHave=garden&dontShow=newHome%2Cretirement%2CsharedOwnership&furnishTypes=&keywords='
# response = get(rm)

# soup = BeautifulSoup(response.text, 'lxml')

# return a list of property links and ids from search page


# def get_links(soup):
#     link_container = soup.find_all('a', {'class': 'propertyCard-headerLink'})
#     links = []
#     ids = []
#     rmpage = 'http://www.rightmove.co.uk'
#     for link in link_container:
#         links.append(rmpage+link['href'])
#         ids.append(re.findall(r'\d+', link['href'])[0])
#     return links, ids

# download images given the url and file path


def link_to_img(link, id, file_path, i=1):
    file_name = 'image-{}-{}.jpg'.format(id, i)
    full_path = file_path+file_name
    urllib.request.urlretrieve(link, full_path)
    return None

# a function takes in property links and downloads images and floor plans to local folder


# def get_img(links, ids):
#     file_path_img = 'images/'
#     file_path_fp = 'floorplan/'
#     for link, _id in zip(links, ids):
#         response = get(link)
#         soup = BeautifulSoup(response.text, 'lxml')
#         try:
#             fp_link = soup.find('img', src=lambda x: x and "FLP" in x)['src']
#             link_to_img(fp_link, _id, file_path_fp)
#         except:
#             print('No floor plan!')
#         img_links = get_img_links(soup)
#         for i, url in enumerate(img_links):
#             link_to_img(url, _id, file_path_img, i)

#         time.sleep(5)

#     return None


# from search page to property pages and scrape texts and store in SQL database
def get_prop_data(links, ids):
    print('There are {} listings today.'.format((len(links))))
    keys = []
    title = []
    price = []
    description = []
    file_path_img = '../propertyimages/'
    file_path_fp = '../floorplan/'

    conn = sqlite3.connect('../database/rightmove.db')
    c = conn.cursor()

    n = 0

    for link, _id in zip(links, ids):
        n += 1
        print('This is property number {}'.format(n))
        response = get(link)
        soup = BeautifulSoup(response.text, 'lxml')
        keys.append(_id)
        title.append(get_title(soup))
        price.append(get_price(soup))
        description.append(get_desc(soup))
        try:
            c.execute(""" INSERT INTO rightmove VALUES (?,?,?,?)""",
                      (_id, get_title(soup), get_price(soup), get_desc(soup)))
        except:
            print('Duplicate property')

        try:
            fp_link = soup.find('img', src=lambda x: x and "FLP" in x)['src']
            link_to_img(fp_link, _id, file_path_fp)
        except:
            print('No floor plan!')
        img_links = get_img_links(soup)
        for i, url in enumerate(img_links):
            link_to_img(url, _id, file_path_img, i)

        time.sleep(5)

    data_tuple = list(zip(keys, title, price, description))
    conn.commit()
    conn.close()

    return data_tuple

# return property image links on property page


def get_img_links(soup):
    links = []
    for tag in soup.find_all('meta', {'property': 'og:image'}):
        links.append(tag['content'])
    return links


# return property price on property page


def get_title(soup):
    return soup.find('title').text

# return property price on property page


def get_price(soup):
    try:
        text = soup.find(lambda tag: tag.name ==
                         'strong' and '£' in tag.text).text
        price = int(Decimal(sub(r'[^\d.]', '', text)))
        return price
    except:
        return 0

# return the full description on property page


def get_desc(soup):
    return soup.find('p', {'itemprop': 'description'}).text.strip()


url = "https://www.rightmove.co.uk/property-for-sale/find.html?locationIdentifier=REGION%5E87490&propertyTypes=&maxDaysSinceAdded=1&includeSSTC=false&mustHave=&dontShow=&furnishTypes=&keywords="
rm = RightmoveData(url)

property_links = list(rm.get_results.url)
ids = [re.findall(r'\d+', link)[0] for link in property_links]

# links, ids = get_links(soup)

get_prop_data(property_links, ids)
# get_img(property_links, ids)
