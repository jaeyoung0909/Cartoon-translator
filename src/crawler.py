import urllib
from urllib import request
from bs4 import BeautifulSoup
import os 

def url2img (filename, url):
    req = request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = request.urlopen(req).read()
    open(filename, 'wb').write(response)

def imgCrawler (url):
    req = request.Request(url)
    response = request.urlopen(req).read()
    soup = BeautifulSoup(response, 'html.parser')
    imgs = soup.select('#comic_view_area > div.wt_viewer > img')
    imgs = [img.attrs.get('src') for img in imgs]
    return imgs

def imgDownload (url, path):
    if not os.path.exists(path):
        os.makedirs(path)
    imgs = imgCrawler(url)
    for i, img in enumerate(imgs):
        url2img(path + '/{}.jpg'.format(i), img)

