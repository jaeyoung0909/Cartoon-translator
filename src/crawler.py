import urllib
from urllib import request
from bs4 import BeautifulSoup
import os 

def url2img (filename, url):
    req = request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = request.urlopen(req).read()
    open(filename, 'wb').write(response)

def urls2byteImgs(urls):
    return [request.urlopen( request.Request(i, headers={'User-Agent': 'Mozilla/5.0'}) ).read() for i in urls]



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
        url2img(os.path.join(path , str(i).zfill(2)+'.jpg'), img)

def byteImgDownload (url):
    urls = imgCrawler(url)
    return urls2byteImgs(urls)


# imgDownload("https://comic.naver.com/webtoon/detail.nhn?titleId=568986&no=183&weekday=sat", "./ex_img" )
