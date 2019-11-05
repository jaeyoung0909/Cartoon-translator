import urllib
from urllib import request

url = "https://image-comic.pstatic.net/webtoon/20853/1199/20191104171251_d4dd2673b862d1b2d34ef9059d368d3d_IMAG01_3.jpg"
# header = {'User=Agent':'Mozilla/5.0', 'referer':'https://comic.naver.com/webtoon/detail.nhn?titleId=20853&no=1199&weekday=tue'}
# req = request.Request(url, headers=header)
# data = request.urlopen(req).read()
req = request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
# req = request.Request(url)

response = request.urlopen(req).read()
f = open('ex.jpg', 'wb').write(response)

# request.urlretrieve(url, 'ex.jpg') 