from django.shortcuts import render

import os 
import base64

# Create your views here.

def post_list(request):
    url = 'comic.naver.com'
    url = os.path.join(url, request.get_full_path())
    filenames = sorted(os.listdir('imgs'))
    base64Imgs = []
    for filename in filenames:
        path = os.path.join('imgs', filename)
        f = open(path, 'rb')
        img = f.read()
        e = b"data:image/jpg;base64," + base64.b64encode(img)
        base64Imgs.append(e.decode('utf-8'))
        f.close()
    return render(request, 'post_list.html', {'imgs': base64Imgs})