from django.shortcuts import render

import os 
import base64
import numpy as np
import cv2
import io
from PIL import Image

import sys
sys.path.append('/'.join(os.getcwd().split('/')[:-1]))
from src.OCR.detector import detect_text_from_byte
from src.MT.interactive import translator 
from src.PargraphGenerating.insert_text import remover, inpainting
from src.PargraphGenerating.combine_images import PreProcessByteImages
from src.crawler import byteImgDownload
# Create your views here.

def PIL_image_to_byte_array(image:Image):
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, 'BMP')
    imgByteArr = imgByteArr.getvalue()
    e = b"data:image/jpg;base64," + base64.b64encode(imgByteArr)
    return e.decode('utf-8')


def post_list(request):
    url = 'https://comic.naver.com'
    url = url + request.get_full_path()

    print(url)
    
    cwd = os.getcwd()
    ckpt_path = os.path.join(cwd, '../src', 'MT/ckpt/checkpoint77.pt')
    vocab_path = os.path.join(cwd, '../src', 'MT/wiki.ko.model')
    data_path = os.path.join(cwd, '../src', 'MT/')
    font_path = os.path.join(cwd,'../src', 'PargraphGenerating/font/KOMIKJI_.ttf')
    try:
        byteImgs = byteImgDownload(url)
    except:
        Exception("wrong URL format")
    print("Crawling complete!")

    id = int(url.split('=')[1].split('&')[0])
    
    json_files = [detect_text_from_byte(img) for img in byteImgs]
    print("OCR complete")

    images = [cv2.imdecode(np.frombuffer(img, dtype=np.uint8), -1) for img in byteImgs]
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]

    Contents, Paragraphs = remover(images, json_files, discriminative_power = 0.55, autotune=True)
    print("text removing complete!")
    translated = translator(Paragraphs, ckpt_path, vocab_path, data_path)
    print("translation complete!")

    images = inpainting(Contents, translated, font_path)
    print("text inserting complete!")

    base64Imgs = [PIL_image_to_byte_array(img.original_image) for img in images]

    return render(request, 'post_list.html', {'imgs': base64Imgs})