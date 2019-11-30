from OCR.detector import detect_text_from_byte
from MT.interactive import translator 
from PargraphGenerating.insert_text import post_process, Post, TextBox, remover, inpainting
from PargraphGenerating.combine_images import PreProcessByteImages
from crawler import byteImgDownload
import shutil
import matplotlib.pyplot as plt
import os 
import sys
import numpy as np
import cv2

def main(url):
    cwd = os.getcwd()
    ckpt_path = os.path.join(cwd, 'MT/ckpt/checkpoint77.pt')
    vocab_path = os.path.join(cwd, 'MT/wiki.ko.model')
    data_path = os.path.join(cwd, 'MT/')
    cache_path = os.path.join(cwd, 'cache')
    img_path = os.path.join(cache_path, 'ori_img')
    src_path = os.path.join(cwd, '00.jpg')
    dest_path = os.path.join(cwd, 'ex.jpg')

    
    byteImgs = byteImgDownload(url)
    print("Crawling complete!")
    combinedByteImg = byteImgs
    json_files = [detect_text_from_byte(img) for img in byteImgs]
    print("OCR complete")
    images = [cv2.imdecode(np.frombuffer(img, dtype=np.uint8), -1) for img in byteImgs]
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
    plt.imshow(images[3])
    plt.show()

    ### remove text and insert translated text
    Contents, Paragraphs = remover(images, json_files, autotune=True)
    # remove text
    translated = translator(Paragraphs, ckpt_path, vocab_path, data_path)
    # Generate text inserted image in list
    Outputs = inpainting(Contents, translated)


    


    


    

    # post_process(src_path, dest_path)
    # translated = translator(['흑흑 기수형 너무좋아], ckpt_path, vocab_path, data_path)
    
ex_url = 'https://comic.naver.com/webtoon/detail.nhn?titleId=651673&no=435&weekday=sat'
main(ex_url)