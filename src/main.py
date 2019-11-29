from OCR.detector import detect_text_from_byte
from MT.interactive import translator 
from PargraphGenerating.insert_text import post_process, Post, TextBox
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
    for i in range(len(images)):
        dst = str(i)+'.jpg'
        Post_object = Post(images[i], json_files[i])
        print("Remove text!")
        box = Post_object.get_jsonbox()
        if box is None:
            Post_object.original_image.save(dst)
            continue
        textbox = Post_object.get_textbox()
        text = Post_object.get_text(textbox)
        Post_object.erase_text(box,(255,0,0),True)  
        translated = translator(text, ckpt_path, vocab_path, data_path)
        print(text)
        print(translated)
        for i in range(len(textbox)):
            input_sentence = translated[i]
            textbox_Position = (textbox[i][3], textbox[i][1])
            textbox_Size = (textbox[i][4] - textbox[i][3], textbox[i][2] - textbox[i][1])
            default_font_size = textbox[i][2] - textbox[i][1]
            r,g,b = Post_object.get_background_color(textbox[i])
            color = (abs(255-r),abs(255-g), abs(255-b))
            textBox01 = TextBox(Post_object.original_image, input_sentence, textbox_Position, textbox_Size, default_font_size)
            textBox01.generateText(color)
        Post_object.original_image.save(dst)
    


    


    

    # post_process(src_path, dest_path)
    # translated = translator(['흑흑 기수형 너무좋아], ckpt_path, vocab_path, data_path)
    
ex_url = 'https://comic.naver.com/webtoon/detail.nhn?titleId=651673&no=435&weekday=sat'
main(ex_url)