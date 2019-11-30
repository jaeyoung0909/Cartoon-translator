from OCR.detector import detect_text_from_byte
from MT.interactive import translator 
from PargraphGenerating.insert_text import remover, inpainting
from PargraphGenerating.combine_images import PreProcessByteImages
from crawler import byteImgDownload
import os 
import numpy as np
import cv2

def main(url):
    cwd = os.getcwd()
    ckpt_path = os.path.join(cwd, 'MT/ckpt/checkpoint77.pt')
    vocab_path = os.path.join(cwd, 'MT/wiki.ko.model')
    data_path = os.path.join(cwd, 'MT/')
    font_path = os.path.join(cwd, 'PargraphGenerating/font/KOMIKHI_.ttf')
    
    byteImgs = byteImgDownload(url)
    print("Crawling complete!")
    combinedByteImg = byteImgs
    json_files = [detect_text_from_byte(img) for img in byteImgs]
    print("OCR complete")
    images = [cv2.imdecode(np.frombuffer(img, dtype=np.uint8), -1) for img in byteImgs]
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]

    ### remove text and insert translated text
    Contents, Paragraphs = remover(images, json_files, discriminative_power = 0.55, autotune=True)
    # remove text
    translated = translator(Paragraphs, ckpt_path, vocab_path, data_path)
    # Generate text inserted image in list
    
    images = inpainting(Contents, translated, font_path)
    for i in images:
        plt.imshow(i.original_image)
        plt.show()


if __name__ == "__main__":
    ex_url = 'https://comic.naver.com/webtoon/detail.nhn?titleId=651673&no=435&weekday=sat'
    main(ex_url)