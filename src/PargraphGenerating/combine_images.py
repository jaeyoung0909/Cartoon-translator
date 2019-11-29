import cv2
import numpy as np
import glob, os

image_directory = '../ex_img/*.jpg'

class PreProcessImages:
    def __init__(self,image_directory):
        self.url = image_directory


    def saveCombinedImage(self):
        entries = glob.glob( self.url)
        entries.sort()
        imgs = []
        for file in entries:
            imgs.append(cv2.imread(file))
        stacked_img = np.concatenate(tuple(imgs), axis=0)
        path = os.path.join('..', 'ex_img_combined')
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(os.path.join(path, 'combined_img.jpg'), stacked_img)
        print(os.path.join(path, 'combined_img.jpg'))
        # del imgs


combinedImage = PreProcessImages(image_directory)
combinedImage.saveCombinedImage()
