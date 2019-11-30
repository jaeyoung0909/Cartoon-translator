import cv2
import numpy as np
import glob, os


class PreProcessImages:
    def __init__(self,image_directory, dest):
        self.url = image_directory
        self.dest = dest


    def saveCombinedImage(self):
        entries = glob.glob( self.url)
        entries.sort()
        imgs = []
        for file in entries:
            imgs.append(cv2.imread(file))
        stacked_img = np.concatenate(tuple(imgs), axis=0)
        path = os.path.join(self.dest, 'combined_img.jpg')
        if not os.path.exists(self.dest):
            raise Exception("combine_images: Cache not found")
        cv2.imwrite(path, stacked_img)
        print(os.path.join(path, 'combined_img.jpg'))
        # del imgs

class PreProcessByteImages:
    def __init__(self, imgs):
        self.imgs = imgs


    def combineImage(self):
        imgs = [cv2.imdecode(np.frombuffer(img, dtype=np.uint8), -1) for img in self.imgs] 
        stacked_img = np.concatenate(tuple(imgs), axis=0)
        success, encoded_image = cv2.imencode('.jpg', stacked_img)
        return encoded_image.tobytes()


