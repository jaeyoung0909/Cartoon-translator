from PIL import Image, ImageDraw, ImageFont
import json
import os
import matplotlib.pyplot as plt
import copy
import cv2
import numpy as np
import textwrap
import math

class post:

    def __init__(self, name):
        self.name = name
        image_path = os.path.join('..','ex_img', name+'.jpg')
        print(image_path)
        if os.path.isfile(image_path):
            self.original_image = Image.open(image_path)
        else:
            raise Exception('There is no "{0}" image file'.format(name))
        self.json_data = self.treat_json_file()

    def treat_json_file(self):
        json_path = os.path.join('..','ex_json', self.name + '.json')
        replace_json_path = os.path.join('..','ex_json', 'remove_breakline', self.name + '.json')
        if os.path.isfile(json_path):
            replace_json_dir = os.path.join('..','ex_json', 'remove_breakline')
            if not os.path.exists(replace_json_dir):
                os.mkdir(replace_json_dir)

            f = open(json_path,'r')
            f_replace = open(replace_json_path,'w',encoding='utf-8')
            lines = f.readlines()

            for line in lines:
                f_replace.write(line.replace('\\n', ' ').replace('\\','').strip('"'))

            f.close()
            f_replace.close()

            with open(replace_json_path) as json_file:
                json_data = json.load(json_file)
        else:
            raise Exception('There is no "{0}" json file'.format(self.name))

        return json_data

    def erase_text(self,boxes, color):
        img_draw = ImageDraw.Draw(self.original_image)
        for box in boxes:
            img_draw.rectangle([(box[2],box[0]),(box[3],box[1])],fill = color)

        del img_draw

    def detect_overlap(self, margin, threshold):
        box = []
        box_coor = [-1, -1, -1, -1]
        for i in range(len(self.json_data['textAnnotations'])):
            if i == 0:
                continue
            coor_data = self.json_data['textAnnotations'][i]['boundingPoly']['vertices']
            coor = [coor_data[0]['y'], coor_data[2]['y'], coor_data[0]['x'], coor_data[2]['x']]
            if (box_coor[0] == -1):
                box_coor = copy.deepcopy(coor)
            elif (box_coor[1] - margin <= coor[0] + margin or box_coor[0] + margin >= coor[1] - margin or abs(
                    box_coor[3] - coor[2]) > threshold):
                box.append(box_coor)
                box_coor = copy.deepcopy(coor)
            else:
                box_coor[0] = min(box_coor[0], coor[0])
                box_coor[1] = max(box_coor[1], coor[1])
                box_coor[2] = min(box_coor[2], box_coor[3], coor[2], coor[3])
                box_coor[3] = max(box_coor[2], box_coor[3], coor[2], coor[3])
        box.append(box_coor)
        return box

    def detect_box(self, margin, threshold, overlap_boxes):
        t = 10
        img_size = self.original_image.size
        print(img_size)
        boxes = []
        temp_image = copy.deepcopy(self.original_image)
        img_draw = ImageDraw.Draw(temp_image)
        for box in overlap_boxes:
            img_draw.rectangle([(box[2], box[0]), (box[3], box[1])], fill= 'white')

        gray_image = np.array(temp_image.convert('L'))
        ret, dst = cv2.threshold(gray_image, 40, 255, cv2.THRESH_BINARY)

        for box in overlap_boxes:
            x_left = 0
            x_right = 0
            y_top = box[0]
            y_bot = box[1]
            color = 255
            for length in range(box[2]):

                color_bot = dst[y_bot,box[2] - length]
                color_top = dst[y_top,box[2] - length]
                if(self.color_diff(color,color_bot) > t or self.color_diff(color,color_top) > t):
                    x_left = box[2] - length
                    break
            for length in range(int(img_size[0]-box[3])):
                color_bot = dst[y_bot,box[3] + length]
                color_top = dst[y_top,box[3] + length]
                if(self.color_diff(color,color_bot) > t or self.color_diff(color,color_top) > t):
                    x_right = box[3]+length
                    break
            boxes.append([box[0],box[1],x_left+5,x_right-5])
        del img_draw
        del temp_image
        return boxes

    def merge_box(self, boxes,margin):
        refer_box = [-1,-1,-1,-1]
        merge = []
        merge_list = []
        for i, box in enumerate(boxes):
            if (refer_box[0] == -1):
                refer_box = copy.deepcopy(box)
                merge_list.append(i);
            elif (refer_box[1] + margin <= box[0] - margin or refer_box[0] - margin >= box[1] + margin):
                merge.append(copy.deepcopy(merge_list))
                refer_box = copy.deepcopy(box)
                del merge_list
                merge_list = [i]

            elif (refer_box[2] > box[3] or refer_box[3] < box[2]):
                merge.append(copy.deepcopy(merge_list))
                refer_box = copy.deepcopy(box)
                del merge_list
                merge_list = [i]

            else:
                refer_box[0] = min(refer_box[0], box[0])
                refer_box[1] = max(refer_box[1], box[1])
                refer_box[2] = min(refer_box[2], refer_box[3], box[2], box[3])
                refer_box[3] = max(refer_box[2], refer_box[3], box[2], box[3])
                merge_list.append(i)
        merge.append(merge_list)
        return merge

    def get_textbox(self,boxes,merge,img):
        img_size = img.size
        temp_image = copy.deepcopy(img)
        gray_image = np.array(temp_image.convert('L'))
        ret, dst = cv2.threshold(gray_image, 40, 255, cv2.THRESH_BINARY)
        color = 255
        textbox = []
        for merge_list in merge:
            max = [0,0,0,0,0]
            for elem in merge_list:
                y_top = 0
                y_bot = 0
                x_left = boxes[elem][2]
                x_right = boxes[elem][3]
                for length in range(boxes[elem][0]):
                    color_left = dst[boxes[elem][0]-length, x_left]
                    color_right = dst[boxes[elem][0] - length, x_right]
                    if(abs(color_left-color) > 10 or abs(color_right-color)):
                        y_top = boxes[elem][0]-length
                        break
                for length in range(img_size[1]-boxes[elem][1]):
                    color_left = dst[boxes[elem][1] + length, x_left]
                    color_right = dst[boxes[elem][1] + length, x_right]
                    if (abs(color_left - color) > 10 or abs(color_right - color)):
                        y_bot = boxes[elem][1] + length
                        break
                if(max[1]-max[0])*(max[3]-max[2]) < (y_bot-y_top)*(x_right-x_left):
                    max[0] = y_top
                    max[1] = y_bot
                    max[2] = x_left
                    max[3] = x_right
                    max[4] = abs(boxes[elem][1]-boxes[elem][0])
            textbox.append(max)
        return textbox


    def color_diff(self, color1, color2):
        return abs(color2-color1)

class TextBox:
    def __init__(self,image,msg,box_position,box_size,default_font_size, font_style = 'arial.ttf'):
        self.im = image
        self.draw = ImageDraw.Draw(image)
        self.msg = msg
        self.box_X = box_position[0]
        self.box_Y = box_position[1]
        self.box_W = box_size[0]
        self.box_H = box_size[1]
        self.default_font_size = default_font_size
        self.font_size = default_font_size
        self.font_style = font_style
        self.font = ImageFont.truetype(font = font_style, size = self.font_size)
        self.textWidth = 0
        self.message_paragraph = ""
        self.message_paragraph_W = 0;
        self.message_paragraph_H = 0;

    def textWidthHeightCalculator(self):
        # Average length of English word is 4.7
        letter_w01, letter_h01 = self.draw.textsize("Aaaaa", font=self.font)
        letter_w, letter_h = self.draw.textsize("Aaaaa\nAaaaa", font=self.font)
        self.textWidth = math.floor(self.box_W / (letter_w / 5))
        # print(letter_h - letter_h01)

    def generateText(self):
        self.font_size = self.default_font_size
        self.font = ImageFont.truetype(font=self.font_style, size=self.font_size)
        self.generateParagraph()

        while self.message_paragraph_H > self.box_H:
            # To be more accuracy
            self.font_size= math.floor(self.font_size * math.sqrt(self.box_H /self.message_paragraph_H))

            # To be safe
            # self.font_size = math.floor(self.font_size * self.box_H / self.message_paragraph_H)
            self.font = ImageFont.truetype(font=self.font_style, size=self.font_size)
            self.textWidthHeightCalculator()
            self.generateParagraph()
        self.draw.text((self.box_X + (self.box_W -self.message_paragraph_W) / 2, self.box_Y + (self.box_H - self.message_paragraph_H) / 2), self.message_paragraph, fill="black", font=self.font)



    def generateParagraph(self):
        self.textWidthHeightCalculator()
        wrapper = textwrap.TextWrapper(width=self.textWidth)
        self.message_paragraph = "\n".join(wrapper.wrap(text=self.msg))
        self.message_paragraph_W, self.message_paragraph_H = self.draw.textsize(self.message_paragraph, font=self.font)

    def whiteBox(self):
        self.draw.rectangle(((self.box_X, self.box_Y), (self.box_X+self.box_W, self.box_Y+self.box_H)), fill="white")

    def imageShow(self):
        self.im.show()
    def save(self):
        self.im.save("result.png")


    def set_textbox(self,box_position,box_size):
        self.box_X = box_position[0]
        self.box_Y = box_position[1]
        self.box_W = box_size[0]
        self.box_H = box_size[1]

def get_color(img ,mask):
    chans = cv2.split(np.array(img))
    colors = ("r","g","b")

    plt.figure()
    plt.title("Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixcels")
    features = []

    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
        features.extend(hist)

        plt.plot(hist, color=color)
        plt.xlim([0, 256])

    print
    "Flattened feature vector size: %d " % (np.array(features).flatten().shape)
    plt.show()

def main():

    a=post(str(5))

    overlap_box = a.detect_overlap(3,10)
    detected_box = a.detect_box(3,10,overlap_box)
    a.erase_text(detected_box,(255,255,150))
    merge_box = a.merge_box(detected_box,7)
    textbox = a.get_textbox(detected_box, merge_box, a.original_image)
    input_sentence = """My name is Hana."""

    im = a.original_image

    for i in range(len(textbox)):
        if(textbox[i][4] == 0):
            continue
        textbox_Position = (textbox[i][2], textbox[i][0])
        textbox_Size = (textbox[i][3]-textbox[i][2],textbox[i][1]-textbox[i][0])
        default_font_size = textbox[i][4]
        textBox01 = TextBox(im, input_sentence, textbox_Position, textbox_Size, default_font_size, font_style='arial.ttf')
        textBox01.generateText()

    imgplot = plt.imshow(a.original_image)
    plt.show()
    a.original_image.save(os.path.join('..','ex_result', a.name+'.jpg'))

main()
