from PIL import Image, ImageDraw, ImageFont
import json
import os
import matplotlib.pyplot as plt
import copy
import cv2
import numpy as np
import textwrap
import math

class Post:

    # def __init__(self, name):
    #     self.name = name
    #     image_path = os.path.join('..','ex_img', name)
    #     print(image_path)
    #     if os.path.isfile(image_path):
    #         image = Image.open(image_path)
    #     else:
    #         raise Exception('There is no "{0}" image file'.format(name))
    #     img_array = np.array(image)
    #     dst = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
    #     self.original_image = Image.fromarray(dst,'RGB')
    #     self.json_data = self.treat_json_file()
    #     del image

    def __init__(self,image,json_file):
        self.name = 'a'
        img_array = np.array(image)
        dst = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
        self.original_image = Image.fromarray(dst,'RGB')
        self.json_data = json.loads(json_file)

    def set_discriminative_power(self, power):
        self. discriminative_power = power;

    def treat_json_file(self):
        json_name = self.name.split(".")[0] + '.json'
        json_path = os.path.join('..','ex_json', json_name)
        replace_json_path = os.path.join('..','ex_json', 'remove_breakline', json_name)
        if os.path.isfile(json_path):
            replace_json_dir = os.path.join('..','ex_json', 'remove_breakline')
            if not os.path.exists(replace_json_dir):
                os.mkdir(replace_json_dir)

            f = open(json_path,'r')
            f_replace = open(replace_json_path,'w')
            lines = f.readlines()

            for line in lines:
                l = line.replace(r'"\\\"','"').replace(r'\\"',"").replace("'",'')
                f_replace.write(l.replace('\\n', ' ').replace(r'\\','%').replace('\\','').strip('"'))

            f.close()
            f_replace.close()

            with open(replace_json_path) as json_file:
                json_data = json.load(json_file)

        else:
            raise Exception('There is no "{0}" json file'.format(self.name))

        return json_data

    def erase_text(self, boxes, color=(255,255,255), boundary = False, img = None, index = None):
        if boxes is None:
            return
        if(img is None):
            img_draw = ImageDraw.Draw(self.original_image)
        else:
            img_draw = ImageDraw.Draw(img)
        if index is None:
            if color is None:
                if not boundary:
                    for box in boxes:
                        color = self.get_background_color(box)
                        img_draw.rectangle([(box[3],box[1]),(box[4],box[2])],fill = color)
                else:
                    for box in boxes:
                        color = self.get_background_color(box)
                        img_draw.rectangle([(box[3],box[1]),(box[4],box[2])],outline=color)
                del img_draw
            else:
                if not boundary:
                    for box in boxes:

                        img_draw.rectangle([(box[3],box[1]),(box[4],box[2])],fill = color)
                else:
                    for box in boxes:

                        img_draw.rectangle([(box[3],box[1]),(box[4],box[2])],outline=color)
                del img_draw
        else:
            if color is None:
                if not boundary:
                    for i in index:
                        box = boxes[i]
                        color = self.get_background_color(box)
                        img_draw.rectangle([(box[3], box[1]), (box[4], box[2])], fill=color)
                else:
                    for i in index:
                        box = boxes[i]
                        color = self.get_background_color(box)
                        img_draw.rectangle([(box[3], box[1]), (box[4], box[2])], outline=color)
                del img_draw
            else:
                if not boundary:
                    for i in index:
                        box = boxes[i]
                        img_draw.rectangle([(box[3], box[1]), (box[4], box[2])], fill=color)
                else:
                    for i in index:
                        box = boxes[i]
                        img_draw.rectangle([(box[3], box[1]), (box[4], box[2])], outline=color)
                del img_draw

    def erase_text_candi(self,boxes, color=(255,255,255), boundary = False):
        img_draw = ImageDraw.Draw(self.original_image)
        if not boundary:
            for box in boxes:
                img_draw.rectangle([(box[3],box[1]),(box[4],box[2])],fill = color)
        else:
            for box in boxes:
                if(self.discriminate_textbox(box)):
                    img_draw.rectangle([(box[3],box[1]),(box[4],box[2])],outline=color)
        del img_draw

    def get_jsonbox(self):
        jsonbox = []
        if not 'textAnnotations'in self.json_data.keys():
            return
        data = self.json_data['textAnnotations']
        for i in range(len(data)):
            if i == 0:
                continue
            try:
                coor_data = data[i]['boundingPoly']['vertices']
                box = [i, coor_data[0]['y'], coor_data[2]['y'], coor_data[0]['x'], coor_data[2]['x']]
                jsonbox.append(box)
            except:
                print("json file has wrong format")
        return jsonbox

    def detect_overlap(self, boxes, overlapping=0.8, distance=15, discriminate = False,
                       x_direction = True, get = False, autotune = False):
        
        overlap_boxes = []
        gather = []
        gatherer = []
        overlap_box = [-1,-1,-1,-1,-1]
        height = 0
        overlap_elem = []
        if discriminate:
            discri = self.discriminate_textboxes(self.get_jsonbox())
        for i, box in enumerate(boxes):
            if(overlap_box[4] == -1):
                overlap_box = copy.deepcopy(box)
                gatherer.append(i)
                height = overlap_box[2] - overlap_box[1]
                if isinstance(box[0],list):
                    overlap_elem.extend(box[0])
                else:
                    overlap_elem.append(box[0])
                continue
            if(autotune):
                if x_direction:
                    distance = height/3
                else:
                    distance = height/5
            if(self.overlap(overlap_box, box,overlapping,distance, x_direction)):
                overlap_box[1] = min(box[1],overlap_box[1])
                overlap_box[2] = max(box[2],overlap_box[2])
                overlap_box[3] = min(box[3],overlap_box[3])
                overlap_box[4] = max(box[4],overlap_box[4])
                height = max(height, overlap_box[2]-overlap_box[1])
                gatherer.append(i)
                if isinstance(box[0], list):
                    overlap_elem.extend(box[0])
                else:
                    overlap_elem.append(box[0])
            else:
                if not discriminate:
                    overlap_box[0] = overlap_elem
                    overlap_box.append(height)
                    overlap_boxes.append(overlap_box)
                    gather.extend(gatherer)

                else:
                    count = 0
                    for elem in overlap_elem:
                        if elem in discri:
                            count = count+1
                        else:
                            count = count-1
                    if(count > 0):
                        overlap_box[0] = overlap_elem
                        overlap_box.append(height)
                        overlap_boxes.append(overlap_box)
                        gather.extend(gatherer)

                del overlap_elem
                del gatherer
                overlap_box = copy.deepcopy(box)
                overlap_elem = []
                gatherer = []
                gatherer.append(i)
                height = overlap_box[2]-overlap_box[1]
                if isinstance(box[0], list):
                    overlap_elem.extend(box[0])
                else:
                    overlap_elem.append(box[0])
        count = 0
        if discriminate:
            for elem in overlap_elem:
                if elem in discri:
                    count = count+1
                else:
                    count = count-1
            if(count > 0):
                overlap_box[0] = overlap_elem
                overlap_box.append(height)
                overlap_boxes.append(overlap_box)
                gather.extend(gatherer)

        else:
            overlap_box[0] = overlap_elem
            overlap_box.append(height)
            overlap_boxes.append(overlap_box)
            gather.extend(gatherer)

        if get:
            return overlap_boxes, gather
        return overlap_boxes

    def overlap(self,box1,box2,p,d, x_direction = True):
        if x_direction:
            y1_top = box1[2]
            y1_bot = box1[1]
            y2_top = box2[2]
            y2_bot = box2[1]
            if(y1_top < y2_bot or y1_bot > y2_top):
                return False;
            x1_left = box1[3]
            x1_right = box1[4]
            x2_left = box2[3]
            x2_right = box2[4]
            if(x1_right < x2_left):
                if(x2_left-x1_right > d):
                    return False
            if(x1_left > x2_right):
                if(x1_left-x2_right):
                    return False
            if(y1_top < y2_top):
                y_top = y1_top
            else:
                y_top = y2_top
            if(y1_bot > y2_bot):
                y_bot = y1_bot
            else:
                y_bot = y2_bot
            if(y1_top-y1_bot < y2_top-y2_bot):
                over = (y_top-y_bot) / (y1_top-y1_bot)
            else:
                over = (y_top-y_bot) / (y2_top-y2_bot)
            if(over < p):
                return False
            return True
        else:
            x1_left = box1[3]
            x1_right = box1[4]
            x2_left = box2[3]
            x2_right = box2[4]
            if (x1_left > x2_right or x1_right < x2_left):
                return False;
            y1_top = box1[2]
            y1_bot = box1[1]
            y2_top = box2[2]
            y2_bot = box2[1]
            if (y1_top < y2_bot):
                if (y2_bot - y1_top > d):
                    return False
            if (y1_bot > y2_top):
                if (y1_bot - y2_top > d):
                    return False
            if (x1_left < x2_left):
                x_left = x2_left
            else:
                x_left = x1_left
            if (x1_right > x2_right):
                x_right = x2_right
            else:
                x_right = x1_right
            if (x1_right - x1_left < x2_right - x2_left):
                over = (x_right - x_left) / (x1_right - x1_left)
            else:
                over = (x_right - x_left) / (x2_right - x2_left)
            if (over < p):
                return False
            return True

    def discriminate_textbox(self, box):
        img = self.original_image
        chans = cv2.split(np.array(img))
        colors = ("r", "g", "b")
        mask = np.zeros(np.array(img).shape[:2], np.uint8)
        mask[box[1]:box[2], box[3]:box[4]] = 255

        features = []
        p = 0.25
        over = 0
        threshold = self.discriminative_power
        a=0
        for (chan, color) in zip(chans, colors):
            hist = cv2.calcHist([chan], [0], mask, [32], [0, 256])
            features.append(hist)
            over = over + np.where(hist >= threshold)[0].shape[0]
            sort_hist = np.sort(hist,axis = None)
            a = a + sort_hist[-1]+sort_hist[-2]+sort_hist[-3]
        b=(box[2]-box[1]) *(box[4]-box[3]) * 3
        if(b<=0):
            return False
        if(a/b>=threshold):
            return True
        else:
            return False

    def discriminate_textboxes(self,boxes):
        result = []
        for box in boxes:
            if(self.discriminate_textbox(box)):
                if isinstance(box[0],list):
                    result.extend(box[0])
                else:
                    result.append(box[0])
        return result

    def check_size(self,boxes):
        size = 0
        result = []
        p = 0.3
        for box in boxes:
            size = size + box[2] - box[1]
        size = size / len(boxes)
        for box in boxes:
            if box[2]-box[1] > size * p:
                result.append(box)
        return result

    def detect_box(self, overlap_boxes):
        t = 5
        img_size = self.original_image.size
        boxes = []
        temp_image = copy.deepcopy(self.original_image)
        img_draw = ImageDraw.Draw(temp_image)

        for box in overlap_boxes:
            color = self.get_background_color(box)
            img_draw.rectangle([(box[3], box[1]), (box[4], box[2])], fill= color)

        gray_image = np.array(temp_image.convert('L'))
        ret, dst = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
        for box in overlap_boxes:
            color = 255
            x_left = -1
            x_right = -1
            y_top = box[1]
            y_bot = box[2]

            for length in range(box[3]):
                color_bot = dst[y_bot,box[3] - length]
                color_top = dst[y_top,box[3] - length]
                if(self.color_diff(color,color_bot) > t or self.color_diff(color,color_top) > t):
                    x_left = box[3] - length
                    break

            for length in range(int(img_size[0]-box[4])):
                color_bot = dst[y_bot,box[4] + length]
                color_top = dst[y_top,box[4] + length]
                if(self.color_diff(color,color_bot) > t or self.color_diff(color,color_top) > t):
                    x_right = box[4]+length
                    break
            if x_left == -1:
                x_left = box[3]
            if x_right == -1:
                x_right = box[4]
            boxes.append([box[0],box[1],box[2],x_left+5,x_right-5])
        del img_draw
        del temp_image
        return boxes

    def color_diff(self, color1, color2):
        return abs(color2-color1)

    def get_background_color(self, box):
        img = self.original_image
        chans = cv2.split(np.array(img))
        colors = ("r", "g", "b")
        mask = np.zeros(np.array(img).shape[:2], np.uint8)
        mask[box[1]:box[2], box[3]:box[4]] = 255;
        Color = []
        features = []

        for (chan, color) in zip(chans, colors):
            hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
            features.append(hist)
            Color.append(np.argmax(hist))

        return tuple(Color)

    def get_largest_box(self, boxes, merge):

        img_size = self.original_image.size
        temp_image = copy.deepcopy(self.original_image)
        self.erase_text(boxes, None, False, temp_image)
        img_draw = ImageDraw.Draw(self.original_image)
        gray_image = np.array(temp_image.convert('L'))
        ret, dst = cv2.threshold(gray_image, 40, 255, cv2.THRESH_BINARY)
        color = 255
        textbox = []

        for merge_list in merge:
            max = [0, 0, 0, 0, 0, 0]
            for box in boxes:
                if box[0][0] in merge_list:
                    y_top = 0
                    y_bot = 0
                    x_left = box[3]
                    x_right = box[4]
                    for length in range(box[1]):
                        color_left = dst[box[1] - length, x_left]
                        color_right = dst[box[1] - length, x_right]
                        if (abs(color_left - color) > 10 or abs(color_right - color)):
                            y_top = box[1] - length
                            break
                    for length in range(img_size[1] - box[2]):
                        color_left = dst[box[2] + length, x_left]
                        color_right = dst[box[2] + length, x_right]
                        if (abs(color_left - color) > 10 or abs(color_right - color)):
                            y_bot = box[2] + length
                            break
                    if (max[2] - max[1]) * (max[4] - max[3]) < (y_bot - y_top) * (x_right - x_left):
                        max[1] = y_top
                        max[2] = y_bot
                        max[3] = x_left
                        max[4] = x_right
                        max[5] = abs(box[2] - box[1])

            max[0] = merge_list
            textbox.append(max)

        return textbox

    def check_chunk(self, boxes):
        result = []
        for box in boxes:
            if isinstance(box, list) and len(box[0]) > 1:
                result.append(box)
        return result

    def get_textbox2(self):
        box = self.get_jsonbox()
        sized_box = self.check_size(box)
        xoverlap_box = self.detect_overlap(sized_box, discriminate = False)
        xdetect_box = self.detect_box(xoverlap_box)
        xxoverlap_box = self.detect_overlap(xdetect_box,discriminate = False)
        overlap_box = self.detect_overlap(xxoverlap_box,0.5,discriminate=True, x_direction = False)
        merge = []
        for i in range(len(overlap_box)):
            merge.append(overlap_box[i][0])
        textbox = self.get_largest_box(xxoverlap_box, merge)
        self.erase_text(overlap_box,(0,0,0),False)

    def get_textbox(self, x_overlapping, y_overlapping, x_distance, y_distance, discriminative_power, autotune=False):
        self.discriminative_power = discriminative_power
        box = self.get_jsonbox()
        sized_box = self.check_size(box)
        xoverlap_box = self.detect_overlap(sized_box, x_overlapping, x_distance ,discriminate=False)
        xdetect_box = self.detect_box(xoverlap_box)
        xxoverlap_box = self.detect_overlap(xdetect_box, 0.8, 0, discriminate=False)
        overlap_box, gather = self.detect_overlap(xxoverlap_box, y_overlapping, y_distance ,discriminate=True, x_direction=False, get = True)
        xcoverlap_box = self.check_chunk(xxoverlap_box)
        self.erase_text(xxoverlap_box, None, False, None, gather)
        return overlap_box

    def get_text(self, boxes):
        text = []
        data = self.json_data['textAnnotations']
        for box in boxes:
            s = ""
            if isinstance(box[0],list):
                for elem in box[0]:
                    ko = data[elem]['description']
                    ko = ko.replace('%', '\\').encode("UTF-8").decode('unicode_escape')
                    #ko.encode('UTF-8')
                    #ko = ko.encode("UTF-8").decode('unicode_escape')
                    s = s + ko + " "
            else:
                s = s + data[box[0]]['description'].encode('UTF-8')
            text.append(s)
        return text

    def setbox(self,box):
        self.box = box

class TextBox:
    def __init__(self,image,msg,box_position,box_size,default_font_size, font_style = 'FreeSans.ttf'):
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

    def generateText(self, color):
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
        self.draw.text((self.box_X + (self.box_W -self.message_paragraph_W) / 2, self.box_Y + (self.box_H - self.message_paragraph_H) / 2), self.message_paragraph, fill=color, font=self.font)



    def generateParagraph(self):
        self.textWidthHeightCalculator()
        if self.textWidth <= 0:
            return
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

def remover(images, json_files, x_overlapping = 0.8, y_overlapping = 0.1 ,
            x_distance = 30, y_distance = 10, discriminative_power = 0.7 , autotune = True):

    Contents = []
    Paragraphs = []
    print("Remove text!")
    for i in range(len(images)):
        dst = str(i)+'.jpg'
        Post_object = Post(images[i], json_files[i])

        box = Post_object.get_jsonbox()
        if box is None:
            Post_object.original_image.save(dst)
            continue
        textbox = Post_object.get_textbox(x_overlapping, y_overlapping ,
            x_distance, y_distance, discriminative_power ,autotune)
        Post_object.setbox(textbox)
        text = Post_object.get_text(textbox)
        #Post_object.erase_text(box,(255,0,0),True)
        Contents.append(Post_object)

        if isinstance(text, list):
            Paragraphs.extend(text)
        else:
            Paragraphs.append(text)
        Post_object.original_image.save(dst)
        del Post_object
    return Contents, Paragraphs


def inpainting(Contents, Translated):
    print("Inesert text!")
    index = 0
    for content in Contents:
        for textbox in content.box:
            input_sentence = Translated[index]
            textbox_Position = (textbox[3], textbox[1])
            textbox_Size = (textbox[4] - textbox[3], textbox[2] - textbox[1])
            default_font_size = textbox[5]
            r, g, b = content.get_background_color(textbox)
            color = (abs(255 - r) + abs(255 - g) + abs(255 - b))/3
            if(color < 150):
                Color = (0,0,0)
            else:
                Color = (255,255,255)
            textBox01 = TextBox(content.original_image, input_sentence, textbox_Position, textbox_Size,
                                default_font_size)
            textBox01.generateText(Color)
            index = index + 1

    return Contents

def main2():
    images_name = ["../ex_img/00.jpg","../ex_img/01.jpg","../ex_img/02.jpg","../ex_img/03.jpg","../ex_img/04.jpg"]
    json_file_names = ["../ex_json/0.json","../ex_json/1.json","../ex_json/2.json","../ex_json/3.json","../ex_json/4.json"]
    replace_json_path = ["../ex_json/remove_breakline/0.json","../ex_json/remove_breakline/1.json"
                         ,"../ex_json/remove_breakline/2.json","../ex_json/remove_breakline/3.json","../ex_json/remove_breakline/4.json"]

    images = []
    json_files = []

    for i in range(len(images_name)):
        image = Image.open(images_name[i])
        if os.path.isfile(json_file_names[i]):
            replace_json_dir = os.path.join('..', 'ex_json', 'remove_breakline')
            if not os.path.exists(replace_json_dir):
                os.mkdir(replace_json_dir)

            f = open(json_file_names[i], 'r')
            f_replace = open(replace_json_path[i], 'w')
            lines = f.readlines()
            for line in lines:
                l = line.replace(r'"\\\"', '"').replace(r'\\"', "").replace("'", '')
                f_replace.write(l.replace('\\n', ' ').replace(r'\\', '%').replace('\\', '').strip('"'))

            f.close()
            f_replace.close()

            with open(replace_json_path[i]) as json_file:
                l = json_file.readline()
        images.append(image)
        json_files.append(l)

    contents, paragraphs = remover(images, json_files, y_overlapping= 0.3 ,discriminative_power=0.65, autotune=True)
    contents = inpainting(contents, paragraphs)
    for i, content in enumerate(contents):
        content.original_image.save(str(i)+'.jpg')

main2()