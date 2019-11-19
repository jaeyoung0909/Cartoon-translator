from PIL import Image, ImageDraw, ImageFont
import textwrap
import math

input_sentence = """This function wraps the input paragraph such that each line in the paragraph is at most width characters long."""

canvas_Size= (500,500)
textbox_Position = (200,100)
textbox_Size = (200,100)

im = Image.new("RGBA",(canvas_Size[0],canvas_Size[1]),"yellow")

default_font_size = 20

class TextBox:
    def __init__(self,image,msg,box_position,box_size,default_font_size, font_style = 'arial.ttf'):
        self.im = image
        self.draw = ImageDraw.Draw(im)
        self.msg = msg
        self.box_X = box_position[0]
        self.box_Y = box_position[1]
        self.box_W = box_size[0]
        self.box_H = box_size[1]
        self.default_font_size = default_font_size;
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
            print("?")
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


textBox01 = TextBox(im,input_sentence,textbox_Position,textbox_Size,default_font_size, font_style = 'arial.ttf')
textBox01.whiteBox()
textBox01.generateText()

textBox01.set_textbox((300,250),(100,100))

textBox01.whiteBox()
textBox01.generateText()

textBox01.set_textbox((100,380),(300,100))

textBox01.whiteBox()
textBox01.generateText()

textBox01.imageShow()
textBox01.save()