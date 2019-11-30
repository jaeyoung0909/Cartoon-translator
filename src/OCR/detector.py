import io
import json
from google.protobuf.json_format import MessageToJson


def detect_text(path):
    """Detects text in the file."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = MessageToJson(response)
    # texts = texts.text_annotations
    return texts
    # text in texts has attributes named description and bounding_poly

def detect_text_from_byte(img):
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    image = vision.types.Image(content=img)

    response = client.text_detection(image=image)
    texts = MessageToJson(response)
    return texts

def save_json(img_path, dest):
    texts = detect_text(img_path)

    with open(dest, 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False, indent=4)

