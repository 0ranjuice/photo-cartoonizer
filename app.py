# -*- coding: utf-8 -*-
# 載入LineBot所需要的套件
from flask import Flask, request, abort
from linebot import (LineBotApi, WebhookHandler)
from linebot.exceptions import (InvalidSignatureError)
from linebot.models import *
import pyimgur
import cv2
import numpy as np

app = Flask(__name__)

# 必須放上自己的Channel Access Token
line_bot_api = LineBotApi(
    'haz544FhuBP/m3xKqQ2B/RATT17n8Vwp380L9BfFj9UN+SpHMLUloC61EG8VE0e/qvU0BaYxdlHtvZoBjVYdrSDsK2ysH+YtyA/ohx8l2QGc3K6TmP8sGYsr6IvlG/B4ARQMbxRSntZMD6QJNBDdlwdB04t89/1O/w1cDnyilFU=')
# 必須放上自己的Channel Secret
handler = WebhookHandler('5c5342dccfd566f8cb7420c77e73d6c5')

# Push a message to me when everything is ready ( 放上 USER ID )
line_bot_api.push_message('U1c7b721a4d53f66c456edf1d30681ae3', TextSendMessage(text='LineBot is ready to go !'))


# 監聽所有來自 /callback 的 Post Request
@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'


# Upload the image and get the image url #
def glucose_graph(client_id, img_path):
    img = pyimgur.Imgur(client_id)
    upload_image = img.upload_image(img_path, title="Uploaded with PyImgur")
    return upload_image.link


def color_quantization(img, k):
    # Transform the image
    data = np.float32(img).reshape((-1, 3))

    # Determine criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    # Implementing K-Means
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result


# Outline of the image #
def edge_mask(img, line_size, blur_value):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)  # To reduce noise
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    return edges


@handler.add(MessageEvent)
def handle_message(event):
    if event.message.type == 'image':  # Check if the type of message is "image"
        message_content = line_bot_api.get_message_content(event.message.id)  # Get the message content by message ID
        img_file = 'Images'
        # Write file
        with open(img_file, 'wb') as fd:
            for chunk in message_content.iter_content():
                fd.write(chunk)

        image = cv2.imread(img_file, -1)  # Read image

        # Parameter setting
        line_size = 7
        blur_value = 7
        total_color = 10

        edges = edge_mask(image, line_size, blur_value)  # Get the Outline of the image

        quant_img = color_quantization(image, total_color)  # Reduce the number of colors in the image

        blurred = cv2.bilateralFilter(quant_img, d=7, sigmaColor=150, sigmaSpace=150)  # Blur the image without ruining the edges

        cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)  # Merge the outline and the color-reduced image

        cv2.imwrite('rtn.png', cartoon)

        img_url = glucose_graph(client_id='4bf5bca0439f960', img_path='rtn.png')  # Upload the image
        #  Reply the cartoon-style image to user
        line_bot_api.reply_message(event.reply_token,
                                   ImageSendMessage(original_content_url=img_url, preview_image_url=img_url))


# 主程式
import os
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
