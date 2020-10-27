import requests
import cv2
import base64
import json
from PIL import Image
import io
#url = 'http://103.124.72.45:9000/v1/models/mask_rcnn_hand_1000:predict'
url = 'http://203.145.218.191:9000/v1/models/mask_rcnn_shapes:predict'
content_type = 'image/jpeg'

im = cv2.imread('031.jpg', 1)[:,:,::-1]
im = cv2.resize(im,(512,512))
print(im.shape)
retval, buffer = cv2.imencode('.jpg', im)
im_encode = base64.b64encode(buffer)
my_img =im_encode
x = requests.post(url, data=my_img, timeout=600)
x = x.json()
print(type(x))