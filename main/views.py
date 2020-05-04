from django.shortcuts import render
from django.shortcuts import render,redirect
from django.template.loader import get_template
from django.template import RequestContext
from django.db.models import Q
from django.http import HttpResponse, HttpResponseRedirect
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from smtplib import SMTP ,SMTPAuthenticationError,SMTPException
from django.core.exceptions import ObjectDoesNotExist
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.conf.urls.static import static
import os,random
from .models import *
from .utils import randomString
from django.shortcuts import render
from django.contrib.auth import login, authenticate
from django.contrib.auth.forms import UserCreationForm,AuthenticationForm
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate,logout
from django.contrib import auth
from django.shortcuts import render, redirect, get_object_or_404, HttpResponseRedirect
from django.contrib.sites.shortcuts import get_current_site
from django.utils.encoding import force_text
from django.contrib.auth.models import User
from django.db import IntegrityError
from django.utils.http import urlsafe_base64_decode
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode
from django.template.loader import render_to_string
from django.contrib.auth.decorators import login_required
import json
from datetime import datetime,date
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
import glob
from PIL import Image
import cv2
import numpy as np
import caffe
import io
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA_DIR = os.path.join(BASE_DIR,"media")
# Create your views here.


def image_save(im = None,str_time="default"):
    if im is None : return False
    """
    image = cv2.imdecode(np.frombuffer(im, np.uint8), -1)
    width_resize = 224
    wpercent = (width_resize/float(image.shape[1]))
    height_resize  = int(float(image.shape[0])*float(wpercent))
    image = cv2.resize(image,(width_resize,height_resize))
    #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    file_folder = os.path.join(MEDIA_DIR,str_time+'.jpg')
    stats = cv2.imwrite(file_folder,image)
    print("Stats " , stats)
    """

    image = Image.open(io.BytesIO(im))
    width_resize = 224
    wpercent = (width_resize/float(image.size[0]))
    height_resize  = int(float(image.size[1])*float(wpercent))
    file_folder = os.path.join(MEDIA_DIR,str_time+'.jpg')
    image = image.convert("RGB")
    stats = image.save(file_folder)
    print("stats ",stats)
    return stats


def predict(data_type = "",file=""):
    caffe_root = os.path.join(BASE_DIR,data_type+"/")
    deploy_file = caffe_root+r'deploy.prototxt'
    model_file = caffe_root+r'model.caffemodel'
    net = caffe.Net(deploy_file, model_file, caffe.TEST)
    mu = np.load(os.path.join(BASE_DIR,r'imagenet/ilsvrc_2012_mean.npy'))
    mu_mean =  mu.mean(axis = 1).mean(axis = 1)
    transform = caffe.io.Transformer({'data' : net.blobs['data'].data.shape})
    transform.set_transpose('data', (2, 0, 1))
    transform.set_raw_scale('data', 255)
    transform.set_channel_swap('data', (2, 1, 0))
    transform.set_mean('data', mu_mean)

    imagenet_labels_filename = caffe_root+"/"+data_type +r'_label.txt'

    labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
    labels = labels.tolist()


    input_image = caffe.io.load_image(file,True)

    trans_image = transform.preprocess('data', input_image)
    net.blobs['data'].reshape = (1, 3, 224, 224)

    net.blobs['data'].data[0:, :] = trans_image
    output = net.forward()
    prediction = output['prob'][0]
    result_predict = str(labels[prediction.argmax()])
    return result_predict;

def predict_init(file=None):
    result1 = predict("infection",file)
    result2 = predict("necrotic",file)

    return result1,result2



def index(request):
    if request.method == "POST":
        str_time = randomString(8)
        if 'myfile' in request.FILES:
            files = request.FILES.get('myfile').read()
            image_save(files,str_time)
            file_folder = os.path.join(MEDIA_DIR,str_time+'.jpg')
            result1 , result2 = predict_init(file_folder)
            file = str_time+".jpg"
            print(file)
    return render(request,"index.html",locals())
