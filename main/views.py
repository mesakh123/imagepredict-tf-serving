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
import sys
from .views_utils import *
from .mrcnn import visualize

from .inferencing.saved_model_inference import detect_mask_single_image_using_grpc

class_names = ['BG', 'wound']


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA_DIR = os.path.join(BASE_DIR,"media")
# Create your views here.
#mu = np.load(os.path.join(BASE_DIR,r'imagenet/ilsvrc_2012_mean.npy'))
#mu =  mu.mean(axis = 1).mean(axis = 1)



def index(request):
    if request.method == "POST":
        str_time = randomString(8)
        if 'myfile' in request.FILES:
            files = request.FILES.get('myfile').read()
            image = image_save(files,str_time)
            opencv_image = np.array(image).copy()[:,:,::-1]
            print("opencv_image.shape : ",opencv_image.shape)
            result = detect_mask_single_image_using_grpc(opencv_image,str_time)

            ori_file_folder = os.path.join(MEDIA_DIR,str_time+'.jpg')
            cropped_file_folder = None
            file_name= str_time+".jpg"
            if result:
                predict_result = visualize.save_image(image, "test", result['rois'], result['mask'],
                    result['class'], result['scores'], class_names,scores_thresh=0.85)
                if predict_result:
                    print("predict_result : ",predict_result.shape)
                    cropped_file_folder =  os.path.join(MEDIA_DIR,str_time+'-cropped.jpg')
                    bounding_box = result['rois']
                    mask = result['mask']
                    bounding_box,mask = process_bounding_mask(bounding_box,mask)
                    x1,y1,x2,y2 = bounding_box
                    opencv_image = opencv_image[y1:y2,x1:x2]
                    cv2.imwrite(cropped_file_folder,opencv_image)
                    cv2.imwrite(ori_file_folder,predict_result)
            if cropped_file_folder:
                ori_file_folder = cropped_file_folder
            result1 , result2, suggestions = predict_init(ori_file_folder,str_time)

    return render(request,"index.html",locals())
