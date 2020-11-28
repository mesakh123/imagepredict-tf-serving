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

from .inferencing.saved_model_inference import detect_mask_single_image_using_grpc,detect_mask_single_image_local

class_names = ['BG', 'wound']


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA_DIR = os.path.join(BASE_DIR,"media")
# Create your views here.
#mu = np.load(os.path.join(BASE_DIR,r'imagenet/ilsvrc_2012_mean.npy'))
#mu =  mu.mean(axis = 1).mean(axis = 1)

def apply_mask(image, mask, color=(1,0,0), alpha=0.75):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def index(request):
    if request.method == "POST":
        str_time = randomString(8)
        if 'myfile' in request.FILES:
            files = request.FILES.get('myfile').read()
            image_ori,image = image_save(files,str_time)#512 and 224
            opencv_image = np.array(image_ori).copy()#512
            result,result_unet = detect_mask_single_image_local(opencv_image,str_time)
            #result = detect_mask_single_image_using_grpc(opencv_image,str_time)

            ori_file_folder = os.path.join(MEDIA_DIR,str_time+'.jpg')
            cropped_file_folder = None
            file_name= str_time+".jpg"
            file_name_ori = str_time+"-ori.jpg"
            file_name_unet = None
            if result:
                for k,v in result.items():
                    result[k] = np.array(v)
                    print(k)
                predict_result = visualize.save_image(image_ori, ori_file_folder, result['rois'], result['masks'],
                    result['class_ids'], result['scores'], class_names,scores_thresh=0.85)
                if predict_result:
                    cropped_file_folder =  os.path.join(MEDIA_DIR,str_time+'-224.jpg')
                    bounding_box = result['rois'][0].copy()
                    mask = result['masks'][...,0].copy()
                    mask = 255.0*mask
                    result['rois'][0],mask = process_bounding_mask(bounding_box,mask)
                    x1,y1,x2,y2 = result['rois'][0]
                    print(x1,y1,x2,y2)
                    cropped_image = opencv_image[y1:y2,x1:x2].copy()[:,:,::-1]
                    cv2.imwrite(cropped_file_folder,cropped_image)

            result_unet = np.array(result_unet)
            if np.count_nonzero(result_unet):
                file_name_unet = str_time+"-unet.jpg"
                image = apply_mask(opencv_image[:,:,::-1],result_unet)
                unet_file_folder =  os.path.join(MEDIA_DIR,file_name_unet)
                cv2.imwrite(unet_file_folder,image)

            if cropped_file_folder:
                ori_file_folder = cropped_file_folder
            result1 , result2, suggestions = predict_init(ori_file_folder,str_time)

    return render(request,"index.html",locals())
