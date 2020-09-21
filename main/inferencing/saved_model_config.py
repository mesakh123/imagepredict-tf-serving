# Your Inference Config Class
# Replace your own config
# MY_INFERENCE_CONFIG = YOUR_CONFIG_CLASS
from . import coco
import numpy as np
class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.9
    NUM_CLASSES = 1 + 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
coco_config = InferenceConfig()

MY_INFERENCE_CONFIG = coco_config


# Tensorflow Model server variable
ADDRESS = '203.145.218.191'
PORT_NO_GRPC = 8500
PORT_NO_RESTAPI = 8501
MODEL_NAME = 'mrcnn'
REST_API_URL = "http://%s:%s/v1/models/%s:predict" % (ADDRESS, PORT_NO_RESTAPI, MODEL_NAME)


# TF variable name
OUTPUT_DETECTION = 'mrcnn_detection/Reshape_1'
OUTPUT_CLASS = 'mrcnn_class/Reshape_1'
OUTPUT_BBOX = 'mrcnn_bbox/Reshape'
OUTPUT_MASK = 'mrcnn_mask/Reshape_1'
INPUT_IMAGE = 'input_image'
INPUT_IMAGE_META = 'input_image_meta'
INPUT_ANCHORS = 'input_anchors'
OUTPUT_NAME = 'predict_images'


# Signature name
SIGNATURE_NAME = 'serving_default'

# GRPC config
GRPC_MAX_RECEIVE_MESSAGE_LENGTH = 4096 * 4096 * 3 # Max LENGTH the GRPC should handle
