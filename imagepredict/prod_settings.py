from .settings import *
STATIC_ROOT = 'static'
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO','https')
ALLOWED_HOSTS  = ['*']
DEBUG = True
