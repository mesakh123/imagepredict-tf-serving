import random
import string
from datetime import datetime
import secrets


def random_string_generator(size=7, chars=string.ascii_lowercase + string.digits):
    return ''.join(secrets.choice(chars) for _ in range(size))


def unique_order_id_generator(instance):
    order_new_id= random_string_generator()

    Klass= instance.__class__
    qs_exists= Klass.objects.filter(order_uuid= order_new_id).exists()
    if qs_exists:
        return unique_order_id_generator(instance)
    return datetime.now().strftime('%Y%m%d%H%M%S')+order_new_id


def randomString(stringLength=10):
    """Generate a random string of fixed length """
    lettersAndDigits = string.ascii_letters + string.digits
    return ''.join(secrets.choice(lettersAndDigits) for i in range(stringLength))
