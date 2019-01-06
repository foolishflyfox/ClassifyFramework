from .train_tools import train_model
from datetime import datetime

def get_timestamp():
    now = datetime.now()
    return datetime.strftime(now, '%Y%m%d%H%M%S')
