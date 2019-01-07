from .train_tools import train_model
from .logger import plot_history
from .test_tools import TestImageFolder
from datetime import datetime

def get_timestamp():
    now = datetime.now()
    return datetime.strftime(now, '%Y%m%d_%H-%M-%S')
