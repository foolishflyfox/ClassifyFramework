from .train_tools import train_model
from .logger import plot_history
from datetime import datetime

def get_timestamp():
    now = datetime.now()
    return datetime.strftime(now, '%Y%m%d_%H-%M-%S')
