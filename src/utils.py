import os
import sys
import dill as pickle
import numpy as np
import pandas as pd
from src.exceptions import Customed_exception

def SaveObject(file_path, obj):
    try:
        Dpath = os.path.dirname(file_path)
        os.makedirs(Dpath, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)

    except Exception as e:
        raise Customed_exception(e, sys)