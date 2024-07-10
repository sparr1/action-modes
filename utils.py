import datetime
import numpy as np

def datetime_stamp():
    now = datetime.datetime.now()
    datetime_stamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    return datetime_stamp

def tolerant_mean(arrs): #averaging performance values between trials of varying lengths
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return np.array(arr.mean(axis = -1)), np.array(arr.std(axis=-1))