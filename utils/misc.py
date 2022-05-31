import cv2
import inspect 
import json 
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from collections import Counter
import glob
import random

from constants import MAPPING, BASE_CLASSES, MARKUPS, ALL_CLASSES

def check_hdf5(hf_fn, group_name, output_img_dir):
    hf = h5py.File(hf_fn, 'r')  
    data, group = parse_hdf5(hf)
   
    known_groups = ['train', f'train/{group_name}', 
                     'val', f'val/{group_name}', 
                     'test', f'test/{group_name}']
    for x in known_groups:
        assert x in group, f"`{x}` not in {group}"
  
    for split in ['train', 'val',  'test']:
        x = hf[split][group_name]
        y = glob.glob(os.path.join(output_img_dir, split, "*.png"))
        assert len(x) == len(y), f"No. of source images and {split}/{group_name} in hdf5 should be the same: {len(x)} != {len(y)}"
    hf.close()

def parse_hdf5(hf):
    data, group = [], []
    def func(name, obj):
        if isinstance(obj, h5py.Dataset):
            data.append(name)
        elif isinstance(obj, h5py.Group):
            group.append(name)
    hf.visititems(func)  
    return data, group

def image_stats(stats, img_area, info): 
    hw_per_img,mw_per_img,mk_per_img = 0, 0, 0
    mw_area, hw_area,mk_area = 0, 0, 0
    total = 0
    ann_counter = Counter([x for x, _ in stats])
    for s in stats:
        #print("s=", s)
        if s[0] in MARKUPS: 
            mk_per_img += 1
            hw_per_img += 1
            mk_area += s[1]
        elif 'Hand' in s[0]:
            hw_per_img += 1
            hw_area += s[1]
        elif 'Mach' in s[0]: 
            mw_per_img += 1
            mw_area += s[1]
        else:
            raise Exception("Unknown class in `image_stats`")
        total += 1 
    
    meta = {}
    meta['area_image']      = img_area
    meta['area_handwritten']= hw_area / img_area 
    meta['area_machine']    = mw_area / img_area
    meta['area_markup']     = mk_area / img_area
    meta['per_handwritten'] = hw_per_img / total 
    meta['per_machine']     = mw_per_img / total 
    meta['per_markup']      = mk_per_img / total 
    meta['count']           = dict(ann_counter)
    meta['objects']         = total
    meta['info']            = info
    return meta

def get_class(string):

    if string in ALL_CLASSES: return string 
    
    if "/" in string:
        string = get_class_from_path(string)

    if string[0].islower(): #machine class
        string = string[0].upper() + string[1:]
        if string in ("Text", "Text_scaled"): 
            string = "Textblock"
        if string == "Title": string = "Textline" 
        return f"Machine-{string}"
    elif string in MARKUPS:
        return string
    else:
        if string == "Word": string = "Textline" #"Textword"
        if string == "Drawing": string = "Figure"
        if string == "Diagram": string = "Figure"
        return f"Handwritten-{string}"

def get_class_from_path(path):
    return os.path.dirname(path).split("/")[-1]


def to_string(var, follow_link=True):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    idx = -1 if follow_link else 0
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[idx]

#https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def write_json(key, new_data, filename='data.json'):
    if not os.path.exists(filename):
        with open(filename,'w') as fp:
            json.dump({key: new_data}, fp,cls=NpEncoder)
        return

    with open(filename,'r+') as fp:
        file_data = json.load(fp)
        file_data[key] = new_data
        fp.seek(0)
        json.dump(file_data, fp, indent = 4, cls=NpEncoder)

def iou_score(bbox,gtbox):
    bbox = np.float32(bbox)
    gtbox = np.float32(gtbox)

    xA = max(bbox[0], gtbox[0])
    yA = max(bbox[1], gtbox[1])
    xB = min(bbox[2], gtbox[2])
    yB = min(bbox[3], gtbox[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    bboxArea = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
    gtboxArea = (gtbox[2] - gtbox[0] + 1) * (gtbox[3] - gtbox[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(bboxArea + gtboxArea - interArea)

    # return the intersection over union value
    return iou
