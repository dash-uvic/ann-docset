import numpy as np
import cv2

from constants import MAPPING, BASE_CLASSES, MARKUPS, ALL_CLASSES, DENSE_LAYERS, DENSE_LAYERS_, INK_COLORS


#todo: get_ink_properties
def find_ink_color(img, dim):
    h,w = dim
    try:
        median = tuple(map(int, np.median(np.median(img[:int(h),:int(w), :], axis=0), axis=0)))
    except:
        median = tuple(map(int, np.median(np.median(img, axis=0), axis=0)))

    if median[0] > 175: #light background
        ink_color = INK_COLORS[np.random.choice([0,1,4,5,6], p=[0.35,0.35,0.1,0.1,0.1])][1]
    else: #dark background
        ink_color = INK_COLORS[np.random.choice([2,3,4,5,6], p=[0.35,0.35,0.1,0.1,0.1])][1]
   
    return median, ink_color

def erase_existing_object(img, block_to_replace, color):
    #select a corresponding mapping class, and erase the original object
    pts = np.array(block_to_replace[-1]).reshape(-1, 1,2).astype(np.int32)
    bx,by,bw,bh = cv2.boundingRect(pts)
    cv2.rectangle(img,(bx,by),(bx+bw,by+bh),color,-1)

def remove_unannotated_elements(img, gt):
    mask = np.uint8(gt.squeeze(0) > 0)
    img_clean = cv2.bitwise_and(img, img, mask=mask)
    mask=mask-1
    img_clean[mask > 0] = np.array([255,255,255])
    return img_clean

def resize_ground_truth(gt, img_shape=[400,300]):
    new_gt = np.zeros((*img_shape, gt.shape[-1]), dtype=np.uint8)
    for idx in range(gt.shape[-1]):
        inst = gt[:,:,idx]
        inst = cv2.resize(inst, tuple(img_shape[::-1]), interpolation=cv2.INTER_CUBIC)
        new_gt[:,:,idx] = inst
    assert new_gt.shape == img_shape, f"shapes don't match: {new_gt.shape} != {img_shape}"
    assert np.unique(gt) == np.unique(new_gt), f"gt values changed"
    return new_gt

def create_dense_ground_truth(gt, dense_layers=False):
    
    gt = gt.transpose((2,0,1))
    _, h, w = gt.shape
    N = len(DENSE_LAYERS_) if dense_layers else 1
    dense_gt = np.zeros((N,h,w), dtype=gt.dtype)
    
    if dense_layers:
        for layer in gt:
            idx = layer.nonzero()
            
            try:
                didx = DENSE_LAYERS[np.unique(layer[idx])[0]]
                dense_layer= dense_gt[didx, ...]
                dense_layer[idx] = layer[idx]
            except Exception as e:
                print(f" !! issue with dense layer")
                raise e
                continue
                #print(DENSE_LAYERS)
                #print(idx)
                #print(np.unique(layer[idx]))
    else:
        for layer in gt:
            layer = layer[np.newaxis, ...]
            idx = layer.nonzero()
            dense_gt[idx] = layer[idx]

    return dense_gt
