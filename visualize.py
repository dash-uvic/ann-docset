import os,sys
import numpy as np
from PIL import Image
import itertools
import cv2
from glob import glob
import h5py
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
from matplotlib import colors
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

from constants import CLASS_DICT, DISTINCT_COLORS 
xkcd = [ colors.to_rgb(x) for x in DISTINCT_COLORS] 
    

def draw_box(img, boxes):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    polygons = []
    color = []

    for bbox in boxes:
        print(bbox, type(bbox))
        poly = Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1])
        polygons.append(poly)
        color.append(bbox[4]) 
    
    p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
    ax.add_collection(p)

    plt.show() 

def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interporlation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image, cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.tight_layout()
    plt.show()

from utils import parse_hdf5
def load_segmentation(fn, split, group):
    hf = h5py.File(fn, 'r')  
    db = hf[split][group]
    return db, hf

def visualize_instance_segmentation(output_dir, split, data, targets):
    instance_segmentation_fn    = 'instance_segmentations.hdf5'
    hf_fn   = os.path.join(output_dir, instance_segmentation_fn)
    gt,hf      = load_segmentation(hf_fn, split, "instance")

    for idx in range(len(data)):
        img     = cv2.imread(data[idx], 1)
        img     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target  = np.array(gt[targets[idx]], dtype=np.uint8)
        obj_ids = np.unique(target)
        obj_ids = obj_ids[1:] #remove background
        print("labels", obj_ids)
        boxes = []
        num_objs = target.shape[0] 
        for i in range(num_objs):
            pos = np.where(target[i,:,:])
            if len(pos[0]) == 0: 
                print(f"Bad layer at {i}")
                continue #bad layer
            label = np.unique(target[i,:,:])[1]
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax, xkcd[label]])

        draw_box(img, boxes)
    
    hf.close()

from constants import ALL_CLASSES
def visualize_dense_segmentation(output_dir, split, data, targets):
    dense_segmentation_fn    = 'dense_segmentations.hdf5'
    hf_fn   = os.path.join(output_dir, dense_segmentation_fn)
    gt,hf   = load_segmentation(hf_fn, split, "dense")
   
    max_class_id = len(ALL_CLASSES)
    for idx in range(10):
        mask_np  = np.array(gt[targets[idx]], dtype=np.uint8).squeeze(0)#.transpose(1,2,0)
        
        img = Image.open(data[idx]).convert("RGB")
        img_orig = img.copy()
        img_bin = img.copy()
        
        masks   = (np.arange(max_class_id) == mask_np[...,None]-1).astype(np.uint8)
        
        titles = [f"{data[idx]}", "Dense Labelling"]
        images = [np.array(img_orig), mask_np]
        for i in range(masks.shape[-1]):
            if np.count_nonzero(masks[:,:,i]) == 0: continue
            mask = Image.fromarray(masks[:,:,i]*255).convert('L')
            img_ = img.copy() 
            img_.putalpha(mask)
            images.append(img_)
            titles.append(CLASS_DICT[i+1])

        cols=len(images)//2
        display_images(images, titles=titles, cols=cols)

    hf.close()

def visualize_multilabel_segmentation(output_dir, split, data, targets):
    multilabel_segmentation_fn    = 'multilabel_segmentations.hdf5'
    hf_fn   = os.path.join(output_dir, multilabel_segmentation_fn)
    gt,hf      = load_segmentation(hf_fn, split, "multilabel")

    
    for idx in range(10):
        masks  = np.array(gt[targets[idx]], dtype=np.uint8)
        
        img = Image.open(data[idx]).convert("RGB")
        img_orig = img.copy()
        img_bin = img.copy()

        titles = ["Original"]
        images = [np.array(img_orig)]
        for i in range(masks.shape[-1]):
            if np.count_nonzero(masks[:,:,i]) == 0: continue
            mask = Image.fromarray(masks[:,:,i]*255).convert('L')
            images.append(mask)
            titles.append(CLASS_DICT[i+1])

        cols=len(images)//2
        display_images(images, titles=titles, cols=cols)

    hf.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', type=str, default="annotated_docset-v1.0",
        help="Path to the annotated docset") 
    parser.add_argument('-s', '--split', type=str, default="val", choices=["val", "test", "train", 
        help="train/test/val split to use")
    parser.add_argument('-i', '--image', type=str, default=None,
        help="Image to load, by default all images in folder are used.")
    parser.add_argument('-v', '--visualize', type=str, choices=["instance", "multilabel", "dense", "stroke"],
        help="Which ground-truth to visualize")
    args = parser.parse_args()

    if args.image is not None:
        data = [os.path.join(args.data_dir, "images", args.split, args.image)]
        targets = [ os.path.basename(x).split(".")[0] for x in data ]
    else:
        data = sorted([ x for x in glob(os.path.join(args.data_dir, "images", args.split, f"*.png")) ])
        targets = [ os.path.basename(x).split(".")[0] for x in data ]

    if args.visualize == "instance":
        visualize_instance_segmentation(args.data_dir, args.split, data, targets)

    elif args.visualize == "dense":
        visualize_dense_segmentation(args.data_dir, args.split, data, targets)
    
    elif args.visualize == "multilabel":
        visualize_multilabel_segmentation(args.data_dir, args.split, data, targets)
    
    else:
        visualize_stroke_segmentation(args.data_dir, args.split, data, targets)

