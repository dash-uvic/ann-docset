#!/usr/env python3
import os
import xmltodict
import numpy as np
import cv2
from glob import glob
import csv
import json

from utils import NpEncoder

classes = {
        "TextRegion":1, #text
        "FrameRegion":2,
        "GraphicRegion":3, #figure
        "ImageRegion":4, #figure
        "LineDrawingRegion":5,
        "MathsRegion":6, #text
        "NoiseRegion":7,
        "SeparatorRegion":8,
        "TableRegion":9, #table
        "ChartRegion":10 #figure
}

categories = {
        "TextRegion": "text", #text
        "FrameRegion": None,
        "GraphicRegion": "figure", #figure
        "ImageRegion": "figure", #figure
        "LineDrawingRegion": None,
        "MathsRegion": "text", #text
        "NoiseRegion": None,
        "SeparatorRegion": None,
        "TableRegion": "table", #table
        "ChartRegion": "chart" #figure?
}

def convert_to_mscoco(root, ann):
    regions = [ x for x in root.keys() if x.endswith('egion') ]
   
    coco = []
    for region in regions:
        details = root[region]
        if not isinstance(details, list):
            details = [details]
        
        for cat in details:
            #print(cat)
            #print(cat["@id"])
            annotation = {"id" : cat["@id"], 
                          "image_id" : ann["image_id"], 
                          "image_name" : ann["image_name"],
                          "category" : categories[region],
                          "category_id" : classes[region],
                          "segmentation" : None, 
                          "bbox" : None,
                          "properties" : None}
       
            coords = cat["Coords"]
            if coords is None:
                print("... no coords.")
                continue

            points = []
            for point in coords["Point"]:
                points.append([int(point['@x']),int(point['@y'])])
            annotation["segmentation"] = points
            pts = np.array(points, dtype=np.int32).reshape((-1,1,2))
            bbox = [x,y,w,h] = cv2.boundingRect(pts)
            annotation["bbox"] = bbox
        
            coco.append(annotation)

    return coco

def process_dataset(mscoco, data_dir, ann_dir): 
    for idx, xml_fn in enumerate(glob.glob(os.path.join(ann_dir, "*.xml"))):
        with open(xml_fn) as fd:
            try:
                e = xmltodict.parse(fd.read())
            except:
                print("Failed to open ",xml_fn)
                continue
             
        root = e['PcGts']['Page']
        height = int(root['@imageWidth'])
        width = int(root['@imageHeight'])
        img_fn = os.path.join(data_dir, root['@imageFilename'])
        if not os.path.exists(img_fn):
            print("{}: No source image found -> {}".format(xml_fn, img_fn))
            continue

        image = {'file_name'  : img_fn, 
                 'image_name' : img_fn,
                 'image_id' : idx, 
                 'height' : height, 
                 'width' : width ,
                 }
        mscoco["images"].append(image)
        mscoco_ = convert_to_mscoco(root, image)
        mscoco["annotations"].extend(mscoco_)

if __name__ == "__main__":
    import argparse as ap
    import glob, os
    
    data_dir="images"
    ann_dir="annotations"
    
    mscoco = {"images": [], "annotations" : []}
    mscoco = process_dataset(mscoco, data_dir, ann_dir)

    with open("prima.json", "w") as fp:
        json.dump(mscoco, fp, cls=NpEncoder)
   
    print("Done.")
