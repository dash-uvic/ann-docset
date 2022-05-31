import numpy as np
from skimage.draw import line
from skimage.morphology import thin

import matplotlib.pyplot as plt
from matplotlib import transforms
import matplotlib._color_data as mcd
import matplotlib.patches as mpatch

import xml.etree.ElementTree as ET
from io import StringIO

import cv2
import json
import hashlib

import pprint 
pp = pprint.PrettyPrinter(indent=4)

from utils.misc import NpEncoder


def colormap(class_colors, output_path):

    fig = plt.figure(figsize=[4.8, 20])
    ax = fig.add_axes([0, 0, 1, 1])

    for j, (cls,cn) in enumerate(class_colors.items()):
        weight = None
        r1 = mpatch.Rectangle((0, j), 1, 1, color=cn)
        txt = ax.text(1, j+.5, ' ' + cls, va='center', fontsize=10,
                      weight=weight)
        ax.add_patch(r1)
        ax.axhline(j, color='k')

    ax.text(.5, j + 1.5, 'xkcd', ha='center', va='center')
    ax.text(1.5, j + 1.5, 'title', ha='center', va='center')
    ax.set_xlim(0, 3)
    ax.set_ylim(0, j + 2)
    ax.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.gcf().clear()

def get_traces_data(inkml_file_abs_path):

        tree = ET.parse(inkml_file_abs_path)
        root = tree.getroot()
      
        author_details = {}
        for ann in root.findall('annotation'):
            if ann.get('type').startswith('author'):
                author_details[ann.get('type')] = ann.text

        matrix = root.find('definitions').find('canvasTransform').find('mapping').find('matrix')
        if matrix is not None:
            matrix = matrix.text.split(',')
            matrix = np.array([ float(x) for d in matrix[:2] for x in d.split(' ')[:2] ])
       
        'Stores traces_all with their corresponding id'
        traces_all = {} 
        for trace_tag in root.findall('trace'):
            id_ = trace_tag.get('{http://www.w3.org/XML/1998/namespace}id')
            coords = (trace_tag.text).replace('\n', '').split(',')
            
            x,y = [ float(x) for x in coords[0].split(' ') if x][:2]
            
            traces_all[id_] = [[x,y]] 
           
            try:
                vx, vy = [ float(x) for x in coords[1].split("'") if len(x) > 0 ][:2]
                px,py = traces_all[id_][-1] 
                traces_all[id_].append([px + vx, py + vy])
            except Exception as e:
                continue

            try:
                ax, ay = [ float(x) for x in coords[2].split('"') if len(x) > 0 ][:2]
                vx+=ax
                vy+=ay
                traces_all[id_].append([px + vx, py + vy])
            except Exception as e:
                continue

            for coord in coords[3:]:
                rel_coords =  [float(n) for n in coord.replace('-', ' -').split(" ") if n]
                vx += rel_coords[0]
                vy += rel_coords[1]

                px,py = traces_all[id_][-1] 
                traces_all[id_].append([px + vx, py + vy])

        traces_data = {'matrix' : matrix, 'traces' : []}
        json_data = []
        traceGroupWrapper = root.find('traceView')
        create_json_annotation(traceGroupWrapper, json_data)
        for ann in json_data:
            pp.pprint(ann['heirarchy'])
            traces_curr=[]
            for traceDataRef in ann["traces"].keys():
                single_trace = traces_all[traceDataRef]
                ann["traces"][traceDataRef] = single_trace
                traces_curr.append(single_trace)
            
            traces_data['traces'].append({'label': ann['heirarchy'], 'trace_group': traces_curr})
        return traces_data, json_data, author_details

def create_json_annotation(root, json_data, group=0, level=0, leaf=False, last_leaf=False, heirarchy=[], points={}, properties=[]):
    if level == 0:
        heirarchy = []
        points = {}
        properties = []

    if root.get('traceDataRef') is not None:
        points[root.get('traceDataRef')[1:]] = None

    for i, elem in enumerate(root.getchildren()):
        if elem.tag == 'annotation' and elem.get('type') in ('transcription', 'orientation'):
            properties.append({elem.get('type') : elem.text}) 
        elif elem.text is not None:
            heirarchy.append((elem.text, group))
        elif elem.tag == 'traceView': pass
        else:
            raise Exception(f"Unknown element: {elem}") 

        json_data, group, level, heirarchy, points, properties = create_json_annotation(elem, 
                                                  json_data,
                                                  group,
                                                  level+1,
                                                  bool(not elem), 
                                                  (i+1)==len(root.getchildren()), 
                                                  heirarchy,
                                                  points,
                                                  properties 
                                                  )
   
    if last_leaf:
        if leaf: 
            json_data.append({'heirarchy' : heirarchy[1:], 'traces' : points, 'properties': properties})
        return json_data, group, level-1, heirarchy[:-1], {}, [] 

    return json_data, group+1, level-1, heirarchy, points, properties
  
from matplotlib import colors
def transform_points(traces, json_data):
    
    M = traces["matrix"]
    traces = traces["traces"]
    
    min_x, min_y = np.inf, np.inf
    max_y, max_x = -np.inf, -np.inf 
    
    if M is not None:
        M = M.reshape((2,2))
    else:
        M = np.array([[1,0],[0,1]])

    for obj in json_data:
        for tr, pts in obj["traces"].items():
            pts = np.array([list(np.dot(M, x)) for x in pts] , dtype=np.int32)
            y,x = zip(*pts)
            min_x, min_y = min(min(x), min_x), min(min(y), min_y)
            max_x, max_y = max(max(x), max_x), max(max(y), max_y)
            obj["traces"][tr] = pts
        
    width   = max_y if min_y > 0 else max_y - min_y
    height  = max_x if min_x > 0 else max_x - min_x
  
    offset = 25
    width = width + 2*offset
    height = height + 2*offset

   
    offset_x, offset_y = min_x, min_y
    min_x, min_y = np.inf, np.inf
    max_y, max_x = -np.inf, -np.inf 

    for obj in json_data:
        for tr, pts in obj["traces"].items():
            pts = np.array([ (y - offset_y + offset if offset_y < 0 else y + offset, 
                              x - offset_x + offset if offset_x < 0 else x + offset) 
                            for (y,x) in pts ], dtype=np.int32)
            obj["traces"][tr] = pts.reshape((-1,1,2))

    return height, width


def convert_to_mscoco(json_data, ann, author, categories=[], ink_colors=[], category_dir="images/categories"):
    height = ann["height"]
    width  = ann["width"]
    ink_color = 255

    mscoco = [] 
    inst_ids = list(set([ h[1] for obj in json_data for h in obj["heirarchy"] ]))
    
    for id_ in inst_ids:
        img = np.zeros((height,width), np.uint8)
        pts_ = np.empty((0,1,2), dtype=np.int32)
        pts_norm = [] 

        subgroups=[]
        annotation = {"id" : id_, 
                      "image_id" : ann["image_id"], 
                      "author_id" : author["authorId"], 
                      "image_name" : None,
                      "category" : None,
                      "category_id" : None,
                      "subcategory" : None, 
                      "subgroups" : None,
                      "segmentation" : None, 
                      "bbox" : None,
                      "properties" : None}
        
        for obj in json_data:
            draw=False
            for cls in obj["heirarchy"]:
                if id_ in cls:
                    draw=True
                    
                    idx = obj["heirarchy"].index(cls)
                    if cls[0] not in [ c["name"] for c in categories ]:
                        categories.append({"id": len(categories), 
                                           "name" : cls[0], 
                                           "supercategory" : obj["heirarchy"][idx-1][0] if idx else '',
                                           })
                        os.makedirs(os.path.join(category_dir, cls[0]), exist_ok=True)

                    annotation["category"] = cls[0] 
                    annotation["category_id"] = [ t["id"] for t in categories if t["name"] == cls[0] ][0]
                    annotation["subcategory"] = [h[0] for h in obj["heirarchy"][idx+1:]]
                    
                    subgroups.extend([ x[1] for x in obj["heirarchy"][idx+1:] ])
                    annotation["properties"] = obj["properties"]
                    break
            if draw:
                for _, pts in obj["traces"].items():
                    cv2.polylines(img,[pts],False,ink_color, thickness=1, lineType=cv2.LINE_AA)
                    pts_norm.append(pts)
                
                pts  = np.concatenate([pts for pts in obj["traces"].values()], axis=0)
                pts_ = np.concatenate([pts_, pts], axis=0)

        annotation["subgroups"]     = list(set(subgroups)) 
        bbox = [x,y,w,h]            = cv2.boundingRect(pts_)
        annotation["bbox"]          = bbox
        annotation["segmentation"]  = pts_norm

        fn = os.path.join(category_dir, 
                          annotation["category"], 
                          os.path.basename(ann["file_name"]).replace('.inkml', f"_{annotation['id']}.png"))
        annotation["image_name"] = fn
        cv2.imwrite(fn,img[y:y+h, x:x+w])

        mscoco.append(annotation)

    xkcd = list(mcd.XKCD_COLORS.keys())
    img_ann = np.zeros((height,width,3), np.uint8)
    img_org  = np.zeros((height,width), np.uint8)
    for obj in json_data:
        
        label = "|".join(x[0] for x in obj["heirarchy"]) 
        if label not in ink_colors.keys():
            ink_colors[label] = xkcd[len(ink_colors.keys())] 
        ink_color = tuple(map(int, [x*255 for x in colors.to_rgba(ink_colors[label])]))
        
        for tr, pts in obj["traces"].items():
            cv2.polylines(img_ann,[pts],False, ink_color, thickness=1, lineType=cv2.LINE_AA)
            cv2.polylines(img_org,[pts],False, 255, thickness=1, lineType=cv2.LINE_AA)
        
        cv2.imwrite(ann["original"],img_org)
        cv2.imwrite(ann["labelled"],img_ann)
    
    return mscoco 

if __name__ == "__main__":
    import argparse as ap
    import glob, os
    
    desc="Annotated DocSet: Convert INKML to PNG"
    parser = ap.ArgumentParser(description=desc)
    parser.add_argument("--iamondo", type=str, 
            default="datasets/IAMonDo-db-1.0",
            help="Path to the IAMonDo directory")
    parser.add_argument("--output-dir", type=str, 
            default="datasets/IAMonDO-Images",
            help="Path to the output directory where the converted images are saved")
    args = parser.parse_args()

    data_dir = args.iamondo 
    out_dir =  args.output_dir
    color_map_fn = os.path.join(out_dir, "colormap.json") 
    
    orig_dir = os.path.join(out_dir, "original")
    lbl_dir = os.path.join(out_dir, "labelled")
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    
    if os.path.exists(color_map_fn):
        print("Loading previously found labels/categories")
        with open(color_map_fn, 'r') as fp:
            ink_colors = json.load(fp)
    else:
        ink_colors={}
   
    categories = []
    mscoco = {"images": [], "annotations" : [], "authors" : []}
    for idx, input_path in enumerate(sorted(glob.glob(os.path.join(data_dir, "*.inkml")))):
        output_path = os.path.basename(input_path)
        output_path = os.path.join(out_dir, output_path.replace(".inkml", ".png"))
        
        print(f"\n{input_path} -> {output_path}\n")
        
        traces, json_data, author = get_traces_data(input_path)
        height,width = transform_points(traces, json_data)
        basename = os.path.basename(input_path).replace('.inkml', '.png')
        image = {'file_name'  : input_path, 
                 'labelled'   : os.path.join(lbl_dir, basename),
                 'original' : os.path.join(orig_dir, basename),
                 'image_id' : idx, 
                 'height' : height, 
                 'width' : width ,
                 'author' : author,
                 }
        mscoco["images"].append(image)
        mscoco["authors"].append(author)
        category_dir=os.path.join(out_dir, "categories")
        mscoco_ = convert_to_mscoco(json_data, image, author, categories, ink_colors=ink_colors, category_dir=category_dir)
        mscoco["annotations"].extend(mscoco_)
        
    mscoco["categories"] = categories
    with open("inkml.json", "w") as fp:
        json.dump(mscoco, fp, cls=NpEncoder)

    if not os.path.exists(color_map_fn):
        with open(color_map_fn, 'w') as fp:
            json.dump(ink_colors, fp)
            colormap(ink_colors, color_map_fn.replace(".json", ".png"))
