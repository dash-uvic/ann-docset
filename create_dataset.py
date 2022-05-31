import json
import os
import cv2
import random
import numpy as np
import json
import h5py
import glob
import shutil
import matplotlib.pyplot as plt
import warnings
import itertools
import copy

from utils.misc import NpEncoder, parse_hdf5, check_hdf5, get_class, image_stats, iou_score 
from utils.data_manip import find_ink_color, erase_existing_object, create_dense_ground_truth, remove_unannotated_elements
from utils.file_io import load_anns, load_image, save_result 

from constants import MAPPING, MARKUPS, ALL_CLASSES, CLASS_DICT, DENSE_LAYERS_, BOX_IDX, INK_THICKNESS, MIN_BOX_SIZE

def setup_dirs():
    """ setup directories """

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    output_img_dir = os.path.join(output_dir, "images")
    os.makedirs(os.path.join(output_img_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_img_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(output_img_dir, "test"), exist_ok=True)

    return output_img_dir

def setup_hdf5():

    """ setup h5 files """

    hf_ml = h5py.File(os.path.join(output_dir, multilabel_segmentation_fn), 'a')  
    hf_in = h5py.File(os.path.join(output_dir, instance_segmentation_fn), 'a')  
    hf_ds = h5py.File(os.path.join(output_dir, dense_segmentation_fn), 'a')  
    hf_st = h5py.File(os.path.join(output_dir, stroke_segmentation_fn), 'a')  

    for split in ['train', 'val', 'test']:
        hf_ml.create_group(split).create_group('multilabel')
        hf_in.create_group(split).create_group('instance')
        hf_st.create_group(split).create_group('stroke')
        h = hf_ds.create_group(split)
        h.create_group('dense')
        h.create_group('dense_multilabel')

    return hf_ml, hf_in, hf_ds, hf_st


def add_overlay(img, markup, dont_add_here, debug=False): 
   
    """ add markup """

    sx,sy=1,1
    nx,ny,nw,nh = markup[2] 
    h,w  = [max(MIN_BOX_SIZE, nh), max(MIN_BOX_SIZE, nw)]
    H,W = img.shape[:2]
    sx, sy = w/nw,h/nh
   
    #randomly select point in image 
    for _ in range(10):
        x = np.random.uniform(10, W-nw, 1)[0]
        y = np.random.uniform(10, H-nh, 1)[0]
        if len(np.unique(img[int(y):int(y+nh),int(x):int(x+nw)])) > 1: #non-blank area
            break 
    
    bkg_color, ink_color = find_ink_color(img, (nh,nw))
    thickness = np.random.choice(INK_THICKNESS)
    lineType = cv2.LINE_AA if thickness == 1 else cv2.LINE_8

    img_backup = img.copy()
    pts_adj = []
    for pts in markup[-1]:
        pts_ = []
        for p in pts:
            pts_.append([[(p_[0]-nx)*sx+x, (p_[1]-ny)*sy+y] for p_ in p ])
        
        if len(pts_):
            pts = np.array(pts_).reshape(-1, 1,2).astype(np.int32)
            cv2.polylines(img,[pts], False, ink_color, thickness=thickness, lineType=lineType)
            pts_adj.append(pts_) 
    
    if len(pts_adj):
        pts_ = np.concatenate([pts for pts in pts_adj], axis=0)
        pts_ = np.array(pts_).reshape(-1, 1,2).astype(np.int32)
        bbox = cv2.boundingRect(pts_)
     
        #Too small
        if bbox[2]*bbox[3] < 25:
            img = img_backup
            return None
        
        try:
            for oldbox in dont_add_here:
                if iou_score(oldbox, bbox) > 0.2:
                    break
        except:
            if debug: print(f"!! zero-area: {bbox}")
            img = img_backup
            return None

        new_obj = ((markup[1], thickness), get_class(markup[1]), bbox, pts_adj)
        return new_obj

    img = img_backup
    return None

def update_annotation(img, block_to_replace, replace_with, existing, debug=False):
    
    """ update the publaynet annotation with a new handwritten or rescaled text """

    if replace_with is None: 
        if debug: print("     `replace_with` is None.")
        return (None, img)

    H,W = img.shape[:2]
    x,y,w,h = block_to_replace[2]

    #Assume the median value is the background
    bkg_color, ink_color = find_ink_color(img, (h,w))
    thickness = np.random.choice(INK_THICKNESS)
    lineType = cv2.LINE_AA if thickness == 1 else cv2.LINE_8

    #ensure a minimum size
    h,w  = [max(MIN_BOX_SIZE, h), max(MIN_BOX_SIZE, w)]
    nx, ny, nw, nh = bbox = replace_with[2]
    sx, sy = w/nw,h/nh

    if sx > 1.5 and sy > 1.5:
        x = np.random.uniform(x, w-nw, 1)[0]
        y = np.random.uniform(y, h-nh, 1)[0]

    #rescale
    sx = min(max(sx, 0.5), 1.5)
    sy = min(max(sy, 0.5), 1.5)

    #Erase existing block and make a copy for writing 
    erase_existing_object(img, block_to_replace, bkg_color)
    img_backup = img.copy()
    
    #Checks if the scaled image is inside the block
    def is_inside(pxy):
        px = (pxy[0]-nx)*sx+x
        py = (pxy[1]-ny)*sy+y
        if x <= px <= x+w and y <= py <= y+h:
            return [px,py]
        return False 
     
    #Create polyline handwritten array
    pts_adj = []
    for pts in replace_with[-1]:
        pts_ = []
        for p in pts:
            pts_.extend(filter(None, map(is_inside, p)))
        
        if len(pts_):
            pts_ = list(filter(None, pts_))
            pts = np.array(pts_).reshape(-1, 1,2).astype(np.int32)
            cv2.polylines(img,[pts], False, ink_color, thickness=thickness, lineType=lineType)
            pts_adj.append(pts_) 
   

    if len(pts_adj):
        #Get the bounding box and check if enough of the object is written
        pts_ = np.concatenate([pts for pts in pts_adj], axis=0)
        pts_ = np.array(pts_).reshape(-1, 1,2).astype(np.int32)
        #Resize the bounding box to the current polylines
        bbox = cv2.boundingRect(pts_)
        logger.write(f"   | old class={block_to_replace[1]}, new class={get_class(replace_with[1])}\n")
        new_obj = ((block_to_replace[0],replace_with[1],thickness), get_class(replace_with[1]), bbox, pts_adj)
        return (new_obj, img)

    #didn't work out, leave the block empty
    return (None, img_backup) 

areas = { k : [] for k in ALL_CLASSES }
def create_ground_truth(ann, img_shape):

    multilabel_gt = np.zeros((*img_shape, len(ALL_CLASSES)), dtype=np.uint8) 
    instance_gt = np.zeros((*img_shape,0), dtype=np.uint8) 
    stroke_gt = np.zeros((*img_shape,0), dtype=np.uint8) 
    
    for obj_idx, obj in enumerate(ann):
        #handwritten objects have ink thickness
        ann_id,cls,bbox,polygon = obj
        thickness=None
        if isinstance(ann_id, tuple):
            thickness = ann_id[-1]
            ann_id = ann_id[0]

        orig_cls = cls
        cls     = get_class(cls) #Class name
        cls_idx = ALL_CLASSES.index(cls) 
        label   = cls_idx+1 #Class label

        pts_ = np.empty((0,1,2), dtype=np.int32)
        
        onehot_layer    = multilabel_gt[:,:,cls_idx].copy() 
        labelled_layer  = np.zeros((*img_shape, 1), dtype=np.uint8)
        stroke_layer  = np.zeros((*img_shape, 1), dtype=np.uint8)
        
        for pts in polygon:
            pts = np.array(pts).reshape(-1, 1,2).astype(np.int32)
            if thickness is not None: #handwriting
                lineType = cv2.LINE_AA if thickness == 1 else cv2.LINE_8
                cv2.polylines(stroke_layer,[pts], False, label, thickness=thickness, lineType=lineType)
            pts_ = np.concatenate([pts_, pts], axis=0)
    

        logger.write(f"  | {orig_cls},{cls},{label}\n")
        #remove thickness from annotation
        if cls.startswith("Handwritten") or cls in MARKUPS:
            bx,by,bw,bh = bbox
            cv2.rectangle(onehot_layer,(bx,by),(bx+bw,by+bh),1,-1)
            cv2.rectangle(labelled_layer,(bx,by),(bx+bw,by+bh),label,-1)
        elif cls.startswith("Machine"):
            ann[obj_idx] = (ann_id, cls, bbox, pts_)
            cv2.fillPoly(onehot_layer, [pts_], 1)
            cv2.fillPoly(labelled_layer, [pts_], label)
        else:
            raise Exception(f"Unknown scenario: id={ann_id}, class={cls}, bbox={bbox}") 

        if np.count_nonzero(labelled_layer) > 0:
            #H,W,I: I = number of object instances
            instance_gt  = np.concatenate([instance_gt, labelled_layer], axis=-1)
            
        if np.count_nonzero(onehot_layer) > 0:
            #set class layer
            #H,W,C: C = number of classes
            multilabel_gt[:,:,cls_idx] = onehot_layer
        
        if np.count_nonzero(stroke_layer) > 0:
            stroke_gt = np.concatenate([stroke_gt, stroke_layer], axis=-1)
        
        areas[cls].append((bbox[-1]*bbox[-2])/(img_shape[0]*img_shape[1]))

    dense_gt =  create_dense_ground_truth(instance_gt)
    dense_multilabel_gt =  create_dense_ground_truth(instance_gt, dense_layers=True)

    assert dense_gt.shape[0] == 1, f"dense gt should be 1xHxC: {dense_gt.shape}"
    assert dense_multilabel_gt.shape[0] == len(DENSE_LAYERS_) , f"dense multilabel gt should be {len(DENSE_LAYERS_)}xHxC: {dense_multilabel_gt.shape}"

    return dense_gt, dense_multilabel_gt, multilabel_gt.transpose(2,0,1), instance_gt.transpose(2,0,1), stroke_gt.transpose(2, 0, 1)

def get_ratio(ann):
    return [a[1] for a in ann] 

from itertools import groupby
def perform_swaps(img, ann, updated_ann, rel_anns, split, authors, args):
    
    #Number of "swaps" to perform
    erased_blocks = []
    groups = { j : int(len(list(i)) / 2)  for j,i in groupby(get_ratio(ann)) }
    groups = { i : 1 if j == 0 and np.random.choice([True, False]) else j for i,j in groups.items() }
  
    total_anns = len(ann)

    #Sanity check to prevent infinit loop
    sanity = 20*len(ann)
    while sum(groups.values()) > 0 and sanity > 0:

        if len(ann) == 0:
            break
     
        #Randomly select block to replace, remove it
        block_to_replace = random.choice(ann)
        is_text = block_to_replace[1] == "text"
        sanity -= 1
        ann.remove(block_to_replace)

        #don't replace anymore of this type of MP objecs
        if groups[block_to_replace[1]] <= 0:
            updated_ann.append(block_to_replace)
            continue
    
        
        #resize text occassionally
        if is_text and np.random.choice([True, False], p=[args.rescale_rate, 1.-args.rescale_rate]):
            
            #extract textblock area
            scaled = (block_to_replace[0], 'text_scaled', *block_to_replace[2:])
            x,y,w,h = list(map(int, scaled[2]))
            textblock = img[y:y+h,x:x+w]
            
            #resize randomly
            scale_percent = np.random.uniform(1.1, 1.75) # percent of original size
            width = int(w * scale_percent)
            height = int(h * scale_percent)
            resized = cv2.resize(textblock, (width, height), interpolation = cv2.INTER_CUBIC)
            #place back in image
            img[y:y+h,x:x+w] = resized[:h,:w]
           
            #remove it from the sampling ann set
            updated_ann.append(scaled) 
            groups[block_to_replace[1]] -= 1
            
            logger.write(f"  - {block_to_replace[0]},{block_to_replace[1]},{scaled[0]},{scaled[1]},None\n")
            continue 
        else:
            #Possibly Don't certain source elements
            if len(MAPPING[block_to_replace[1]]) == 0:
                continue
        
            #Randomly select a handwriting object based on the mapping function
            for _ in range(25):
                replace_with = None
                map_ = np.random.choice(MAPPING[block_to_replace[1]][0], p=MAPPING[block_to_replace[1]][1])
                
                #No more objects available rn 
                if len(rel_anns[map_]) == 0:
                    continue 
                
                #find a random replacement
                idx = random.randrange(len(rel_anns[map_]))
                replace_with = rel_anns[map_][idx]
                author_id    = replace_with[-1]
                replace_with = replace_with[:-1] 
                    
                #prevent data leak by cross pollinating authors
                if author_id in authors[split]:
                    break 
           
            #remove so we have a balanced selection
            if replace_with is not None:
                del rel_anns[map_][idx]

        #perform update
        new_obj, img = update_annotation(img,block_to_replace,replace_with, updated_ann+ann)
         
        if new_obj is not None: 
            #successful swap found
            logger.write(f"  - {block_to_replace[0]},{block_to_replace[1]},{replace_with[0]},{replace_with[1]},{map_}\n")
            updated_ann.append(new_obj)
            groups[block_to_replace[1]] -= 1
        else:
            #block_to_replace is erased, add it back in to get resampled if necessary
            logger.write(f"     erase {block_to_replace[1]} region and re-add\n")
            erased_blocks.append(block_to_replace) 
            ann.append(block_to_replace) 

    if sanity <= 0:
        print(f"!! -- broke due to sanity check")
  
    #if an erased block is still in ann or updated_ann, remove it
    for blk in erased_blocks:
        if blk in ann:
            ann.remove(blk)
            logger.write(f"  - removed: {blk[0]},{blk[1]}\n")
        if blk in updated_ann:
            updated_ann.remove(blk)
            logger.write(f"  - removed: {blk[0]},{blk[1]}\n")

    info = {"annotations" : len(updated_ann), "ignored" : len(ann),  "erased" : len(erased_blocks)} 
    updated_ann = updated_ann + ann

    return updated_ann, info 

def perform_overlay(img, updated_ann, rel_anns, split, authors, args): 
    #number of markups to apply
    num_markups = np.random.randint(args.min_markup, args.max_markup, size=1)[0]
    n_markup = 0
    sanity_check = 50 
    dont_add_here = []
    while n_markup < num_markups and sanity_check > 0: 

        map_ = random.choice(MARKUPS) 
        
        if len(rel_anns[map_]) == 0:
            sanity_check -= 1
            continue
        
        idx          = random.randrange(len(rel_anns[map_]))
        markup       = rel_anns[map_][idx]
        author_id    = markup[-1]
        markup       = markup[:-1] 

        if author_id not in authors[split]:
            sanity_check -= 1
            continue
        
        new_obj = add_overlay(img, markup, dont_add_here, debug=args.debug) 
        if new_obj is not None:
            updated_ann.append(new_obj)

            #remove markup and don't add markups close by 
            del rel_anns[map_][idx] 
            n_markup += 1
            dont_add_here.append([new_obj[2][0], new_obj[2][1], new_obj[2][0]+new_obj[2][2], new_obj[2][1]+new_obj[2][3]])

    return updated_ann



def erase_non_annotated_elements(img, ann_orig):
    dense_gt = create_ground_truth(ann_orig[:], img.shape[:2])[0]
    img = remove_unannotated_elements(img, dense_gt) 
    return img

def main(hf_ml, hf_in, hf_ds, hf_st, output_dir, output_img_dir, args): 

    #Save the labels 
    with open(f'{output_dir}_labels.json', "w") as json_file:
        json.dump(CLASS_DICT, json_file, indent=4)
  
    #Load annotations

    pub_fn = f"{data_dir}/val.json"
    inkml_fn = "inkml.json"
    
    if not os.path.exists(pub_fn):
        raise Exception(f"{pub_fn} must exist. Check your PubLayNet path or download.")
    if not os.path.exists(inkml_fn):
        raise Exception(f"{inkml_fn} must exist. Run `parse_inkml.py` first.")

    img_anns, rel_anns, authors = load_anns(image_dir, pub_fn, inkml_fn)
   
    #Load splits
    print("Load dataset train/test/val splits from splits.csv")
    with open("splits.csv", "r") as fh:
        dset_split = dict( (x.strip().split(",")[1] , x.strip().split(",")[0]) for x in fh.readlines() )

    json_fn = f'{output_dir}/{output_dir}.json'
    
    train_json_obj = {}
    test_json_obj = {}
    val_json_obj = {}
    stats = {}
  
    orig_rel_anns = copy.deepcopy(rel_anns) 
    
    print("Generating images (this will take a while!)")
    for visit in range(args.variations_per_doc):
        for img_idx, (img_fn, ann_orig) in enumerate(img_anns.items()):
            basename = os.path.basename(img_fn).split(".")[0]
            ver = f"{basename}-{visit}" 
            split=dset_split[img_fn]
            
            print(f"[{visit}:{img_idx}] {basename}:{split}/{ver}", flush=True)
            logger.write(f"[{visit}:{img_idx}] {basename}\n")

            img = load_image(os.path.join(data_dir, img_fn))
            img_height, img_width = img.shape[:2]

            #make sure that there isn't anything not annotated in the image
            img = erase_non_annotated_elements(img, ann_orig)
            
            updated_ann = [] 
            ann         = ann_orig[:]
          
            logger.write(f"  - templateAnnId,templateImg,authorId,annImg,class\n")
            
            updated_ann,  info = perform_swaps(img, ann, updated_ann, rel_anns, split, authors, args)
            updated_ann = perform_overlay(img, updated_ann, rel_anns, split, authors, args) 

            #once all the existing samples are used, reload
            for map_ in rel_anns.keys():
                if len(rel_anns[map_]) == 0:
                    logger.write(f"{map_}: RELOAD\n")
                    rel_anns[map_] = orig_rel_anns[map_][:] 
            
 
            #Bit of a hack to ensure that the order of: Machine, Handwritten, Marking is used
            updated_ann.sort(key=lambda tup: "A" if "Mark" in tup[1] else tup[1], reverse=True) 
    
            #calculate some stats
            stats[ver] = image_stats([ (get_class(x[1]), x[2][2]*x[2][3]) for x in updated_ann ], img_height*img_width, info)
           
            dense_gt, \
            dense_multilabel_gt, \
            multilabel_gt,\
            instance_gt, \
            stroke_gt = create_ground_truth(updated_ann, img.shape[:2])
            
            #save the newly create image
            img_name = os.path.join(output_img_dir, split, f"{ver}.png")
            save_result(img_name, img)
            
            #save results 
            hf_st[split]["stroke"].create_dataset(ver, data=stroke_gt, compression='gzip', compression_opts=9)
            hf_ml[split]["multilabel"].create_dataset(ver, data=multilabel_gt, compression='gzip', compression_opts=9)
            hf_in[split]["instance"].create_dataset(ver, data=instance_gt, compression='gzip', compression_opts=9)
            hf_ds[split]["dense"].create_dataset(ver, data=dense_gt, compression='gzip', compression_opts=9)
            hf_ds[split]["dense_multilabel"].create_dataset(ver, data=dense_multilabel_gt, compression='gzip', compression_opts=9)

            if split == "train":
                train_json_obj[ver] = [ (ann_id, get_class(cls), bbox,poly) for ann_id, cls,bbox,poly in updated_ann ]  
            elif split == "test":
                test_json_obj[ver] = [ (ann_id, get_class(cls), bbox,poly) for ann_id, cls,bbox,poly in updated_ann ]  
            elif split == "val":
                val_json_obj[ver] = [ (ann_id, get_class(cls), bbox,poly) for ann_id, cls,bbox,poly in updated_ann ]  
            else:
                raise Exception(f"Unknown `split`: {split}")


            if args.debug and img_idx >= 60:
                warnings.warn(f"TESTING ENABLED: only running {img_idx} images")
                break
    
    hf_in.close()            
    hf_ml.close()
    hf_ds.close()
    hf_st.close()

    print("Saving some stats")
    with open(f'{output_dir}/stats.json', "w") as json_file:
        json.dump(stats, json_file, cls=NpEncoder)
    
    with open(f'{output_dir}/authors.json', "w") as json_file:
        json.dump(authors, json_file, cls=NpEncoder)
    
    shutil.move(f"{output_dir}.log", output_dir)
    shutil.move(f"{output_dir}_labels.json", output_dir)

    print("Saving updated annotation to json")
    with open(f'{output_dir}/{output_dir}_train.json', "w") as json_file:
        json.dump(train_json_obj, json_file, cls=NpEncoder)
    with open(f'{output_dir}/{output_dir}_test.json', "w") as json_file:
        json.dump(test_json_obj, json_file, cls=NpEncoder)
    with open(f'{output_dir}/{output_dir}_val.json', "w") as json_file:
        json.dump(val_json_obj, json_file, cls=NpEncoder)
       
def verify(output_img_dir):
    #verify
    print("Verifying the data")
   
    print(f" checking `{instance_segmentation_fn}` saved correctly")
    hf = os.path.join(output_dir, instance_segmentation_fn)
    check_hdf5(hf, "instance", output_img_dir)

    print(f" checking `{dense_segmentation_fn}` saved correctly")
    hf = os.path.join(output_dir, dense_segmentation_fn)
    check_hdf5(hf, "dense", output_img_dir) 
    check_hdf5(hf, "dense_multilabel", output_img_dir) 

    print(f" checking `{multilabel_segmentation_fn}` saved correctly")
    hf = os.path.join(output_dir, multilabel_segmentation_fn)
    check_hdf5(hf, "multilabel", output_img_dir) 


if __name__ == "__main__":
    
    import argparse as ap
    desc="Annotated DocSet: Create Dataset"
    parser = ap.ArgumentParser(description=desc)
    parser.add_argument("--seed", type=int,
            default=42,
            help="Set random seed for reproducability")
    parser.add_argument("--publaynet", type=str, 
            default="datasets/publaynet",
            help="Path to the PubLayNet directory")
    parser.add_argument("--tag", type=str,
            default="v1.0",
            help="Version tag for the dataset")
    parser.add_argument("--variations-per-doc", type=int,
            default=1,
            help="Number of variations per document")
    parser.add_argument("--max-markup", type=int,
            default=10,
            help="Maximum number of markups/annotations per document")
    parser.add_argument("--min-markup", type=int,
            default=5,
            help="Minimum number of markups/annotations per document")
    parser.add_argument("--rescale-rate", type=int,
            default=0.25,
            help="Probability that machine-print text is rescaled")
    parser.add_argument("--debug",
            action="store_true",
            help="enable debug mode")

    args = parser.parse_args()
    
    np.random.seed(args.seed)
    random.seed(args.seed)

    data_dir = args.publaynet
    image_dir = "val" 
    output_dir = f"annotated_docset-{args.tag}"
    logger = open(f"{output_dir}.log", "w")

    instance_segmentation_fn    = 'instance_segmentations.hdf5'
    multilabel_segmentation_fn  = 'multilabel_segmentations.hdf5'
    dense_segmentation_fn       = 'dense_segmentations.hdf5'
    stroke_segmentation_fn      = 'stroke_segmentations.hdf5'
    num_classes = len(ALL_CLASSES)

    print(f"Creating dataset version {args.tag} (debug={args.debug})")

    output_img_dir = setup_dirs()
    hf_ml, hf_in, hf_ds, hf_st = setup_hdf5()
    main(hf_ml, hf_in, hf_ds, hf_st, output_dir, output_img_dir, args)
    verify(output_img_dir)
    
    for k, lst in areas.items():
        print(f"{k}: mean={np.mean(lst):.5f} +/ {np.std(lst):.5f}")

    print("All done!")
    logger.close()
