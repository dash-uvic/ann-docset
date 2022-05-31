import json
import cv2
import numpy as np
import os
import random

def load_image(img_fn):
    img = cv2.imread(img_fn, 1)
    if img is None:
        raise Exception(f"`{img_fn}` is not a valid image filename.")
    return img

def save_result(fname, np_arr): 
    if fname.endswith("npy"):
        np.save(fname, np_arr)
    else:
        cv2.imwrite(fname, np_arr)

def load_anns(image_dir, publaynet_fn, inkml_fn):
    print(f"Loading publaynet annotations")
    with open(publaynet_fn, "r") as fp:
        data = json.load(fp)
    
    print(f"Loading inkml annotations")
    with open(inkml_fn, "r") as fp:
        rel = json.load(fp)

    print(f"Setting up lookup tables")
    images = { x['id'] : os.path.join(image_dir, x['file_name']) for x in data["images"] }
    cats   = { x['id'] : x['name'] for x in data["categories"] }

    img_anns = {}
    anns   = data["annotations"]
    for ann in anns:
        item = [ (ann["id"], cats[ann["category_id"]], ann["bbox"], ann["segmentation"])]
        if images[ann["image_id"]] not in img_anns.keys():
            img_anns[images[ann["image_id"]]] = item
        else:
            img_anns[images[ann["image_id"]]].extend(item)
 
    authors = []
    for ann in rel["authors"]:
        authors.append(int(ann["authorId"]))
    authors = list(set(authors))
    rand_idxs = list(range(len(authors)))
    random.shuffle(rand_idxs)
    idx = int(len(authors)*0.30)
    test_idxs = rand_idxs[:idx]
    train_idxs = rand_idxs[idx:]
    print(f"Reserving {len(test_idxs)}/{len(test_idxs)+len(train_idxs)} authors for testing/validation")
    authors = np.array(authors)
    author_splits = {"train" : authors[train_idxs], 
                      "test" : authors[test_idxs],
                      "val" : authors[test_idxs]}
    
    assert set(authors[train_idxs]).isdisjoint(authors[test_idxs]), "crossover in the author lists"

    print(f"Mapping annotations ...")
    rel_anns = {}
    for ann in rel["annotations"]:
        item = [ (ann["id"], ann["image_name"], ann["bbox"], ann["segmentation"], int(ann["author_id"]))]
        if ann["category"] not in rel_anns.keys():
            rel_anns[ann["category"]] = item
        else:
            rel_anns[ann["category"]].extend(item)


    return img_anns, rel_anns, author_splits 
