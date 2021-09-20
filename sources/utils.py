import cv2
import inspect 
import json 
import numpy as np
import h5py
import matplotlib.pyplot as plt

def parse_hdf5(hf):
    data, group = [], []
    def func(name, obj):
        if isinstance(obj, h5py.Dataset):
            data.append(name)
        elif isinstance(obj, h5py.Group):
            group.append(name)
    hf.visititems(func)  
    return data, group


def store_many_hdf5(images, labels):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    """
    f2 = h5py.File('complevel_9.h5', 'w')
    f2.create_dataset('img', data=img, compression='gzip', compression_opts=9)
    f2.close()
    """
    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{num_images}_many.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    meta_set = file.create_dataset(
        "meta", np.shape(labels), h5py.h5t.STD_U8BE, data=labels
    )
    file.close()

def load_image(img_fn, debug=False):
    print(f"reading img: {img_fn}")
    img = cv2.imread(img_fn, 1)
    if img is None:
        raise Exception(f"`{img_fn}` is not a valid image filename.")
    
    if debug:
        visualize(img, wait=1)

    return img

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

def visualize(img, title=None, wait=1):
    if title is None:
        title = to_string(img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    cv2.imshow(title, img)
    cv2.waitKey(wait)
    
    if wait == 0:
        cv2.destroyAllWindows()

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
