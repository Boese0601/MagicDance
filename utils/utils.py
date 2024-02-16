import random 
import numpy as np
import torch 
import matplotlib.pyplot as plt
from PIL import Image 

def merge_lists_by_index(list1, list2):
    # Check if both lists have the same number of elements
    if len(list1) != len(list2):
        raise ValueError("Both lists should have the same number of elements.")

    # Merge the lists by concatenating strings at the same index
    merged_list = [string1 + '. ' + string2 for string1, string2 in zip(list1, list2)]
    return merged_list

def plot_x_y(x, y, x_label, y_label, save_path, **kwargs):
    
    plt.plot( x , y, **kwargs)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(save_path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# count # of param for a list of module
def count_param(module_list):
    return sum(x.numel() for module in module_list for x in module.parameters()) / 10**6
    
# display the peak memory of cuda
def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")

def anal_tensor(tensor, name):
    sent = f" name: {name} mean: {tensor.mean().item()}  std: {tensor.std().item()}  min: {tensor.min().item()}  max: {tensor.max().item()}"
    print(sent)


def split(lst, split_nbr):
    div = len(lst) // split_nbr
    rest = len(lst) % split_nbr
    results = []
    start, end = 0, div
    while start < len(lst):
        if rest >= 1:
            end += 1
            rest -= 1
        results.append(lst[start:end])
        start, end = end, end+div
    return results

def chunk(iterable, chunk_size):
    ret = []
    for record in iterable:
        ret.append(record)
        if len(ret) == chunk_size:
            yield ret
            ret = []
    if ret:
        yield ret

def image_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def image_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst
