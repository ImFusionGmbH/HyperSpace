import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from typing import Optional, Union

import imfusion
imfusion.init()
from imfusion import SharedImageSet


def write_image(im: np.ndarray, path: str) -> None:
    """
    Saves numpy array as a SharedImageSet

    Args:
        im (np.ndarray): Array to be saved, should have the following shape (1, X, Y, Z, C)
        path (str): file name of the saved SharedImageSet (.imf file)
    """
    im = SharedImageSet(im)
    imfusion.io.write([im], path)


def resample_to_img(data_np: np.ndarray, template_sis: SharedImageSet, resampling_target: SharedImageSet) -> SharedImageSet:
    """
    Resample data_np to resampling_target
    This assumes that data_np is at least as large as the template_sis
    If it is larger, extracts a center crop of data_np that matches the template_sis shape

    Args:
        data_np (np.ndarray): data to be resampled
        template_sis (SharedImageSet): template SharedImageSet containing the meta data corresponding to data_np 
        resampling_target (SharedImageSet): SharedImageSet the data_np should be resampled to
    """
    templated = template_sis.clone()
    shape = np.array(template_sis).shape
    shape_np = data_np.shape
    for d in range(1, 4):
        if shape[d]==shape_np[d]:
            continue
        l = (shape_np[d] - shape[d]) // 2
        r = (shape_np[d] - shape[d]) - l
        if d==1:
            data_np = data_np[:, l:-r]
        elif d==2:
            data_np = data_np[:, :, l:-r]
        elif d==3:
            data_np = data_np[:, :, :, l:-r]
    templated.assignArray(data_np)
    res = imfusion.executeAlgorithm("Image Resampling", [templated, resampling_target], {"interpolation": "Nearest"})[0]
    return res


def save_case(im: torch.Tensor, pred: torch.Tensor, lbl: torch.Tensor, path: str, spacing: Optional[np.ndarray]=None):
    """
    Save training or validation batch experiment

    Args:
        im (torch.tensor): Image to be saved
        pred (torch.tensor): Prediction to be saved
        lbl (torch.tensor): Label to be saved
        path (str): path to the training/validation file
        spacing (np.ndarray): Spacings of the underlying data
    """
    if len(im.size()) == 5:
        im = im.permute(0, 2, 3, 4, 1).detach().cpu().numpy()
        pred = torch.argmax(pred, dim=1, keepdims=True).permute(0, 2, 3, 4, 1).detach().cpu().numpy()
        lbl = torch.argmax(lbl, dim=1, keepdims=True).permute(0, 2, 3, 4, 1).detach().cpu().numpy()
    elif len(im.size()) == 4:
        im = im.permute(0, 2, 3, 1).detach().cpu().numpy()
        pred = torch.argmax(pred, dim=1, keepdims=True).permute(0, 2, 3, 1).detach().cpu().numpy()
        lbl = torch.argmax(lbl, dim=1, keepdims=True).permute(0, 2, 3, 1).detach().cpu().numpy()

    im = SharedImageSet(im)
    lbl = SharedImageSet(lbl.astype(np.uint8))
    pred = SharedImageSet(pred.astype(np.uint8))
    if spacing is not None:
        for i, s in enumerate(spacing):
            im[i].spacing = s
            lbl[i].spacing = s
            pred[i].spacing = s
    imfusion.io.write([im, lbl, pred], path)

def moving_average(data: Union[list, np.ndarray], k: int) -> Union[list, np.ndarray]:
    """
    computes a moving average of the data with a rolling window of size k

    Args:
        data (np.ndarray): data to be running averaged
        k (int): size of the rolling window
    """
    if len(data)<k+1:
        return data
    else:
        return [np.mean(data[i:i+k]) for i in range(len(data) - k)]

def block_average(data: Union[list, np.ndarray], k: int) -> Union[list, np.ndarray]:
    """
    computes a block average of the data with window of size k

    Args:
        data (np.ndarray): data to be block averaged
        k (int): size of the window
    """
    if len(data)<k:
        return data
    else:
        return [np.mean(data[i*k:(i+1)*k]) for i in range(len(data)//k)] + [np.mean(data[-k:])]


def plot_loss(loss_dict: dict, path: str, ma: int=100) -> None:
    """
    Plot training log

    Args:
    loss_dict (dict): log object in the form a dictionary as follow 
                        {
                            phase: {
                                loss_name : [list_values]
                            }
                        }
    path (str): path to image file for the plot
    ma (int): moving average window size
    """
    # Only plot varying losses
    n_losses_train = len([k for k in loss_dict.get("Train", {}).keys() if np.std(loss_dict.get("Train", {}).get(k, [0]))>0.00001])
    n_losses_val = len([k for k in loss_dict.get("Val", {}).keys() if np.std(loss_dict.get("Val", {}).get(k, [0]))>0.00001])
    n_losses = max(n_losses_train, n_losses_val)
    fig, axes = plt.subplots(2,
                                 n_losses,
                                 figsize=(15 * n_losses, 20))

    for i, phase in enumerate(["Train", "Val"]):
        idx = 0
        for loss_name in loss_dict.get(phase, {}):
            if np.std(loss_dict[phase][loss_name])<=0.00001:
                continue
            data = moving_average(loss_dict[phase][loss_name], ma)
            axes[i][idx].plot(np.arange(len(data)),
                                data)
            axes[i][idx].set_title(phase + ' ' + loss_name)
            idx += 1

    fig.savefig(path, bbox_inches='tight')
    plt.clf()



def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Dice loss

    Args:
        pred (torch.Tensor): prediction tensor
        target (torch.Tensor): target tensor
    """
    axes = list(range(2, len(pred.size())))
    return 1 - 2*(pred*target).sum(dim=axes) / (pred.sum(dim=axes)+target.sum(dim=axes)+1e-8)


def cross_entropy_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Cross Entropy loss

    Args:
        pred (torch.Tensor): prediction tensor
        target (torch.Tensor): target tensor
    """
    return F.cross_entropy(pred, target)


def log_dice_loss(log: dict, dice, phase, verbose=False, prefix=""):
    """
    Log dice loss in log

    Args:
        log (dict): log
        dice (torch.Tensor): Dice loss tensor
        phase (str): phase which the dice tensor originated from
        verbose (bool): if True, prints the mean Dice score per class
        predix (str): predix for the log entry (prepended to 'dice class_num')
    """
    n_labels = dice.size(1)
    dice_np = dice.detach().cpu().numpy()
    for n in range(n_labels):
        log[phase][prefix+"dice "+str(n)] = log[phase].get(prefix+"dice "+str(n), []) + list(dice_np[:, n])
    
    if verbose:
        print(", ".join([f"Dice {i}: {dice_np[:, i].mean()}" for i in range(n_labels)]))
    return log


def makedir(dir: str) -> None:
    """
    Create a directory if it does not exist

    Args:
        dir (str): directory to be created
    """
    if not os.path.isdir(dir):
        os.makedirs(dir)



def split_file(file_name: str, p_train: float, file_train: str, file_val: str) -> None:
    """
    Splits a data list file into training and validation data list files

    Args:
        file_name (str): data list file to be splitted
        p_train (float): proportion of the data list to be used for training
        file_train (str): path to training data list file
        file_val (str): path to validation data list file
    """
    with open(file_name, "r") as f:
        file_all = f.readlines()

    train_list = [file_all[0]]
    val_list = [file_all[0]]

    perm = np.random.permutation(len(file_all)-1)
    n_train = int((len(file_all) - 1) * p_train)

    for i, line in enumerate(file_all[1:]):
        if i in perm[:n_train]:
            train_list.append(line)
        else:
            val_list.append(line)

    train_list = "\n".join(train_list)
    val_list = "\n".join(val_list)

    with open(file_train, "w") as f:
        f.write(train_list)

    with open(file_val, "w") as f:
        f.write(val_list)


def convert_to_uint(im: SharedImageSet) -> SharedImageSet:
    """
    Changes image data type to unsigned char

    Args:
        im (SharedImageSet): SharedImageSet which data type needs to be changed
    """
    im_np = np.array(im)
    new_im = SharedImageSet(im_np.astype(np.uint8))
    new_im[0].spacing = im[0].spacing
    new_im[0].matrix = im[0].matrix
    return new_im



def make_size_divisible(im: torch.Tensor, div: int) -> tuple[torch.Tensor, slice, slice, slice]:
    """
    Pads an image to make its dimensions divible by div
    Returns the padded image and the slices to retrieve the original image content

    Args:
        im (torch.Tensor): image tensor
        div (int): desired dividor of the image dimensions
    """
    B, C, X, Y, Z = im.size()
    r_z = (Z % div)
    pad_z_m = (div - r_z) // 2 if r_z!=0 else 0
    pad_z_p = (div - r_z) - pad_z_m if r_z!=0 else 0
    r_y = Y % div
    pad_y_m = (div - r_y) // 2 if r_y!=0 else 0
    pad_y_p = (div - r_y) - pad_y_m if r_y!=0 else 0
    r_x = X % div
    pad_x_m = (div - r_x) // 2  if r_x!=0 else 0
    pad_x_p = (div - r_x) - pad_x_m  if r_x!=0 else 0
    im = F.pad(im, (pad_z_m, pad_z_p, pad_y_m, pad_y_p, pad_x_m, pad_x_p), mode="replicate")
    return im, slice(pad_x_m, -pad_x_p or X), slice(pad_y_m, -pad_y_p or Y), slice(pad_z_m, -pad_z_p or Z)



def patch_based_inference(im: torch.Tensor, spacing: torch.Tensor, model: nn.Module, patch_size: list):
    """
    Perform patch based inference on an image

    Args:
        im (torch.Tensor): input image
        spacing (torch.Tensor): image spacing
        model (torch.Tensor): model
        patch_size (list): patch_size, should correspond to the crop_size used during training
    """
    im, sx, sy, sz = make_size_divisible(im, 8)

    half_patch_size = [(patch_size[0] // 2), (patch_size[1] // 2), (patch_size[2] // 2)]
    B, C, X, Y, Z = im.size()

    if patch_size[0]>X:
        n_x = 1
    else:
        n_x = (X-half_patch_size[0]) // half_patch_size[0] 
        if (X-half_patch_size[0]) % half_patch_size[0]!=0:
            n_x += 1
    
    if patch_size[1]>Y:
        n_y = 1
    else:
        n_y = (Y-half_patch_size[1]) // half_patch_size[1] 
        if Y % half_patch_size[1]!=0:
            n_y += 1
    
    if patch_size[2]>Z:
        n_z = 1
    else:
        n_z = (Z - half_patch_size[2]) // half_patch_size[2] 
        if Z % half_patch_size[2]!=0:
            n_z += 1

    res = torch.zeros((B, model.out_c, X, Y, Z))

    patch_weight = torch.exp(-((torch.stack(torch.meshgrid(
                        torch.linspace(-1, 1, min(patch_size[0], X)),
                        torch.linspace(-1, 1, min(patch_size[1], Y)),
                        torch.linspace(-1, 1, min(patch_size[2], Z))
    ), dim=-1))**2).sum(-1) / (2.0))[None, None, ...].cuda()

    if hasattr(model, "hypernetwork"):
        model = model.get_unet(spacing)
        model.eval()

    with torch.no_grad():
        for x in range(n_x):
            x_start = x * half_patch_size[0] if (x+2) * half_patch_size[0]<=X else -patch_size[0]
            x_end = (x+2) * half_patch_size[0] if (x+2) * half_patch_size[0]<=X else X
            slice_x = slice(x_start, x_end)
            if patch_size[0]>X:
                slice_x = slice(0, X)
            for y in range(n_y):
                y_start = y * half_patch_size[1] if (y+2) * half_patch_size[1]<=Y else -patch_size[1]
                y_end = (y+2) * half_patch_size[1] if (y+2) * half_patch_size[1]<=Y else Y
                slice_y = slice(y_start, y_end)
                if patch_size[1]>Y:
                    slice_y = slice(0, Y)
                for z in range(n_z):
                    z_start = z * half_patch_size[2] if (z+2) * half_patch_size[2]<=Z else -patch_size[2]
                    z_end = (z+2) * half_patch_size[2] if (z+2) * half_patch_size[2]<=Z else Z
                    slice_z = slice(z_start, z_end)
                    if patch_size[2]>Z:
                        slice_z = slice(0, Z)
                    pred = model(im[:, :, slice_x, slice_y, slice_z].cuda()) * patch_weight
                    res[:, :, slice_x, slice_y, slice_z] += pred.cpu()

    return res[:, :, sx, sy, sz]
                




