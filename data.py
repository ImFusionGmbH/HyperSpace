import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.ndimage import map_coordinates
import time
from pathlib import Path
from typing import Optional

import imfusion

from utils import *

resample_yaml_text = """
- Resample:
    resolution: [s0, s1, s2]
"""

bakeTransfo_yaml_text = """
- BakeTransformation
"""

markaslbl_yaml_text = """
- MarkAsLabel
- MarkAsTarget
- ReplaceValues:
        old_values : [38, 52, 82, 88, 164, 205, 244]
        new_values : [1, 2, 3, 4, 5, 6, 7]
"""


class ImFusionDataset(Dataset):
    """
    Standard ImFusion dataset
    Parses a data list file which is a tab separated csv file in the form:
    #datafield0    datafield1    ...    datafieldN
    file_patient0_0    file_patient0_1    file_patient0_N
    ...
    file_patientM_0    file_patientM_1    file_patientM_N

    if you are not an ImFusion user, implement you own standard dataset

    Args:
        data_file (str): path to data list file
        fields (dict): dictionary with 2 entries ('image' and 'label') respectively indicating which datafield should be considered for the network input and label maps
        max_lbl (int): maximum value of the label map (i.e. number of label classes if labels are incrementally set up)
        crop_size (tuple): size of cropping for batch creation
    """

    def __init__(self, data_file: str, fields: dict[str, list[str]], max_lbl: int=4, crop_size: tuple=tuple()) -> None:
        self.crop_size = crop_size
        self.data_dim = len(crop_size)

        self.parent_folder = "/".join(data_file.split("/")[:-1]) + "/"
        self.one_hot = np.eye(max_lbl + 1)

        if len(fields["image"])!=1:
            wrong_fields = fields["image"]
            raise ValueError(f"Current implementation only supports exactly one image as input, more than one image field configured: {wrong_fields}.")
        
        if len(fields["label"])!=1:
            wrong_fields = fields["label"]
            raise ValueError(f"Current implementation only supports exactly one label map, more than one label field configured: {wrong_fields}.")

        with open(data_file) as f:
            data_list = f.read()
            data_list = data_list.split("\n")
        data_list[0] = data_list[0][1:]  # Remove first hashtag
        data_list = [x.split("\t") for x in data_list if x != ""]
        data_list = np.array(data_list)
        
        columns = data_list[0]
        idx_im = [x in fields["image"] for x in columns]
        idx_lbl = [x in fields["label"] for x in columns]

        self.im_files = data_list[1:, idx_im]
        self.lbl_files = data_list[1:, idx_lbl]
        self.n_images = self.im_files.shape[1]
        self.n_labels = self.lbl_files.shape[1]

        self.lbl_files = self.lbl_files[self.im_files[:, 0] != "null", :]
        self.im_files = self.im_files[self.im_files[:, 0] != "null", :]

    def process_input(self, im: SharedImageSet) -> np.ndarray:
        """
        Perform basic pre-processing on input

        Args:
            im (SharedImageSet): input
        """
        im = np.array(im)
        im = im[:, :, :, :, 0]
        im = self.normalize(im)
        return im
    
    def process_lbl(self, lbl: SharedImageSet) -> np.ndarray:
        """
        Perform basic pre-processing on label map

        Args:
            lbl (SharedImageSet): label map
        """
        return np.array(lbl)[0, ..., 0].astype(int)

    def random_crop(self, im: np.ndarray, lbl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        extract random crop from an image and corresponding label map

        Args:
            im (np.ndarray): input image
            lbl (np.ndarray): corresponding label map
        """
        # Use set up crop size or ensure at least 4 downsampling are applicable
        crop_size = self.crop_size if self.crop_size[0]>0 else [(x//16+(1*(x%16)!=0)) * 16 for x in im.shape]
        for dim in range(self.data_dim):
            if im.shape[dim] <= crop_size[dim]:
                n_pad = crop_size[dim] - im.shape[dim]
                pad_left = n_pad // 2
                pad_right = n_pad - pad_left
                padding = [[0, 0] for k in range(self.data_dim)] 
                padding[dim][0] = pad_left
                padding[dim][1] = pad_right
                im = np.pad(im, padding)
                lbl = np.pad(lbl, padding)
                idx = 0
            else:
                idx = np.random.randint(0, high=im.shape[dim] - crop_size[dim])
            if dim==0:
                im = im[idx:idx+crop_size[dim]]
                lbl = lbl[idx:idx+crop_size[dim]]
            elif dim==1:
                im = im[:, idx:idx+crop_size[dim]]
                lbl = lbl[:, idx:idx+crop_size[dim]]
            else:
                im = im[:, :, idx:idx+crop_size[dim]]
                lbl = lbl[:, :, idx:idx+crop_size[dim]]
        return im, lbl

    def normalize(self, im: np.ndarray) -> np.ndarray:
        """
        Applies N(0, 1) normalization to im

        Args:
            im (np.ndarray): image to be normalized 
        """
        axes = (1, 2, 3) if self.data_dim == 3 else (1, 2)
        im = (im - im.mean(axis=axes, keepdims=True)) / (
            (im - im.mean(axis=axes, keepdims=True))**2 + 1e-8).mean(axis=axes, keepdims=True)**(0.5)
        return im

    def __len__(self) -> int:
        return len(self.im_files)

    def __getitem__(self, idx: int) -> dict:
        im = [self.process_input(imfusion.io.open(self.im_files[idx, k])[0]) for k in range(self.n_images)]
        lbl = [self.process_input(imfusion.io.open(self.lbl_files[idx, k])[1]) for k in range(self.n_labels)]

        im, lbl = self.random_crop(im[0], lbl[0])
        im = self.normalize(im)
        lbl = self.one_hot[lbl].transpose(3, 0, 1, 2) if self.data_dim == 3 else self.one_hot[lbl].transpose(2, 0, 1)
        return {"im": im.astype(np.float32), "lbl": lbl.astype(np.float32)}

    def load_idx(self, idx: int) -> tuple[SharedImageSet, SharedImageSet]:
        """
        Load the element from the dataset corresponding to the index idx

        Args:
            idx (int): index of the dataset element to be loaded 
        """
        im = imfusion.io.open(self.im_files[idx, 0])[0]
        lbl = imfusion.io.open(self.lbl_files[idx, 0])[0]
        return im, lbl

    def get_resampled_item(self, idx: int, new_spacing: tuple[float, float, float], intermediate_spacing: Optional[tuple[float, float, float]]=None) -> dict:
        """
        Load the element from the dataset corresponding to the index idx, resample it to new_spacing
        and apply the basic preprocessing to it

        Args:
            idx (int): index of the dataset element to be loaded
            new_spacing (tuple): target spacing 
            intermediate_spacing (tuple): If not None, first resample the image to this spacing before resampling it to new_spacing
        """
        res = {}

        im, lbl = self.load_idx(idx)

        if intermediate_spacing is not None:
            resampling_yaml = resample_yaml_text.replace("s0", str(intermediate_spacing[0])).replace("s1", str(intermediate_spacing[1])).replace("s2", str(intermediate_spacing[2]))
            im = imfusion.executeAlgorithm("Operations Sequence", [im], {"yamlText": resampling_yaml})[0]
            lbl = imfusion.executeAlgorithm("Operations Sequence", [lbl], {"yamlText": resampling_yaml})[0]

        resampling_yaml = resample_yaml_text.replace("s0", str(new_spacing[0])).replace("s1", str(new_spacing[1])).replace(
            "s2", str(new_spacing[2]))
        im = imfusion.executeAlgorithm("Operations Sequence", [im], {"yamlText": resampling_yaml})[0]
        lbl = imfusion.executeAlgorithm("Operations Sequence", [lbl], {"yamlText": resampling_yaml})[0]

        res["im_imf"] = im
        res["lbl_imf"] = lbl

        im = self.process_input(im)
        lbl = self.process_lbl(lbl)

        im, lbl = self.random_crop(im[0], lbl)
        im = im[None, ...]
        lbl = self.one_hot[lbl].transpose(3, 0, 1, 2) if self.data_dim == 3 else self.one_hot[lbl].transpose(2, 0, 1)

        return {
            "im": im.astype(np.float32),
            "lbl": lbl.astype(np.float32),
            "spacing": np.array(new_spacing).astype(np.float32)
        } | res

    def get_item(self, idx: int) -> dict: 
        """
        Load the element from the dataset corresponding to the index idx
        and apply the basic preprocessing to it
        
        Args:
            idx (int): index of the dataset element to be loaded 
        """
        im, lbl = self.load_idx(idx)

        res = {}
        res["im_imf"] = im
        res["lbl_imf"] = lbl

        spacing = np.array(im[0].spacing)

        im = self.process_input(im)
        lbl = self.process_lbl(lbl)

        im, lbl = self.random_crop(im[0], lbl)
        im = im[None, ...]

        lbl = self.one_hot[lbl].transpose(3, 0, 1, 2) if self.data_dim == 3 else self.one_hot[lbl].transpose(2, 0, 1)

        return {
            "im": im.astype(np.float32), "lbl": lbl.astype(np.float32), "spacing": np.array(spacing).astype(np.float32)
        } | res


class BRATSDataset(ImFusionDataset):
    """
    Dataset used for the BRATS experiments

    Args:
        data_file (str): path to data list file
        fields (dict): dictionary with 2 entries ('image' and 'label') respectively indicating which datafield should be considered for the network input and label maps
        max_lbl (int): maximum value of the label map (i.e. number of label classes if labels are incrementally set up)
        crop_size (tuple): size of cropping for batch creation
        max_spacing_scaling (float): maximum spacing scaling for spacing augmentation
        baseline (bool): if True, always resample back to a constant, high, voxel resolution
    """

    def __init__(self, data_file: str, fields: dict[str, list[str]], max_lbl: int=4, crop_size: tuple=tuple(), max_spacing_scaling: float=3, baseline: bool=False) -> None:
        super().__init__(data_file, fields, max_lbl, crop_size)
        self.max_spacing_scaling = max_spacing_scaling
        self.baseline = baseline
        self.data_root: Path = Path(data_file).parent

    def load_idx(self, idx: int) -> tuple[SharedImageSet, SharedImageSet]:
        """
        Load the element from the dataset corresponding to the index idx

        Args:
            idx (int): index of the dataset element to be loaded 
        """
        im = imfusion.io.open((self.data_root / self.im_files[idx, 0]).as_posix())[0]
        lbl = imfusion.io.open((self.data_root / self.lbl_files[idx, 0]).as_posix())[0]
        lbl = convert_to_uint(lbl)
        lbl = imfusion.machinelearning.SetLabelModalityOperation([str(x) for x in range(1, 4)]).process_images(lbl)
        # Binary segmentation of the tumor core
        lbl = imfusion.machinelearning.ReplaceLabelsValuesOperation([2, 4], [0, 1]).process_images(lbl)
        lbl = imfusion.executeAlgorithm("Image Resampling", [lbl, im], {"interpolation": "Nearest"})[0]
        return im, lbl

    def __getitem__(self, idx: int) -> dict:
        im, lbl = self.load_idx(idx)

        spacing = np.array(im[0].spacing)
        new_spacing = []
        for d in range(3):
            scaling = 1 + np.random.rand() * (self.max_spacing_scaling - 1)
            new_spacing.append(spacing[d] * scaling)

        resampling_yaml = resample_yaml_text.replace("s0", str(new_spacing[0])).replace("s1", str(new_spacing[1])).replace(
            "s2", str(new_spacing[2]))
        im = imfusion.executeAlgorithm("Operations Sequence", [im], {"yamlText": resampling_yaml})[0]
        lbl = imfusion.executeAlgorithm("Operations Sequence", [lbl], {"yamlText": resampling_yaml})[0]

        if self.baseline:
            resampling_yaml = resample_yaml_text.replace("s0", "1.0").replace("s1", "1.0").replace("s2", "1.0")
            im = imfusion.executeAlgorithm("Operations Sequence", [im], {"yamlText": resampling_yaml})[0]
            lbl = imfusion.executeAlgorithm("Operations Sequence", [lbl], {"yamlText": resampling_yaml})[0]

        im = self.process_input(im)
        lbl = self.process_lbl(lbl)
        im, lbl = self.random_crop(im[0], lbl)
        im = im[None, ...]
        lbl = self.one_hot[lbl].transpose(3, 0, 1, 2) if self.data_dim == 3 else self.one_hot[lbl].transpose(2, 0, 1)

        return {
            "im": im.astype(np.float32),
            "lbl": lbl.astype(np.float32),
            "spacing": np.array(new_spacing).astype(np.float32)
        }


class SpineMRIDataset(ImFusionDataset):
    """
    Dataset used for the Spine experiments

    Args:
        data_file (str): path to data list file
        fields (dict): dictionary with 2 entries ('image' and 'label') respectively indicating which datafield should be considered for the network input and label maps
        max_lbl (int): maximum value of the label map (i.e. number of label classes if labels are incrementally set up)
        crop_size (tuple): size of cropping for batch creation
        max_spacing_scaling (float): maximum spacing scaling for spacing augmentation
        baseline (bool): if True, always resample back to a constant, high, voxel resolution
    """
    def __init__(self, data_file, fields, max_lbl=4, crop_size=tuple(), baseline=False):
        super().__init__(data_file, fields, max_lbl, crop_size)
        self.baseline = baseline


    def sample_spacing(self) -> list[float]:
        """
        Sample a random voxel resolution for spacing augmentation
        """
        new_spacing = []
        new_spacing.append(np.random.rand() * 4 + 1.0)
        new_spacing.append(np.random.rand() * 1.3 + 0.2)
        new_spacing.append(np.random.rand() * 1.3 + 0.2)
        return new_spacing

    def load_idx(self, idx: int) -> tuple[SharedImageSet, SharedImageSet]:
        """
        Load the element from the dataset corresponding to the index idx

        Args:
            idx (int): index of the dataset element to be loaded 
        """
        im = imfusion.io.open(self.parent_folder + self.im_files[idx, 0].split("\\")[-1])[0]
        lbl = imfusion.io.open(self.parent_folder + self.lbl_files[idx, 0].split("\\")[-1])[0]

        lbl = convert_to_uint(lbl)
        lbl = imfusion.machinelearning.SetLabelModalityOperation([str(x) for x in range(1, 4)]).process_images(lbl)
        lbl = imfusion.machinelearning.ReplaceLabelsValuesOperation(list(range(1, 27)),
                                                                    [1] * 25 + [2]).process_images(lbl)

        lbl = imfusion.executeAlgorithm("Image Resampling", [lbl, im], {"interpolation": "Nearest"})[0]
        im, lbl = imfusion.executeAlgorithm("Operations Sequence", [lbl, im], {"yamlText": bakeTransfo_yaml_text})
        return im, lbl

    def __getitem__(self, idx: int) -> dict:
        im, lbl = self.load_idx(idx)

        new_spacing = self.sample_spacing()

        resampling_yaml = resample_yaml_text.replace("s0", str(new_spacing[0])).replace("s1", str(new_spacing[1])).replace(
            "s2", str(new_spacing[2]))
        im = imfusion.executeAlgorithm("Operations Sequence", [im], {"yamlText": resampling_yaml})[0]
        lbl = imfusion.executeAlgorithm("Operations Sequence", [lbl], {"yamlText": resampling_yaml})[0]

        if self.baseline:
            resampling_yaml = resample_yaml_text.replace("s0", "3.3").replace("s1", "0.6").replace("s2", "0.6")
            im = imfusion.executeAlgorithm("Operations Sequence", [im], {"yamlText": resampling_yaml})[0]
            lbl = imfusion.executeAlgorithm("Operations Sequence", [lbl], {"yamlText": resampling_yaml})[0]

        im = self.process_input(im)
        lbl = self.process_lbl(lbl)
        im, lbl = self.random_crop(im[0], lbl)
        im = im[None, ...]
        lbl = self.one_hot[lbl].transpose(3, 0, 1, 2) if self.data_dim == 3 else self.one_hot[lbl].transpose(2, 0, 1)

        return {"im": im.astype(np.float32), "lbl": lbl.astype(np.float32), "spacing": np.array(new_spacing).astype(np.float32)}


class CardiacMRDataset(ImFusionDataset):
    """
    Dataset used for the Cardiac experiments

    Args:
        data_file (str): path to data list file
        fields (dict): dictionary with 2 entries ('image' and 'label') respectively indicating which datafield should be considered for the network input and label maps
        max_lbl (int): maximum value of the label map (i.e. number of label classes if labels are incrementally set up)
        crop_size (tuple): size of cropping for batch creation
        max_spacing_scaling (float): maximum spacing scaling for spacing augmentation
        baseline (bool): if True, always resample back to a constant, high, voxel resolution
    """

    def __init__(self,
                 data_file,
                 fields,
                 max_lbl=4,
                 crop_size=tuple(),
                 max_spacing_scaling=3,
                 baseline=False):
        super().__init__(data_file, fields, max_lbl, crop_size)
        self.max_spacing_scaling = max_spacing_scaling if isinstance(max_spacing_scaling,
                                                                     list) else [max_spacing_scaling] * 3
        self.baseline = baseline

    def sample_spacing(self,) -> list[float]:
        """
        Sample a random voxel resolution for spacing augmentation
        """
        new_spacing = []
        for d in range(3):
            new_spacing.append(np.random.rand() * 3 + 0.5)
        return new_spacing

    def __getitem__(self, idx: int) -> dict:
        im, lbl = self.load_idx(idx)
        spacing = np.array(im[0].spacing)

        new_spacing = self.sample_spacing()

        resampling_yaml = resample_yaml_text.replace("s0", str(new_spacing[0])).replace("s1", str(new_spacing[1])).replace(
            "s2", str(new_spacing[2]))
        im = imfusion.executeAlgorithm("Operations Sequence", [im], {"yamlText": resampling_yaml})[0]
        lbl = imfusion.executeAlgorithm("Operations Sequence", [lbl], {"yamlText": resampling_yaml})[0]

        if self.baseline:
            resampling_yaml = resample_yaml_text.replace("s0", "0.9").replace("s1", "0.9").replace("s2", "1.2")
            im = imfusion.executeAlgorithm("Operations Sequence", [im], {"yamlText": resampling_yaml})[0]
            lbl = imfusion.executeAlgorithm("Operations Sequence", [lbl], {"yamlText": resampling_yaml})[0]

        im = self.process_input(im)
        lbl = self.process_lbl(lbl)

        im, lbl = self.random_crop(im[0], lbl)
        im = im[None, ...]

        lbl = self.one_hot[lbl].transpose(3, 0, 1, 2) if self.data_dim == 3 else self.one_hot[lbl].transpose(2, 0, 1)

        return {
            "im": im.astype(np.float32),
            "lbl": lbl.astype(np.float32),
            "spacing": np.array(new_spacing).astype(np.float32)
        }

    def load_idx(self, idx: int) -> tuple[SharedImageSet, SharedImageSet]:
        """
        Load the element from the dataset corresponding to the index idx

        Args:
            idx (int): index of the dataset element to be loaded 
        """
        im = imfusion.io.open(self.im_files[idx, 0])[0]
        lbl = imfusion.io.open(self.lbl_files[idx, 0])[0]

        lbl = convert_to_uint(lbl)
        lbl = imfusion.machinelearning.SetLabelModalityOperation([str(x) for x in range(1, 8)]).process_images(lbl)
        lbl = imfusion.machinelearning.ReplaceLabelsValuesOperation([38, 52, 82, 88, 164, 205, 244, 165],
                                                                    [1, 2, 3, 4, 5, 6, 7, 8]).process_images(lbl)

        lbl = imfusion.executeAlgorithm("Image Resampling", [lbl, im])[0]

        return im, lbl

