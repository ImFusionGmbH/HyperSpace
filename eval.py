import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import argparse
import json
import pandas as pd
import seaborn as sns
from typing import Optional

from utils import *
from data import *
from models import *
import time

import imfusion
imfusion.init()
from imfusion.machinelearning import DiceMetric




def evaluate_batch(batch: dict, model: nn.Module, path_save: Optional[str]=None, patch_size: list=[128, 128, 128]) -> tuple[dict, float, float, float]:
    """
    Runs evaluation on one batch
    Returns the Dice metric, the cpu time, gpu time, gpu memory consumption associated with the evaluation

    Args:
        batch (dict): batch
        model (nn.Module): model
        path_save (str): if not None, path to imf file to be saved
        patch_size (list): patch size used for evaluation, should be the same as the crop_size used during training
    """
    torch.cuda.reset_peak_memory_stats()
    start_cuda = torch.cuda.Event(enable_timing=True)
    end_cuda = torch.cuda.Event(enable_timing=True)

    start_cpu = time.time()
    # measure CUDA time
    start_cuda.record()
    
    im_sis_resampled = batch["im_imf"]
    lbl_sis_resampled = batch["lbl_imf"]

    data_np_resampled = batch["im"][None, :]
    lbl_np_resampled = batch["lbl"][None, :]
    spacing_resampled = batch["spacing"][None, :]

    data_torch_resampled = torch.tensor(data_np_resampled).float().cuda()
    spacing_torch_resampled = torch.tensor(spacing_resampled).float().cuda()

    # pred_resampled = model(data_torch_resampled, spacing_torch_resampled)
    pred_resampled = patch_based_inference(data_torch_resampled, spacing_torch_resampled, model, patch_size)
    pred_np_resampled = torch.argmax(pred_resampled, dim=1, keepdim=True).permute(0, 2, 3, 4, 1).detach().cpu().numpy().astype(np.uint8)
    pred_sis_resampled = resample_to_img(pred_np_resampled, lbl_sis_resampled, im_sis_resampled)

    dice = DiceMetric().compute_dice(pred_sis_resampled, lbl_sis_resampled)
    if path_save is not None:
        lbl_torch_resampled = torch.tensor(lbl_np_resampled).float().cuda()
        save_case(data_torch_resampled, pred_resampled, lbl_torch_resampled, path_save, spacing=spacing_resampled)

    end_cuda.record() # schedules the measurement event
    torch.cuda.synchronize() # cuda events are processed in order
    end_cpu = time.time()
    mem_usage = torch.cuda.memory_stats()

    return dice, end_cpu-start_cpu, start_cuda.elapsed_time(end_cuda), mem_usage["allocated_bytes.all.peak"]


def log_eval_metrics(log, dice, cpu_time, gpu_time, mem_usage, log_name):
    mean_dice = 0
    for i in dice[0].keys():
        log["resolution"].append(f"{s[0]:.2f}_{s[1]:.2f}_{s[2]:.2f}")
        log["dice"].append(dice[0][i])
        log["label"].append(i)
        log["model"].append(log_name)
        log["cpu_time"].append(cpu_time)
        log["gpu_time"].append(gpu_time)
        log["peak_gpu_memory"].append(mem_usage)
        mean_dice += dice[0][i]
    log["resolution"].append(f"{s[0]:.2f}_{s[1]:.2f}_{s[2]:.2f}")
    log["dice"].append(mean_dice / len(list(dice[0].keys())))
    log["label"].append(-1)
    log["model"].append(log_name)
    log["cpu_time"].append(cpu_time)
    log["gpu_time"].append(gpu_time)
    log["peak_gpu_memory"].append(mem_usage)
    return log



def evaluate(dataset: dict, model_dict: dict, path_save: str, fixed_res, interval_s, max_res=[3.5, 3.5, 3.5], patch_size=(128, 128, 128)):
    """
    Run evaluation

    Args:
        dataset (Dataset): evaluation dataset
        model_dict (dict): model dictionary in the form
                            {
                                input_res_type: [model_name, model]
                            }
        path_save (str): path to result folder
        fixed_res (list): fixed resolution at which FS is trained
        interval_s (list): list of spacing for evaluation
        max_res (list): maximum resolution of the training resolution interval
        patch_size (list): patch size used for evaluation, should be the same as the crop_size used during training
    """

    log = {
        "resolution": [],
        "label": [],
        "dice": [],
        "model": [],
        "cpu_time": [],
        "gpu_time": [],
        "peak_gpu_memory": []
    }

    with torch.no_grad():

        for i_s, s in tqdm(enumerate(interval_s)):
            for n in tqdm(range(len(dataset))):

                path_im_n = os.path.join(path_save, str(n))
                makedir(path_im_n)

                if np.any(np.array(s)>=np.array(max_res)):
                    max_res_ = [min(s_, mr_) for s_, mr_ in zip(s, max_res)]
                    batch = dataset.get_resampled_item(n, max_res_, s)
                    for model_name, model in model_dict["variable spacing"]:
                        dice, cpu_time, gpu_time, mem_usage = evaluate_batch(batch, model, path_save=None, patch_size=patch_size)
                        log = log_eval_metrics(log, dice, cpu_time, gpu_time, mem_usage, model_name+"_MaxRes")
        

                batch = dataset.get_resampled_item(n, s)
                for model_name, model in model_dict["variable spacing"]:
                    dice, cpu_time, gpu_time, mem_usage = evaluate_batch(batch, model, path_save=None, patch_size=patch_size)
                    log = log_eval_metrics(log, dice, cpu_time, gpu_time, mem_usage, model_name)
                
                batch_fixed = dataset.get_resampled_item(n, fixed_res, s)
                for model_name, model in model_dict["fixed spacing"]:
                    dice, cpu_time, gpu_time, mem_usage = evaluate_batch(batch, model, path_save=None, patch_size=patch_size)
                    log = log_eval_metrics(log, dice, cpu_time, gpu_time, mem_usage, model_name)

    makedir(path_save)

    with open(os.path.join(path_save, "results.json"), "w") as f:
        json.dump(log, f)

    log_pd = pd.DataFrame(log)
    log_pd["MaxRes"] = log_pd["model"].apply(lambda x: "MaxRes" in x)
    log_pd["log_cpu_time"] = log_pd["cpu_time"].apply(lambda x: np.log(x))
    log_pd["log_gpu_time"] = log_pd["gpu_time"].apply(lambda x: np.log(x))
    plt.figure(figsize=(15, 15))
    sns.lineplot(data=log_pd[(log_pd.label==-1)], x="resolution", y="dice", hue="model", style="MaxRes")  #, x_order=[tuple(s) for s in interval_s])
    plt.xticks(rotation=80)
    plt.savefig(os.path.join(path_save, "mean_dice_per_model.png"))

    models = np.unique(log["model"])
    fig, axes = plt.subplots(len(models), 1, figsize=(10, 10*len(models)))

    for i, m in enumerate(models):
        sns.lineplot(data=log_pd[(log_pd.model==m)&(log_pd.label!=-1)], x="resolution", y="dice", hue="label", ax=axes[i])  #, x_order=[tuple(s) for s in interval_s])
        axes[i].tick_params(axis='x', rotation=80)
        axes[i].set_title(m)
    plt.savefig(os.path.join(path_save, "dice_per_labels_per_model.png"))

    plt.figure(figsize=(15, 15))
    sns.lineplot(data=log_pd[(log_pd.label==-1)], x="resolution", y="cpu_time", hue="model", style="MaxRes")  #, x_order=[tuple(s) for s in interval_s])
    plt.xticks(rotation=80)
    plt.yscale('log')
    plt.savefig(os.path.join(path_save, "mean_cpu_time_per_model.png"))

    plt.figure(figsize=(15, 15))
    sns.lineplot(data=log_pd[(log_pd.label==-1)], x="resolution", y="gpu_time", hue="model", style="MaxRes")  #, x_order=[tuple(s) for s in interval_s])
    plt.yscale('log')
    plt.xticks(rotation=80)
    plt.savefig(os.path.join(path_save, "mean_gpu_time_per_model.png"))

    plt.figure(figsize=(15, 15))
    sns.lineplot(data=log_pd[(log_pd.label==-1)], x="resolution", y="peak_gpu_memory", hue="model", style="MaxRes")  #, x_order=[tuple(s) for s in interval_s])
    plt.xticks(rotation=80)
    plt.savefig(os.path.join(path_save, "mean_peak_gpu_memory_per_model.png"))
    return 



def native_eval(dataset: Dataset, model_dict: dict, path_save, fixed_res, max_lbl=1, n_down=3, layers=[3, 10, 10, 50], max_res=[3.5, 3.5, 3.5], patch_size=(128, 128, 128)):
    """
    Run evaluation at native resolution

    Args:
        dataset (Dataset): evaluation dataset
        model_dict (dict): model dictionary in the form
                            {
                                input_res_type: [model_name, model]
                            }
        path_save (str): path to result folder
        fixed_res (list): fixed resolution at which FS is trained
        max_res (list): maximum resolution of the training resolution interval
        patch_size (list): patch size used for evaluation, should be the same as the crop_size used during training
    """

    log = {
        "resolution": [],
        "label": [],
        "dice": [],
        "model": [],
        "cpu_time": [],
        "gpu_time": [],
        "peak_gpu_memory": []
    }

    with torch.no_grad():

        for n in tqdm(range(len(dataset))):

            batch = dataset.get_item(n)
            s = batch["spacing"]
            for model_name, model in model_dict["variable spacing"]:
                dice, cpu_time, gpu_time, mem_usage = evaluate_batch(batch, model, path_save=None, patch_size=patch_size)
                log = log_eval_metrics(log, dice, cpu_time, gpu_time, mem_usage, model_name)
            
            batch_fixed = dataset.get_resampled_item(n, fixed_res, s)
            for model_name, model in model_dict["fixed spacing"]:
                dice, cpu_time, gpu_time, mem_usage = evaluate_batch(batch, model, path_save=None, patch_size=patch_size)
                log = log_eval_metrics(log, dice, cpu_time, gpu_time, mem_usage, model_name)

    makedir(path_save)

    with open(os.path.join(path_save, "results.json"), "w") as f:
        json.dump(log, f)

    return 





if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="BRATS", help="One of: [BRATS, Spine]")
    parser.add_argument("-m", "--model", type=str, default="HyperSpace", help="One of: [HyperSpace, AS, FS]")
    parser.add_argument("--suffix", type=str, default="", help="Suffix added to experiment folder.")
    parser.add_argument("--data_file_val", "-dfv", type=str, help="Path to validation data list file.")
    parser.add_argument("--checkpoint_HS", "-chs", type=str, help="Path to checkpoint of HyperSpace.")
    parser.add_argument("--checkpoint_AS", "-cas", type=str, help="Path to checkpoint of the AS baseline.")
    parser.add_argument("--checkpoint_FS", "-cfs", type=str, help="Path to checkpoint of the FS baseline.")
    args, placeholders = parser.parse_known_args()

    dataset = args.dataset
    model_type = args.model
    suffix = args.suffix
    data_file_val = args.data_file_val
    path_hyperunet = args.checkpoint_HS
    path_unet = args.checkpoint_FS
    path_unet_augm = args.checkpoint_AS


    path_save = f"Evaluation/{dataset}{model_type}{suffix}"

    if dataset=="CardiacMR":
        print("Running Cardiac evaluation...")

        fields = {
            "image": ["ImagePath"],
            "label": ["LabelPath"]
        }

        max_lbl = 8
        n_down = 3
        layers = [3, 10, 10, 50]

        path_folder = os.path.join(path_save, "CardiacMR")
        fixed_res = [0.9, 0.9, 1.2]

        interval_s = [[s1, s2, s3] for s1 in np.linspace(0.7, 5, 44) for s2 in np.linspace(0.7, 5, 44) for s3 in np.linspace(0.7, 5, 44)]
        max_res = [3.4, 3.4, 3.4]

        dataset_val = CardiacMRDataset(data_file_val, fields, max_lbl=8, crop_size=None, max_spacing_scaling=2)

    elif dataset=="BRATS":
        print("Running BRATS evaluation...")

        fields = {
            "image": ["T1CPath"],
            "label": ["labelPath"]
        }

        max_lbl = 1
        n_down = 3
        
        path_folder = os.path.join(path_save, "BRATS")
        fixed_res = [1.0, 1.0, 1.0]
        interval_s = [[np.random.rand()*3 + 0.5, np.random.rand()*3 + 0.5, np.random.rand()*3 + 0.5] for k in range(100)]
        max_res = [3.4, 3.4, 3.4]

        dataset_val = BRATSDataset(data_file_val, fields, max_lbl=max_lbl, crop_size=(-1, -1, -1))

    elif dataset=="Spine":
        print("Running Spine evaluation...")

        fields = {
            "image": ["dataPath"],
            "label": ["labelPath"]
        }

        max_lbl = 2
        n_down = 4
        layers = [3, 10, 10, 20]

        path_folder = os.path.join(path_save, "Spine")
        fixed_res = [1, 0.3, 0.3]
        interval_s = [[np.random.rand()*0.5 + 3, np.random.rand()*0.4 + 0.4, np.random.rand()*0.4 + 0.4] for k in range(100)]
        max_res = [4.9, 1.4, 1.4]
        patch_size = (256, 128, 48)

        dataset_val = SpineMRIDataset(data_file_val, fields, max_lbl=max_lbl, crop_size=(-1, -1, -1))

    hyperunet = HyperUnet(layers, in_c=1, out_c=max_lbl+1, n_down=n_down, n_fix=3, C=16, n_dim=3).cuda()
    hyperunet.load_state_dict(torch.load(path_hyperunet)["model"])
    hyperunet.eval()

    unet = UNet(in_c=1, out_c=max_lbl+1, n_down=n_down, n_fix=3, C=16, n_dim=3).cuda()
    unet.load_state_dict(torch.load(path_unet)["model"])
    unet.eval()

    unet_augm = UNet(in_c=1, out_c=max_lbl+1, n_down=n_down, n_fix=3, C=16, n_dim=3).cuda()
    unet_augm.load_state_dict(torch.load(path_unet_augm)["model"])
    unet_augm.eval()

    model_dict = {
        "fixed spacing": [
            ["FS", unet]
        ],
        "variable spacing": [
            ["AS", unet_augm],
            ["FSNR", unet],
            ["HyperSpace", hyperunet]
        ]
    }
    evaluate(dataset_val, model_dict, path_folder, fixed_res, interval_s, max_res=max_res)

    



