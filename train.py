import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from typing import Optional

from tqdm import tqdm
import argparse

from utils import *
from data import *
from models import *

def train(
        dataloader_train: DataLoader, 
        dataloader_val: DataLoader, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        n_epoch: int, 
        save_dir: str, 
        device: str="cuda", 
        repeat: int=1) -> None:
    """
    Run the training loop for a model

    Args:
        dataloader_train: train dataloader
        dataloader_val: validation dataloader
        model: model to train
        optimizer: torch optimizer optmizing the model
        n_epoch: number of training epochs to run
        save_dir: Path to directory where results should be saved
        device: device on which to run the training (should be 'cuda')
        repeat: Number of training epochs between validations
    """

    # statistics
    log = {
        "Train" : {
            "EMA totloss" : [0]
        },
        "Val" : {
            "EMA totloss" : [0]
        }
    }

    log = validation(dataloader_val, model, save_dir, log, save=True, epoch=0)

    for e in range(n_epoch):
        print(f"Starting epoch {e}...\n")
        for it, (batch) in tqdm(enumerate(dataloader_train)):
            
            # zero out gradients
            optimizer.zero_grad()
        
            # load data
            im = batch["im"].to(device)
            lbl = batch["lbl"].to(device)
            spacing = batch["spacing"].to(device)

            b = lbl.size(0)
            
            pred_log = model(im, spacing)
            pred_prob = F.softmax(pred_log, dim=1)

            # loss
            loss_dice = dice_loss(pred_prob, lbl)
            loss_cc = cross_entropy_loss(pred_log, lbl)

            loss_avg = loss_dice.mean() + loss_cc.mean()
            
            #logs
            log = log_dice_loss(log, loss_dice, "Train", verbose=True, prefix="")
            log["Train"]["EMA totloss"] += [0.1 * loss_avg.item() + 0.9*log["Train"]["EMA totloss"][-1]]

            # backward + optimize
            loss_avg.backward()
            optimizer.step()

        if e%repeat==0:
            log = validation(dataloader_val, model, save_dir, log, save=e%50==0, epoch=e)
            path_model = os.path.join(save_dir, "training_results", "model_w.pt")
            torch.save(model.state_dict(), path_model)
            print(f"Saved model at: {path_model}")
            plot_loss(log, os.path.join(save_dir, "loss.png"))

            path_epoch = os.path.join(save_dir, "training_results", str(e)+"_training.imf")
            if not os.path.isdir(os.path.join(save_dir, "training_results")):
                os.makedirs(os.path.join(save_dir, "training_results"))
            save_case(im, pred_prob, lbl, path_epoch, spacing=spacing.detach().cpu().numpy())

            torch.save({"model":model.state_dict(), "optimizer": optimizer.state_dict()}, os.path.join(save_dir, "training_results", f"{e}_checkpoint.pt"))
            torch.save({"model":model.state_dict(), "optimizer": optimizer.state_dict()}, os.path.join(save_dir, "checkpoint.pt"))

    torch.save(model.state_dict(), os.path.join(save_dir, "model_w.pt"))

    return 




def validation(dataloader: DataLoader, model: nn.Module, save_dir: str, log: dict, save: bool=False, device: str="cuda", epoch: Optional[int]=None):
    """
    Run validation for a model

    Args:
        dataloader (DataLoader): validation dataloader
        model (nn.Module): model
        save_dir (str): Path where to save results
        log (dict): log dictionary
        save (bool): if True, saves the last validation image
        device (str): device on which to run the validation (should be 'cuda')
        epoch (int): if provided, add the epoch number to the validation file name
    """
    print("Running evaluation...\n")
    with torch.no_grad():
        model.eval()
        for it, batch in enumerate(dataloader):
                
            # load data
            im = batch["im"].to(device)
            lbl = batch["lbl"].to(device)
            spacing = batch["spacing"].to(device)

            pred_log = model(im, spacing)
            pred_prob = F.softmax(pred_log, dim=1)

            # loss
            loss = dice_loss(pred_prob, lbl)
            loss_avg = loss.mean()
            
            #logs
            log = log_dice_loss(log, loss, "Val")
            log["Val"]["EMA totloss"] += [0.1 * loss_avg.item() + 0.9*log["Val"]["EMA totloss"][-1]]

        if save:
            path_save_case = os.path.join(save_dir, "training_results", str(epoch)+"_validation.imf") if isinstance(epoch, int) else os.path.join(save_dir, "training_results", "validation.imf")
            if not os.path.isdir(os.path.join(save_dir, "training_results")):
                os.makedirs(os.path.join(save_dir, "training_results"))
            save_case(im, pred_prob, lbl, path_save_case, spacing=spacing.detach().cpu().numpy())

    model.train()
    return log


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="BRATS", help="One of: [BRATS, Spine]")
    parser.add_argument("-m", "--model", type=str, default="HyperSpace", help="One of: [HyperSpace, AS, FS]")
    parser.add_argument("--suffix", type=str, default="", help="Suffix added to experiment folder.")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to checkpoint to resume training.")
    parser.add_argument("--data_file_train", "-dft", type=str, help="Path to train data list file.")
    parser.add_argument("--data_file_val", "-dfv", type=str, help="Path to validation data list file.")
    
    args, placeholders = parser.parse_known_args()

    dataset = args.dataset
    model_type = args.model
    suffix = args.suffix
    checkpoint = args.checkpoint
    data_file_train = args.data_file_train
    data_file_val = args.data_file_val

    if dataset=="BRATS":
        fields = {
            "image": ["T1CPath"],
            "label": ["labelPath"]
        }

        max_lbl = 1
        n_down = 3
        baseline = model_type=="UNet"
        repeat = 1
        hn_layers = [3, 10, 10, 50]

        dataset_train = BRATSDataset(data_file_train, fields, max_lbl=max_lbl, crop_size=(128, 128, 128), baseline=baseline)
        dataset_val = BRATSDataset(data_file_val, fields, max_lbl=max_lbl, crop_size=(-1, -1, -1), baseline=baseline)

    elif dataset=="Spine":
        fields = {
            "image": ["dataPath"],
            "label": ["labelPath"]
        }

        max_lbl = 2
        n_down = 4
        hn_layers = [3, 10, 10, 20]
        repeat = 5
        baseline = model_type=="UNet"

        dataset_train = SpineMRIDataset(data_file_train, fields, max_lbl=max_lbl, crop_size=(256, 128, 48), baseline=baseline)
        dataset_val = SpineMRIDataset(data_file_val, fields, max_lbl=max_lbl, crop_size=(256, 128, 48), baseline=baseline)

    elif dataset=="CardiacMR":
        fields = {
            "image": ["ImagePath"],
            "label": ["LabelPath"]
        }

        max_lbl = 8
        n_down = 3
        baseline = model_type=="UNet"
        p_resampling = 1
        repeat = 100
        hn_layers = [3, 10, 10, 50]

        dataset_train = CardiacMRDataset(data_file_train, fields, max_lbl=max_lbl, crop_size=(128, 128, 128), max_spacing_scaling=2, baseline=baseline)
        dataset_val = CardiacMRDataset(data_file_val, fields, max_lbl=max_lbl, crop_size=(256, 256, 256), max_spacing_scaling=2, baseline=baseline)

    
    train_dataloader = DataLoader(dataset_train, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=1, shuffle=False)

    # Model
    if model_type=="FS" or model_type=="AS":
        model = UNet(in_c=1, out_c=max_lbl+1, n_down=n_down, n_fix=3, C=16, n_dim=3).cuda()
    elif model_type=="HyperSpace":
        model = HyperUnet(hn_layers, in_c=1, out_c=max_lbl+1, n_down=n_down, n_fix=3, C=16, n_dim=3).cuda()
    else:
        ValueError("Choose a correct model type.")
    
    # Optimizer
    opt = Adam(model.parameters(), lr=0.001)

    if checkpoint!="":
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])

    # logs
    path_log = f"Results/{dataset}{model_type}{suffix}"
    makedir(path_log)

    train(train_dataloader, val_dataloader, model, opt, 20000, path_log, device="cuda", repeat=repeat)


    