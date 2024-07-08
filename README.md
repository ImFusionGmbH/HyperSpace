# HyperSpace
Code for the MICCAI 2024 paper "[HyperSpace: Hypernetworks for spacing-adaptive image segmentation](https://arxiv.org/abs/2407.03681)"

# Overview

HyperSpace conditions an underlying segmentation UNet on the voxel spacing using hypernetworks. 

![HyperSpace overview](images/HyperSpaceIllustration.png "HyperSpace overview")

Our experiments across multiple datasets demonstrate that this approach achieves competitive performance compared to resolution-specific models, while offering greater flexibility for the end user. This also simplifies model development, deployment and maintenance.

![Illustration of deployment flexibility](images/Suite_illustration.png "Illustration of deployment flexibility")

# Repository organization

The repository is only composed of 5 files:

- **data.py**: This file contains the different data loaders used for our 3 experiments (one per dataset). This part of the pipeline relies on the [ImFusion Python SDK](https://docs.imfusion.com/python/) for basic preprocessing and loading. These expects data list files which are files where the first row contains datafield and then each row correspond to one patient and contains corresponding file paths for each data field. Example for such files are available in the **data_files** folder.

- **models.py**: This file contains the model definitions, in particular the definition of a standard UNet and a conditional UNet which are the two types of architectures used in our experiments in different training or inference settings.

- **train.py**: This file contains the training and validation loops. In order to reproduce our training, simply run:

    `python train.py -d datasetName -m modelName -data_file_train path_to_train_data_file -data_file_val path_to_validation_data_file`

    Results will be saved in **Results/{datasetName}{ModelName}**

- **eval.py**: This file contains our evaluation script which we used to compute the different metrics reported in the paper.

- **utils.py**: This file contains various utils functions including the losses, image saving functionalities (relying on the [ImFusion Python SDK](https://docs.imfusion.com/python/) as well), and others.


## Citation

```
@misc{2024hyperspace,
title={HyperSpace: Hypernetworks for spacing-adaptive image segmentation}, 
author={Samuel Joutard and Maximilian Pietsch and Raphael Prevost},
year={2024},
eprint={2407.03681},
url={https://arxiv.org/abs/2407.03681}, 
}
```

