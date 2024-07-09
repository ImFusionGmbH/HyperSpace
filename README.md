# HyperSpace
Code for the MICCAI 2024 paper "[HyperSpace: Hypernetworks for spacing-adaptive image segmentation](https://arxiv.org/abs/2407.03681)"

# Overview

HyperSpace conditions an underlying segmentation UNet on the voxel spacing using [HyperNetworks](https://arxiv.org/abs/1609.09106). 
This allows, the end user to specify the spatial resolution of the segmentation network at inference time, and to 
dynamically adjust the network to the computation constraints and image properties, potentially reducing vRAM and time 
requirements by orders of magnitude.

![HyperSpace overview](images/HyperSpaceIllustration.png "HyperSpace overview")

Our experiments across multiple datasets demonstrate that this approach achieves competitive performance compared to resolution-specific models, while offering greater flexibility for the end user. This also simplifies model development, deployment and maintenance.

![Illustration of deployment flexibility](images/Suite_illustration.png "Illustration of deployment flexibility")

# Repository organization

The repository is only composed of 5 files:

- **data.py**: This file contains the different data loaders used for our 3 experiments (one per dataset). This part of the pipeline relies on the [ImFusion Python SDK](https://docs.imfusion.com/python/) for image loading and basic preprocessing. These loaders expect data list files that specify input data files and associated field names. Example data lists are available in the **data_files** folder.

- **models.py**: This file contains the model definitions, in particular the definition of a standard UNet and a conditional UNet which are the two types of architectures used in our experiments in different training or inference settings.

- **train.py**: This file contains the training and validation loops. To reproduce our training, simply run:

    `python train.py -d datasetName -m modelName -data_file_train path_to_train_data_file -data_file_val path_to_validation_data_file`

    Results will be saved in **Results/{datasetName}{ModelName}**

- **eval.py**: This file contains our evaluation script which we used to compute the different metrics reported in the paper.

- **utils.py**: This file contains various utility functions including the losses, image saving functionalities (relying on the [ImFusion Python SDK](https://docs.imfusion.com/python/) as well), and others.


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

