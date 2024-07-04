import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from typing import Union, Optional


class ConditionerActivation(nn.Module):
    """
    Custom activation used as final activation of the hypernetwork to constraint the predicted weigts within the sphere of infinit norm 5
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.tanh(x) * 5
    
class ReLUModule(nn.Module):
    """
    ReLU activation as a module
    """
    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        return F.relu(x)

class ConditionalConv(nn.Module):
    """
    Convolution layer which weights are predicted from a latent weight representation
    The latent weight representation is simply processed with a single linear layer followed by a ConditionerActivation

    Args:
        cond_c (int): dimension of the latent weight representation
        n_dim (int): dimension of the convolution (2 or 3)
        in_c (int): number of input channels
        out_c (int): number of output channels
        kernel_size (int or tuple): size of the convolution kernel
        stride (int or tuple): stride of the convolution operation
        padding (int or tuple): padding added to the input tensor before the convolution
        transposed (bool): if True, the convolution applied is a transposed convolution, otherwise, a standard one
    """
    def __init__(
            self, 
            cond_c: int, 
            n_dim: int, 
            in_c: int, 
            out_c: int, 
            kernel_size: Union[int, tuple[int, ...]], 
            stride: Union[int, tuple[int, ...]]=1, 
            padding: Union[int, tuple[int, ...]]=0,
            transposed: bool = False
        ) -> None:
        super(ConditionalConv, self).__init__()

        self.cond_c = cond_c
        self.n_dim = n_dim
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.transposed = transposed

        if n_dim not in [2, 3]:
            raise ValueError(f"n_dim should be 2 or 3, got {n_dim}.")
        
        if transposed:
            self.conv = F.conv_transpose2d if n_dim==2 else F.conv_transpose3d
        else:
            self.conv = F.conv2d if n_dim==2 else F.conv3d
            
        
        if isinstance(kernel_size, int):
            self.kernel_size = (self.kernel_size,) * n_dim
        
        n_param = np.prod(self.kernel_size) * in_c * out_c + out_c
        self.conditioner = nn.Sequential(
            nn.Linear(cond_c, n_param), ConditionerActivation()
        )

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        w, b = self.get_w_b(conditioning)
        res = []
        for i in range(x.size(0)):
            res.append(self.conv(x[i:i+1], w[i], b[i], self.stride, self.padding))
        return torch.cat(res, dim=0)
    
    def get_w_b(self, conditioning: torch.tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w_b = self.conditioner(conditioning)
        b = w_b[:, -self.out_c:]
        channel_shape = (-1, self.in_c, self.out_c) if self.transposed else (-1, self.out_c, self.in_c) 
        w = torch.reshape(w_b[:, :-self.out_c], channel_shape + self.kernel_size)
        return w, b
    

class ConditionalInstanceNorm(nn.Module):
    """
    Instance norm layer which parameters are predicted from a latent weight representation
    The latent weight representation is simply processed with a single linear layer followed by a ConditionerActivation

    Args:
        ccond_c (int): dimension of the latent weight representation
        n_dim (int): dimension of the tensor to be normalized (2 or 3)
        in_c (int): number of input channels
        affine (bool): if True, also predicts shift and scale to be applied after the normalization
    """
    def __init__(
            self, 
            cond_c: int, 
            n_dim: int, 
            in_c: int, 
            affine: bool=True
        ) -> None:
        super(ConditionalInstanceNorm, self).__init__()

        self.cond_c = cond_c
        self.n_dim = n_dim
        self.in_c = in_c
        self.affine = affine

        if n_dim not in [2, 3]:
            raise ValueError(f"n_dim should be 2 or 3, got {n_dim}.")
        
        n_param = in_c * 2
        self.conditioner = nn.Sequential(
            nn.Linear(cond_c, n_param), ConditionerActivation()
        )

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        w, b = self.get_w_b(conditioning)
        if w is None:
            return F.instance_norm(x)
        else:
            res = []
            for i in range(x.size(0)):
                res.append(F.instance_norm(x[i:i+1], weight=w[i], bias=b[i]))
            return torch.cat(res, dim=0)
    
    def get_w_b(self, conditioning: torch.Tensor) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.affine:
            w_b = self.conditioner(conditioning)
            return w_b[:, :self.in_c], w_b[:, self.in_c:]
        else:
            return None, None
    

class UNet(nn.Module):
    """
    Standard Unet

    Args:
        int_c (int): input channels
        out_c (int): output channels
        n_down (int): number of downsampling steps
        n_fix (int): number of convolutional layers at each resolution
        C (int): number of channels at max resolution
        Instance_norm (bool): Weither to use instance norm or batchnorm
        n_dim (int): dimension of input tensor (2 or 3)
    """
    def __init__(self, in_c: int, out_c: int, n_down: int, n_fix: int, C: int, Instance_norm: bool=True, n_dim: int=3) -> None:
        super(UNet, self).__init__()
        
        self.n_down = n_down
        self.n_fix = n_fix
        self.C = C
        self.IN = Instance_norm
        self.in_c = in_c
        self.out_c = out_c
        self.n_dim = n_dim

        if n_dim not in [2, 3]:
            raise ValueError(f"n_dim should be 2 or 3, got {n_dim}.")

        conv = nn.Conv3d if n_dim == 3 else nn.Conv2d
        transposed_conv = nn.ConvTranspose3d if n_dim == 3 else nn.ConvTranspose2d
        instance_norm = nn.InstanceNorm3d if n_dim == 3 else nn.InstanceNorm2d
        batch_norm = nn.BatchNorm3d if n_dim == 3 else nn.BatchNorm2d
        norm_layer = instance_norm if self.IN else batch_norm

        self.conv_init = conv(in_c, C, 3, 1, 1)  
        self.act_init = ReLUModule()
        self.norm_init = norm_layer(C, affine=True)

        for l in range(n_fix):
            setattr(self, "conv_0_" + str(l), conv(C, C, 3, 1, 1))
            setattr(self, "act_0_" + str(l), ReLUModule())
            setattr(self, "norm_0_" + str(l), norm_layer(C, affine=True))
        for lvl in range(n_down):
            setattr(self, "down_" + str(lvl), conv(2**(lvl) * C, 2**(lvl + 1) * C, 3, 2, 1))
            setattr(self, "down_act_" + str(lvl), ReLUModule())
            setattr(self, "down_norm_" + str(lvl), norm_layer(2**(lvl + 1) * C, affine=True))
            for l in range(n_fix):
                setattr(self, f"conv_{lvl+1}_{l}", conv(2**(lvl + 1) * C, 2**(lvl + 1) * C, 3, 1, 1))
                setattr(self, f"act_{lvl+1}_{l}", ReLUModule())
                setattr(self, f"norm_{lvl+1}_{l}", norm_layer(2**(lvl + 1) * C, affine=True))
        for lvl in range(n_down):
            setattr(self, "up_" + str(lvl), transposed_conv(2**(lvl + 1) * C, 2**(lvl) * C, 4, 2, 1))
            setattr(self, "up_act_" + str(lvl), ReLUModule())
            setattr(self, "up_norm_" + str(lvl), norm_layer(2**(lvl) * C, affine=True))
            for l in range(n_fix):
                if l == 0:
                    setattr(self, "dec_conv_" + str(lvl) + "_0", conv(2**(lvl + 1) * C, 2**(lvl) * C, 3, 1, 1))
                else:
                    setattr(self, "dec_conv_" + str(lvl) + "_" + str(l), conv(2**(lvl) * C, 2**(lvl) * C, 3, 1, 1))
                setattr(self, "dec_act_" + str(lvl) + "_" + str(l), ReLUModule())
                setattr(self, "dec_norm_" + str(lvl) + "_" + str(l), norm_layer(2**(lvl) * C, affine=True))
        self.conv_final = conv(C, out_c, 3, 1, 1)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        x = self.conv_init(x)
        x = self.act_init(x)
        x = self.norm_init(x)
        L = []
        for l in range(self.n_fix):
            x = getattr(self, "conv_0_" + str(l))(x)
            x = getattr(self, "act_0_" + str(l))(x)
            x = getattr(self, "norm_0_" + str(l))(x)
        L.append(x)
        for lvl in range(self.n_down):
            x = getattr(self, "down_" + str(lvl))(x)
            x = getattr(self, "down_act_" + str(lvl))(x)
            x = getattr(self, "down_norm_" + str(lvl))(x)
            for l in range(self.n_fix):
                x = getattr(self, f"conv_{lvl+1}_{l}")(x)
                x = getattr(self, f"act_{lvl+1}_{l}")(x)
                x = getattr(self, f"norm_{lvl+1}_{l}")(x)
            L.append(x)
        for lvl in range(self.n_down - 1, -1, -1):
            x = getattr(self, "up_" + str(lvl))(x)
            x = getattr(self, "up_act_" + str(lvl))(x)
            x = getattr(self, "up_norm_" + str(lvl))(x)
            x = torch.cat([x, L[lvl]], dim=1)
            for l in range(self.n_fix):
                x = getattr(self, "dec_conv_" + str(lvl) + "_" + str(l))(x)
                x = getattr(self, "dec_act_" + str(lvl) + "_" + str(l))(x)
                x = getattr(self, "dec_norm_" + str(lvl) + "_" + str(l))(x)
        x = self.conv_final(x)
        return x
    

class ConditionalUNet(nn.Module):
    """
    UNet with Condional convolutions/InstanceNorm layers instead of standard convolutions/instance norm layers.
    The forward method takes as input the input tensor and the latent weight representation shared by all convolutions
    and instance norm layers.

    Args:
        cond_c (int): dimension of the latent weight representation
        in_c (int): number of input channels
        out_c (int): number of output channels
        n_down (int): number of downsampling steps
        n_fix (int): number of convolutional layers at each resolution
        C (int): number of channels at max resolution
        n_dim (int): dimension of the convolution (2 or 3)
    """
    def __init__(self, cond_c: int, in_c: int, out_c: int, n_down: int, n_fix: int, C: int, n_dim: int=3) -> None:
        super().__init__()

        self.cond_c = cond_c
        self.in_c = in_c
        self.out_c = out_c
        self.n_down = n_down
        self.n_fix = n_fix
        self.C = C
        self.n_dim = n_dim

        self.conv_init = ConditionalConv(cond_c, n_dim, in_c, C, 3, 1, 1)
        self.act_init = ReLUModule()
        self.norm_init = ConditionalInstanceNorm(cond_c, n_dim, C, affine=True)

        for l in range(n_fix):
            setattr(self, "conv_0_" + str(l), ConditionalConv(cond_c, n_dim, C, C, 3, 1, 1))
            setattr(self, "act_0_" + str(l), ReLUModule())
            setattr(self, "norm_0_" + str(l), ConditionalInstanceNorm(cond_c, n_dim, C, affine=True))
        for lvl in range(n_down):
            setattr(self, "down_" + str(lvl), ConditionalConv(cond_c, n_dim, 2**(lvl) * C, 2**(lvl + 1) * C, 3, 2, 1))
            setattr(self, "down_act_" + str(lvl), ReLUModule())
            setattr(self, "down_norm_" + str(lvl), ConditionalInstanceNorm(cond_c, n_dim, 2**(lvl + 1) * C, affine=True))
            for l in range(n_fix):
                setattr(self, f"conv_{lvl+1}_{l}", ConditionalConv(cond_c, n_dim, 2**(lvl+1) * C, 2**(lvl+1) * C, 3, 1, 1))
                setattr(self, f"act_{lvl+1}_{l}", ReLUModule())
                setattr(self, f"norm_{lvl+1}_{l}", ConditionalInstanceNorm(cond_c, n_dim, 2**(lvl+1) * C, affine=True))
        for lvl in range(n_down):
            setattr(self, "up_" + str(lvl), ConditionalConv(cond_c, n_dim, 2**(lvl+1) * C, 2**(lvl) * C, 4, 2, 1, transposed=True))
            setattr(self, "up_act_" + str(lvl), ReLUModule())
            setattr(self, "up_norm_" + str(lvl), ConditionalInstanceNorm(cond_c, n_dim, 2**(lvl) * C, affine=True))
            for l in range(n_fix):
                if l == 0:
                    setattr(self, "dec_conv_" + str(lvl) + "_0", ConditionalConv(cond_c, n_dim, 2**(lvl+1) * C, 2**(lvl) * C, 3, 1, 1))
                else:
                    setattr(self, "dec_conv_" + str(lvl) + "_" + str(l), ConditionalConv(cond_c, n_dim, 2**(lvl) * C, 2**(lvl) * C, 3, 1, 1))
                setattr(self, "dec_act_" + str(lvl) + "_" + str(l), ReLUModule())
                setattr(self, "dec_norm_" + str(lvl) + "_" + str(l), ConditionalInstanceNorm(cond_c, n_dim, 2**(lvl) * C, affine=True))
        self.conv_final = ConditionalConv(cond_c, n_dim, C, out_c, 3, 1, 1)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        x = self.conv_init(x, conditioning)
        x = self.act_init(x)
        x = self.norm_init(x, conditioning)
        L = []
        for l in range(self.n_fix):
            x = getattr(self, "conv_0_" + str(l))(x, conditioning)
            x = getattr(self, "act_0_" + str(l))(x, conditioning)
            x = getattr(self, "norm_0_" + str(l))(x, conditioning)
        L.append(x)
        for lvl in range(self.n_down):
            x = getattr(self, "down_" + str(lvl))(x, conditioning)
            x = getattr(self, "down_act_" + str(lvl))(x, conditioning)
            x = getattr(self, "down_norm_" + str(lvl))(x, conditioning)
            for l in range(self.n_fix):
                x = getattr(self, f"conv_{lvl+1}_{l}")(x, conditioning)
                x = getattr(self, f"act_{lvl+1}_{l}")(x, conditioning)
                x = getattr(self, f"norm_{lvl+1}_{l}")(x, conditioning)
            L.append(x)
        for lvl in range(self.n_down - 1, -1, -1):
            x = getattr(self, "up_" + str(lvl))(x, conditioning)
            x = getattr(self, "up_act_" + str(lvl))(x, conditioning)
            x = getattr(self, "up_norm_" + str(lvl))(x, conditioning)
            x = torch.cat([x, L[lvl]], dim=1)
            for l in range(self.n_fix):
                x = getattr(self, "dec_conv_" + str(lvl) + "_" + str(l))(x, conditioning)
                x = getattr(self, "dec_act_" + str(lvl) + "_" + str(l))(x, conditioning)
                x = getattr(self, "dec_norm_" + str(lvl) + "_" + str(l))(x, conditioning)
        x = self.conv_final(x, conditioning)
        return x
    
    def get_unet(self, conditioning: torch.Tensor) -> UNet:
        """
        Create a UNet from a latent weight representation

        Args:
            conditioning (torch.Tensor): latent weight representation
        """
        if conditioning.size(0) != 1:
            raise ValueError(f"Provide only one latent weight representation, size should be (1, {self.cond_c}).")
        unet = UNet(self.in_c, self.out_c, self.n_down, self.n_fix, self.C, True, self.n_dim)
        state_dict = unet.state_dict()
        for key in state_dict.keys():
            if "bias" in key:
                continue
            query_name = key.replace(".weight", "")
            w, b = getattr(self, query_name).get_w_b(conditioning)
            state_dict[key] = w[0]
            state_dict[key.replace("weight", "bias")] = b[0]
        unet.load_state_dict(state_dict)
        return unet
    



class HyperUnet(nn.Module):
    """
    HyperNetwork with a UNet as primary network

    Args:
        hypernetwork_layers (list): list of sizes of the hypernetwork hidden layers, starts with the dimension of the conditioning variable
                                    (3 for HyperSpace) and then corresponds to hidden layer widths 
        in_c (int): number of input channels
        out_c (int): number of output channels
        n_down (int): number of convolutional layers at each resolution
        n_fix (int): number of convolutional layers at each resolution
        C (int): number of convolutional layers at each resolution
        n_dim (int): dimension of the convolution (2 or 3)
    """

    def __init__(self, hypernetwork_layers: list[int], in_c: int, out_c: int, n_down: int, n_fix: int, C: int, n_dim: int=3) -> None:
        super(HyperUnet, self).__init__()
        self.hypernetwork_layers = hypernetwork_layers
        self.n_down = n_down
        self.n_fix = n_fix
        self.C = C
        self.in_c = in_c
        self.out_c = out_c
        self.n_dim = n_dim

        self.conditioner = nn.Sequential(*[
            nn.Sequential(nn.Linear(hypernetwork_layers[i], hypernetwork_layers[i + 1]), nn.ReLU())
            for i in range(len(hypernetwork_layers) - 1)
        ])

        self.unet = ConditionalUNet(hypernetwork_layers[-1], in_c, out_c, n_down, n_fix, C, n_dim=n_dim)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        conditioning = self.conditioner(conditioning)
        return self.unet(x, conditioning)
    
    def get_unet(self, conditioning: torch.Tensor) -> UNet:
        """
        Create a UNet from a conditioning variable

        Args:
            conditioning (torch.Tensor): conditioning variable
        """
        if conditioning.size(0) != 1:
            raise ValueError(f"Provide only one conditioning variable, size should be (1, {self.hypernetwork_layers[0]}).")
        conditioning = self.conditioner(conditioning)
        return self.unet.get_unet(conditioning)
