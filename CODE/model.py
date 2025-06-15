"""
Deep Unfolding Network for Underwater Image Enhancement

This module implements a model-driven deep unfolding approach to underwater
image enhancement. The network (Dual_Net) iteratively estimates transmission maps
and background light to restore underwater images.

Classes:
    BasicBlock: A single unfolding block of the network
    IPMM: Implicit Prior Modeling Module
    Dual_Net: Main model architecture for underwater image enhancement
"""
# py libs
# import random
import os
# import glob
import wandb
import torch
import torch.nn as nn
from net import *
from CODE.config import CONFIG

# Configure CUDA device using settings from config
os.environ["CUDA_DEVICE_ORDER"] = CONFIG["system"].CUDA_DEVICE_ORDER
os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG["system"].CUDA_VISIBLE_DEVICES
DEVICE = CONFIG["system"].DEVICE

def get_mean_value(batch: torch.Tensor) -> tuple:
    """
    Compute mean values and channel statistics for a batch of images.
    
    Args:
        batch: Input batch of images with shape [B, C, H, W]
        
    Returns:
        tuple: Contains sorted mean values, indices, and channel information
    """
    # Get batch size of input
    batch_size = batch.shape[0]
    # Create output containers
    list_mean_sorted = []
    list_indices = []
    largest_index = []
    medium_index = []
    smallest_index = []
    largest_channel = []
    medium_channel = []
    smallest_channel = []

    # Get the largest, medium, and smallest value/channels for each image in batch
    for bs in range(batch_size):
        image = batch[bs,:,:,:]
        mean = torch.mean(image, (2,1))
        mean_I_sorted, indices = torch.sort(mean)
        list_mean_sorted.append(mean_I_sorted)
        list_indices.append(indices)
        # Index of largest, medium and smallest value
        largest_index.append(indices[2])
        medium_index.append(indices[1])
        smallest_index.append(indices[0])
        # Get largest, medium and smallest channel
        largest_channel.append(torch.unsqueeze(image[indices[2],:,:], 0))
        medium_channel.append(torch.unsqueeze(image[indices[1],:,:], 0))
        smallest_channel.append(torch.unsqueeze(image[indices[0],:,:], 0))

    # Stack all results into tensors
    list_mean_sorted = torch.stack(list_mean_sorted)
    list_indices = torch.stack(list_indices)
    largest_index = torch.stack(largest_index)
    medium_index = torch.stack(medium_index)
    smallest_index = torch.stack(smallest_index)
    largest_channel = torch.stack(largest_channel)
    medium_channel = torch.stack(medium_channel)
    smallest_channel = torch.stack(smallest_channel)

    return list_mean_sorted, list_indices, largest_channel, medium_channel, smallest_channel, largest_index, medium_index, smallest_index

def mapping_index(batch: torch.Tensor, value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Maps values to specific channels in batch according to the provided indices.
    
    Args:
        batch: Input batch of images with shape [B, C, H, W]
        value: Values to map into the channels
        index: Channel indices for mapping
        
    Returns:
        torch.Tensor: Batch with mapped values
    """
    batch_size = batch.shape[0]
    new_batch = []
    for bs in range(batch_size):
        image = batch[bs,:,:,:]
        image[index[bs],:,:] = value[bs]
        new_batch.append(image)
    new_batch = torch.stack(new_batch)

    return new_batch

def get_dark_channel(x: torch.Tensor, patch_size: int) -> tuple:
    """
    Computes the dark channel prior for underwater image enhancement.
    
    Args:
        x: Input tensor of shape [B, C, H, W]
        patch_size: Size of patch for computing dark channel
        
    Returns:
        tuple: Dark channel map and corresponding index map
    """
    pad_size = (patch_size - 1) // 2
    # Get dimensions
    H, W = x.size()[2], x.size()[3]
    # Minimum among three channels
    x, _ = x.min(dim=1, keepdim=True)  # (B, 1, H, W)
    x = nn.ReflectionPad2d(pad_size)(x)  # (B, 1, H+2p, W+2p)
    x = nn.Unfold(patch_size)(x) # (B, k*k, H*W)
    x = x.unsqueeze(1)  # (B, 1, k*k, H*W)
    
    # Minimum in (k, k) patch
    index_map = torch.argmin(x, dim=2, keepdim=False)
    dark_map, _ = x.min(dim=2, keepdim=False)  # (B, 1, H*W)
    dark_map = dark_map.view(-1, 1, H, W)

    return dark_map, index_map

def softThresh(x: torch.Tensor, lamda: float) -> torch.Tensor:
    """
    Applies soft thresholding function for proximal gradient methods.
    
    Args:
        x: Input tensor
        lamda: Threshold value
        
    Returns:
        torch.Tensor: Soft-thresholded tensor
    """
    # Move operation entirely to the device of the input tensor
    device = x.device
    relu = nn.ReLU()
    return torch.sign(x).to(device) * relu(torch.abs(x).to(device) - lamda)

# Proposed algorithm
class BasicBlock(nn.Module):
    """
    Basic unfolding block representing one iteration of the optimization algorithm.
    
    This block implements the core operations of the unfolding network including 
    transmission map estimation, background light estimation, and image reconstruction.
    """
    def __init__(self):
        super(BasicBlock, self).__init__()
        print('Loading subnetworks .....')
        H_Net = [RDN(CONFIG["model"].IN_CHANNELS, 1, 
                     num_features=CONFIG["model"].RDN_NUM_FEATURES,
                     growth_rate=CONFIG["model"].RDN_GROWTH_RATE,
                     num_blocks=CONFIG["model"].RDN_NUM_BLOCKS,
                     num_layers=CONFIG["model"].RDN_NUM_LAYERS)]
        self.H_Net = nn.Sequential(*H_Net)

        self.t_1D_Net = nn.Sequential(
            nn.Conv2d(in_channels=CONFIG["model"].IN_CHANNELS, 
                      out_channels=1, 
                      kernel_size=1, 
                      bias=False),
        )
        # Learnable parameters - Initialize with values from config
        self.gamma_1 = nn.Parameter(torch.tensor([CONFIG["model"].INIT_GAMMA]), requires_grad=True)
        self.gamma_2 = nn.Parameter(torch.tensor([CONFIG["model"].INIT_GAMMA]), requires_grad=True)
        self.gamma_3 = nn.Parameter(torch.tensor([CONFIG["model"].INIT_GAMMA]), requires_grad=True)
        self.gamma_4 = nn.Parameter(torch.tensor([CONFIG["model"].INIT_GAMMA]), requires_grad=True)
        self.gamma_5 = nn.Parameter(torch.tensor([CONFIG["model"].INIT_GAMMA]), requires_grad=True)

    def forward(self, I, t_p, B_p, B, t, J, G, H, P, Q, u, v, X, Y, 
                patch_size=CONFIG["model"].PATCH_SIZE, 
                eps=CONFIG["model"].EPSILON):
        """
        Forward pass of the BasicBlock.
        
        Args:
            I: Input underwater image
            t_p: Transmission map prior
            B_p: Background light prior
            B: Current estimate of background light
            t: Current estimate of transmission map
            J: Current estimate of enhanced image
            G, H, P, Q, u, v, X, Y: Auxiliary variables for optimization
            patch_size: Patch size for dark channel computation
            eps: Small constant for numerical stability
            
        Returns:
            tuple: Updated estimates of all variables
        """
        # Fixed lambda parameters from config
        lambda_1 = CONFIG["model"].LAMBDA_1
        lambda_2 = CONFIG["model"].LAMBDA_2
        lambda_3 = CONFIG["model"].LAMBDA_3
        lambda_4 = CONFIG["model"].LAMBDA_4
        lambda_5 = CONFIG["model"].LAMBDA_5
        
        # Learnable parameters
        gamma_1 = self.gamma_1
        gamma_2 = self.gamma_2
        gamma_3 = self.gamma_3
        gamma_4 = self.gamma_4
        gamma_5 = self.gamma_5

        # Math modules
        ## B-module
        D = torch.ones(I.shape).to(I.device)  # Use the device of input tensor
        B = (lambda_3*B_p - lambda_1*(J*t - I)*(1 - t))/(lambda_1*(1.0 - t)*(1 - t) + lambda_3)
        B = torch.mean(B,(2,3), True)
        B = B*D

        ## t-module
        t = (lambda_2*t_p + gamma_4*H - lambda_1*(B - I)*(J - B) - X)/(lambda_1*(J - B)*(J - B) + lambda_2 + gamma_4)
        t = self.t_1D_Net(t)
        t = torch.cat((t,t,t), 1)

        M_T_P = u
        M_T_Q = v

        ## J-module
        J = (lambda_1*(t*(I - B*(1.0 - t))) + gamma_3*G + gamma_4*u - gamma_5*v - Y + gamma_5)/(lambda_1*t*t + gamma_3 + gamma_4 + gamma_5)

        u = (gamma_1*M_T_P + gamma_4*J)/(gamma_1 + gamma_4)
        v = (gamma_2*M_T_Q - gamma_5*J + gamma_5)/(gamma_2 + gamma_5)

        ## Z_Net
        H = self.H_Net(t + (1.0/gamma_4) * X)

        ## P & Q module
        X = X + gamma_4*(t - H)
        Y = Y + gamma_3*(J - G)

        M_u, index_map_dark = get_dark_channel(u, patch_size)
        M_v, index_map_dark = get_dark_channel(v, patch_size)

        ## M & N module
        P = softThresh(M_u, lambda_4/gamma_1)
        Q = softThresh(M_v, lambda_5/gamma_1)
        
        return B, t, J, G, H, P, Q, u, v, X, Y, gamma_3

class IPMM(nn.Module):
    """
    Implicit Prior Modeling Module (IPMM) for enhancing the image restoration process.
    
    This module uses a UNet-like architecture to extract and refine features,
    helping to model implicit image priors for more accurate restoration.
    
    Args:
        in_c: Number of input channels
        out_c: Number of output channels
        n_feat: Number of feature channels
        scale_unetfeats: Scaling factor for UNet features
        scale_orsnetfeats: Scaling factor for ORSNet features
        num_cab: Number of channel attention blocks
        kernel_size: Convolution kernel size
        reduction: Channel reduction factor in attention mechanism
        bias: Whether to use bias in convolutions
    """
    def __init__(self, in_c=CONFIG["model"].IN_CHANNELS, 
                 out_c=CONFIG["model"].OUT_CHANNELS, 
                 n_feat=CONFIG["model"].NUM_FEATURES, 
                 scale_unetfeats=CONFIG["model"].SCALE_UNET_FEATS, 
                 scale_orsnetfeats=CONFIG["model"].SCALE_ORSNET_FEATS, 
                 num_cab=CONFIG["model"].NUM_CAB, 
                 kernel_size=CONFIG["model"].KERNEL_SIZE, 
                 reduction=CONFIG["model"].REDUCTION, 
                 bias=CONFIG["model"].USE_BIAS):
        super(IPMM, self).__init__()
        act = nn.PReLU()
        self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=4, csff=True)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=4)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias)
        self.r1 = nn.Parameter(torch.Tensor([0.5]))
        self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.merge12 = mergeblock(n_feat, in_c, True)
    
    def forward(self, x2_img, stage1_img, feat1, res1, x2_samfeats):
        """
        Forward pass of the IPMM module.
        
        Args:
            x2_img: Input image for the second stage
            stage1_img: Output image from the first stage
            feat1: Features from the first stage encoder
            res1: Features from the first stage decoder
            x2_samfeats: Features from the first stage SAM module
            
        Returns:
            tuple: Processed features and enhanced image
        """
        ## PMM
        x2 = self.shallow_feat2(x2_img)

        ## Process features through encoder and decoder
        x2, res2 = self.stage2_encoder(x2, feat1, res1)
        x3 = self.stage2_decoder(x2, res2)
        
        ## Feature integration
        stage2_img = self.sam23(x3[-1], x2_img)
        x4_cat = self.concat12(torch.cat([x2_samfeats, stage2_img[0]], 1))
        stage1_2_img = self.merge12(x4_cat, stage1_img)
        
        return stage2_img[0], stage1_2_img, feat1, res1, stage2_img[0]

class Dual_Net(torch.nn.Module):
    """
    Dual-Channel Deep Unfolding Network for Underwater Image Enhancement.
    
    This is the main model that implements the iterative deep unfolding approach
    for underwater image enhancement. It estimates transmission maps and background
    light through multiple unfolding layers, guided by physics-based priors.
    
    Args:
        LayerNo: Number of unfolding layers in the network
    """
    def __init__(self, LayerNo=CONFIG["model"].NUM_UNFOLDING_LAYERS):
        super(Dual_Net, self).__init__()

        self.LayerNo = LayerNo
        
        # Initialize the unfolding layers
        net_layers = []
        for i in range(LayerNo):
            net_layers.append(BasicBlock())
        self.uunet = nn.ModuleList(net_layers)
        
        # Initialize parameters from config
        in_c = CONFIG["model"].IN_CHANNELS
        out_c = CONFIG["model"].OUT_CHANNELS
        n_feat = CONFIG["model"].NUM_FEATURES
        scale_unetfeats = CONFIG["model"].SCALE_UNET_FEATS
        scale_orsnetfeats = CONFIG["model"].SCALE_ORSNET_FEATS
        num_cab = CONFIG["model"].NUM_CAB
        kernel_size = CONFIG["model"].KERNEL_SIZE
        reduction = CONFIG["model"].REDUCTION
        bias = CONFIG["model"].USE_BIAS
        depth = CONFIG["model"].DEPTH
        act = nn.PReLU()
        
        # Feature extraction
        self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act))
        # Encoder-decoder architecture
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=4, csff=True)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=4)
        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.r1 = nn.Parameter(torch.Tensor([0.5]))

        # Feature fusion
        self.concat12 = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.merge12 = mergeblock(n_feat, in_c, True)
        
        # Implicit Prior Modeling Module
        self.basic = IPMM(in_c=in_c, out_c=out_c, n_feat=n_feat, 
                          scale_unetfeats=scale_unetfeats, 
                          scale_orsnetfeats=scale_orsnetfeats, 
                          num_cab=num_cab, kernel_size=kernel_size, 
                          reduction=reduction, bias=bias)

    def forward(self, I, t_p, B_p):
        """
        Forward pass of the Dual_Net model.
        
        Args:
            I: Input underwater image
            t_p: Transmission map prior
            B_p: Background light prior
            
        Returns:
            tuple: Lists of intermediate outputs for each unfolding layer
                   including enhanced images (J), background light (B), 
                   transmission maps (t), and auxiliary variables
        """
        # Get batch size and initialize variables
        bs, _, _, _ = I.shape
        device = I.device
        
        # Initialize variables with proper device placement
        B = torch.zeros((bs, 3, 1, 1)).to(device)
        t = torch.zeros(I.shape).to(device)
        J = I.to(device)
        G = torch.zeros(I.shape).to(device)
        H = torch.zeros(I.shape).to(device)
        X = torch.zeros(I.shape).to(device)
        Y = torch.zeros(I.shape).to(device)
        P = torch.zeros(I.shape).to(device)
        Q = torch.zeros(I.shape).to(device)
        u = torch.zeros(I.shape).to(device)
        v = torch.zeros(I.shape).to(device)

        # Output storage
        list_J = []
        list_B = []
        list_t = []
        list_G = []
        list_H = []
        list_u = []
        list_v = []
        list_P = []
        list_Q = []

        # First stage IPMM: Proximal mapping
        gamma_3 = torch.tensor([CONFIG["model"].INIT_GAMMA]).to(device)
        x1_img = J + (1.0/gamma_3) * Y
        x1 = self.shallow_feat1(x1_img)
        feat1, feat_fin1 = self.stage1_encoder(x1)
        res1 = self.stage1_decoder(feat_fin1, feat1)
        x2_samfeats, stage1_img = self.sam12(res1[-1], x1_img)

        # Set initial G from the first stage
        G = stage1_img

        # Main unfolding iterations
        for j in range(self.LayerNo):
            # Apply BasicBlock (unfolding iteration)
            [B, t, J, G, H, P, Q, u, v, X, Y, gamma_3] = self.uunet[j](I, t_p, B_p, B, t, J, G, H, P, Q, u, v, X, Y)
            
            # Apply IPMM for enhanced restoration
            img = J + (1.0/gamma_3) * Y
            x2_samfeats, stage1_img, feat1, res1 = self.basic(img, stage1_img, feat1, res1, x2_samfeats)
            G = stage1_img
            
            # Store intermediate results
            list_G.append(G)
            list_H.append(torch.cat((H, H, H), 1))
            list_J.append(J)
            list_B.append(B)
            list_t.append(t)
            list_u.append(u)
            list_v.append(v)
            list_P.append(torch.cat((P, P, P), 1))
            list_Q.append(torch.cat((Q, Q, Q), 1))

        return list_J, list_B, list_t, list_G, list_H, list_u, list_v, list_P, list_Q