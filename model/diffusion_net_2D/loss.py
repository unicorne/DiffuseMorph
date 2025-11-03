import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class gradientLoss(nn.Module):
    def __init__(self, penalty='l1'):
        super(gradientLoss, self).__init__()
        self.penalty = penalty

    def forward(self, input):
        dH = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])
        dW = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])
        if(self.penalty == "l2"):
            dH = dH * dH
            dW = dW * dW
        loss = (torch.mean(dH) + torch.mean(dW)) / 2.0
        return loss


class crossCorrelation2D(nn.Module):
    def __init__(self, in_ch, kernel=(9, 9)):
        super(crossCorrelation2D, self).__init__()
        self.kernel = kernel
        self.filt = (torch.ones([1, in_ch, self.kernel[0], self.kernel[1]])).cuda()

    def forward(self, input, target):
        min_max = (-1, 1)
        # The 'target' (fixed image) is in range [-1, 1]
        target = (target - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
        # The 'input' (warped image) is already in range [0, 1]
        
        # Calculate padding as a tuple
        pad = (int((self.kernel[0] - 1) / 2), int((self.kernel[1] - 1) / 2))

        # Get local means of input and target
        I_mean = F.conv2d(input, self.filt, stride=1, padding=pad) / (self.kernel[0]*self.kernel[1])
        T_mean = F.conv2d(target, self.filt, stride=1, padding=pad) / (self.kernel[0]*self.kernel[1])
        
        # Get local variances
        I_var = F.conv2d(input**2, self.filt, stride=1, padding=pad) / (self.kernel[0]*self.kernel[1]) - I_mean**2
        T_var = F.conv2d(target**2, self.filt, stride=1, padding=pad) / (self.kernel[0]*self.kernel[1]) - T_mean**2

        # Get cross-correlation
        IT = F.conv2d(input * target, self.filt, stride=1, padding=pad) / (self.kernel[0]*self.kernel[1])
        
        cross_corr = IT - I_mean * T_mean
        
        # Calculate NCC
        ncc = (cross_corr**2) / (I_var * T_var + 1e-5)
        
        return -1.0 * torch.mean(ncc)


class MINDSSC2D(nn.Module):
    def __init__(self, radius=2, dilation=2):
        """
        Implementation of the MIND-SSC loss function for 2D
        Adapted from 3D version: http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf
        And 3D implementation: https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/TransMorph/losses.py
        """
        super(MINDSSC2D, self).__init__()
        self.radius = radius
        self.dilation = dilation

    def pdist_squared(self, x):
        """
        Calculates squared Euclidean distance between coordinate pairs.
        Input: x (Tensor) - Shape: [1, 2, 4]
        Output: dist (Tensor) - Shape: [4, 4]
        """
        xx = (x ** 2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0
        dist = torch.clamp(dist, 0.0, np.inf)
        return dist.squeeze(0) # Return shape [4, 4]

    def mindssc_descriptor_2d(self, img):
        """
        Calculates the 2D MIND-SSC descriptor for a given image.
        Input: img (Tensor) - Shape: [B, C, H, W], expected in [0, 1] range
        Output: mind (Tensor) - Shape: [B, 4, H, W]
        """
        B, C, H, W = img.shape
        assert H > 1 and W > 1, "Image dimensions must be > 1"
        assert C == 1, "MIND-SSC descriptor expects single-channel images"

        kernel_size = self.radius * 2 + 1
        
        # 4-neighborhood centered on [1, 1]
        # *** FIX: Changed dtype from torch.long to torch.float32 ***
        four_neighbourhood = torch.tensor([[0, 1],
                                           [1, 0],
                                           [1, 2],
                                           [2, 1]], dtype=torch.float32, device=img.device)

        # Shape: [1, 2, 4] -> [4, 4]
        dist = self.pdist_squared(four_neighbourhood.t().unsqueeze(0))

        # Shape: [4], [4] -> [4, 4]
        x, y = torch.meshgrid(torch.arange(4, device=img.device), torch.arange(4, device=img.device), indexing='ij')
        # Shape: [16] & [16] -> [16]
        mask = ((x > y).view(-1) & (dist == 2).view(-1))
        
        # 4 pairs, 2 coords per pair
        # Shape: [16, 2] -> [4, 2]
        # We need .long() here for indexing
        idx_shift1 = four_neighbourhood.long().unsqueeze(1).repeat(1, 4, 1).view(-1, 2)[mask, :]
        idx_shift2 = four_neighbourhood.long().unsqueeze(0).repeat(4, 1, 1).view(-1, 2)[mask, :]
        
        # 4 kernels, 1 in-channel, 3x3
        mshift1 = torch.zeros(4, 1, 3, 3, device=img.device) 
        mshift1.view(-1)[torch.arange(4, device=img.device) * 9 + idx_shift1[:, 0] * 3 + idx_shift1[:, 1]] = 1
        
        mshift2 = torch.zeros(4, 1, 3, 3, device=img.device)
        mshift2.view(-1)[torch.arange(4, device=img.device) * 9 + idx_shift2[:, 0] * 3 + idx_shift2[:, 1]] = 1

        rpad1 = torch.nn.ReplicationPad2d(self.dilation)
        rpad2 = torch.nn.ReplicationPad2d(self.radius)
        
        # compute patch-ssd
        h1 = torch.nn.functional.conv2d(rpad1(img), mshift1, dilation=self.dilation)
        h2 = torch.nn.functional.conv2d(rpad1(img), mshift2, dilation=self.dilation)
        
        diff = rpad2((h1 - h2) ** 2)
        ssd = torch.nn.functional.avg_pool2d(diff, kernel_size, stride=1)
        
        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var, (mind_var.mean() * 0.001).item(), (mind_var.mean() * 1000).item())
        mind /= (mind_var + 1e-6) # Add epsilon for stability
        mind = torch.exp(-mind)
        
        return mind

    def forward(self, input, target):
        """
        Computes the MIND-SSC loss between two images.
        Loss is the Mean Squared Error between descriptors.
        'input' (warped) is expected in [0, 1] range.
        'target' (fixed) is expected in [-1, 1] range.
        """
        min_max = (-1, 1)
        # 'target' (fixed image) is in range [-1, 1]
        target_norm = (target - min_max[0]) / (min_max[1] - min_max[0]) # Normalize target to [0, 1]
        input_norm = (input - min_max[0]) / (min_max[1] - min_max[0]) # Normalize input to [0, 1]
        # 'input' (warped image) is already in range [0, 1]
        
        input_mind = self.mindssc_descriptor_2d(input_norm)
        target_mind = self.mindssc_descriptor_2d(target_norm)
        
        # Return Mean Squared Error between descriptors
        return torch.mean((input_mind - target_mind) ** 2)