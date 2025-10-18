import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        target = (target - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
        
        # Get local means of input and target
        I_mean = F.conv2d(input, self.filt, stride=1, padding=int((self.kernel[0]-1)/2)) / (self.kernel[0]*self.kernel[1])
        T_mean = F.conv2d(target, self.filt, stride=1, padding=int((self.kernel[0]-1)/2)) / (self.kernel[0]*self.kernel[1])
        
        # Get local variances
        I_var = F.conv2d(input**2, self.filt, stride=1, padding=int((self.kernel[0]-1)/2)) / (self.kernel[0]*self.kernel[1]) - I_mean**2
        T_var = F.conv2d(target**2, self.filt, stride=1, padding=int((self.kernel[0]-1)/2)) / (self.kernel[0]*self.kernel[1]) - T_mean**2

        # Get cross-correlation
        IT = F.conv2d(input * target, self.filt, stride=1, padding=int((self.kernel[0]-1)/2)) / (self.kernel[0]*self.kernel[1])
        
        cross_corr = IT - I_mean * T_mean
        
        # Calculate NCC
        ncc = (cross_corr**2) / (I_var * T_var + 1e-5)
        
        return -1.0 * torch.mean(ncc)

""" old 
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
    def __init__(self, in_ch, kernel=(9, 9), voxel_weights=None):
        super(crossCorrelation2D, self).__init__()
        self.kernel = kernel
        self.voxel_weight = voxel_weights
        self.filt = (torch.ones([1, in_ch, self.kernel[0], self.kernel[1]])).cuda()

    def forward(self, input, target):

        min_max = (-1, 1)
        target = (target - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]

        II = input * input
        TT = target * target
        IT = input * target

        pad = (int((self.kernel[0] - 1) / 2), int((self.kernel[1] - 1) / 2))
        T_sum = F.conv2d(target, self.filt, stride=1, padding=pad)
        I_sum = F.conv2d(input, self.filt, stride=1, padding=pad)
        TT_sum = F.conv2d(TT, self.filt, stride=1, padding=pad)
        II_sum = F.conv2d(II, self.filt, stride=1, padding=pad)
        IT_sum = F.conv2d(IT, self.filt, stride=1, padding=pad)
        kernelSize = self.kernel[0] * self.kernel[1]
        Ihat = I_sum / kernelSize
        That = T_sum / kernelSize

        # cross = (I-Ihat)(J-Jhat)
        cross = IT_sum - Ihat * T_sum - That * I_sum + That * Ihat * kernelSize
        T_var = TT_sum - 2 * That * T_sum + That * That * kernelSize
        I_var = II_sum - 2 * Ihat * I_sum + Ihat * Ihat * kernelSize

        cc = cross * cross / (T_var * I_var + 1e-5)
        loss = -1.0 * torch.mean(cc)
        return loss
        """
