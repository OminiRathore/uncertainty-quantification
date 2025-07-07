#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio

def uncertainty_loss(weight, Lambda):
    def loss(y_pred, y_true, U):
        CEloss = nn.CrossEntropyLoss(weight=weight.cuda())
        L=CEloss(y_pred.cuda(),y_true.cuda())
        norm = tio.RescaleIntensity((0, 1))
        processed_U= []
        y_true = F.one_hot(y_true, num_classes = 2)
        y_true = y_true.permute(0, 4, 1, 2, 3)
        for i in range(U.shape[0]):
            U_i = U[i] 
            # Apply torchio normalization
            U_i_norm = norm(U_i.detach())
            processed_U.append(U_i_norm)

        # Stack the processed volumes back into a single tensor
        U = torch.stack(processed_U).cuda()
        y_pred_probs = F.softmax(y_pred,dim=1)
        _epsilon = torch.finfo(y_pred_probs.dtype).eps
        uq = torch.mean((U*y_true)*(torch.log(U*y_pred_probs + _epsilon))) #pixel-wise uncertainty loss
        uq *= Lambda
        return L-uq #cross-entropy loss - pixel-wise uncertainty loss
    return loss
