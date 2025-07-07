#!/usr/bin/env python
# coding: utf-8
import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt
import seg_metrics.seg_metrics as sg
import monai.transforms as mon
from skimage.morphology import skeletonize, skeletonize_3d
import pandas as pd

#https://github.com/jocpae/clDice/blob/master/cldice_metric/cldice.py
def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)

def eval(out_path=str, mask_path=str):
    sensitivity=[]
    precision=[]
    label=[1]
    dice_1=[]
    cl = []
    csv = 'metrics.csv'
    for y_hat, y in zip(sorted(os.listdir(out_path)), sorted(os.listdir(mask_path))):
        y_true = nib.load(mask_path + y)
        y_pred = nib.load(out_path + y_hat)
        y_pred = y_pred.get_fdata()
        y_true = y_true.get_fdata()
        print(y_pred.shape, y_true.shape)
        metrics = sg.write_metrics(labels=[1],  
                  gdth_img=y_true,
                  pred_img=y_pred,
                  csv_file = csv,
                  #TPTNFPFN=True,
                  metrics=['dice','precision','recall'])
        dice_1.append(metrics[0]['dice'][0])
        clScore = clDice(y_pred, y_true)
        sensitivity.append(metrics[0]['recall'][0]) 
        precision.append(metrics[0]['precision'][0])
        cl.append(clScore)

    img = range(1, len(cl)+1)
    plt.plot(img, dice_1, 'g', marker='o', label='Dice')
    plt.title('Dice Score')
    plt.xlabel('Image index')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig('eval.png')
    plt.close() 

    
    plt.plot(img, cl, 'b', marker='o') 
    plt.title('clDice Score ')
    plt.xlabel('Image index')
    plt.ylabel('score')
    plt.legend()
    plt.show()
    plt.savefig('cldice.png')
    plt.close()

    avg_sens = np.mean(sensitivity) * 100
    avg_prec = np.mean(precision) * 100
    avg_cl = np.mean(cl) * 100
    df = pd.DataFrame([{"Avg Sensitivity" : avg_sens, "Avg Precision" : avg_prec, "Avg clDice Score" : avg_cl, "Avg Dice-1" : np.mean(dice_1)}])
    df.to_csv('avg_metrics.csv')

out_path =''
mask_path =''
eval(out_path, mask_path)