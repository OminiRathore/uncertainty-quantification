#!/usr/bin/env python
# coding: utf-8
import argparse
import numpy as np
from dataset import CustomDataModule
import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import monai
from monai.networks.nets import UNet
from monai.networks.layers.factories import Act
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.simplelayers import SkipConnection
from monai.transforms import SaveImage
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchio as tio
import losses
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import RichProgressBar
from matplotlib.ticker import LogLocator, ScalarFormatter
from monai.transforms import ResizeWithPadOrCrop
from monai.transforms import KeepLargestConnectedComponent
import nibabel as nib
import logging


class Ensemble(pl.LightningModule):
    
    def __init__(self, net, criterion, learning_rate, optimizer_class, save_path, num_submodels, num_layers, samples, train_step, ckpt):
        super().__init__()
        device = torch.device("cuda")
        self.lr = learning_rate
        self.net = net
        self.net.to(device)
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.save_path = save_path
        self.automatic_optimization=False
        self.num_submodels = num_submodels
        self.num_layers= num_layers
        self.samples = samples
        self.avg_train_loss=[]
        self.avg_val_loss=[]
        self.all_dices=[]
        self.losses=[]
        self.val_losses=[]
        self.avg_test_loss=[]
        self.train_step = train_step
        self.UList=[]   
        #Initialise sub-models with different weights
        for i in range(self.num_submodels):
            torch.manual_seed(i)  # Set a different seed for each sub-model
            submodel= UNet(spatial_dims=3, in_channels=1, out_channels=2, channels=(32, 64, 128, 256), strides=(2, 2, 2), act=Act.LEAKYRELU)
            self.UList.append(submodel.to(device))
        if self.train_step == 2:
            self.load_models(ckpt, self.UList)
        self.submodel_idx=[]
        self.avg_var=[]

    def load_models(self,ckpt, submodels):
        #Load sub-models trained in step 1
        model = torch.load(ckpt, weights_only = True)
        for i, submodel in enumerate(submodels):
            submodel.load_state_dict(model[f'model{i}_state_dict'])
        self.lr = model['lr']
            

    def forward(self, x):
        return self._model(x)
    
    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr, weight_decay=1e-5)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.2, patience = 30, threshold= 0.005, threshold_mode = 'abs' ),
            "monitor": "train_loss",
            }
        }
    
    def get_conv_layers(self, model):
    #Recursively collect all convolutional layers (Conv3d, ConvTranspose3d) in the model
        conv_layers = []
        for layer in model.children():
            if isinstance(layer, (nn.Conv3d, nn.ConvTranspose3d)):
                conv_layers.append(layer)
            elif isinstance(layer, (nn.Sequential, nn.Module)):
                conv_layers.extend(self.get_conv_layers(layer))
        return conv_layers
    
    def prepare_batch(self, batch):
        return batch['image'][tio.DATA], batch['label'][tio.DATA]
    
    def infer_test_batch(self, batch):
        x, y= self.prepare_batch(batch)
        x = x.float().cuda()
        unet_conv_layers = self.get_conv_layers(self.net)
        self.num_layers = len(unet_conv_layers) 
        for n in range(self.samples):
            self.submodel_idx = [torch.randint(0, self.num_submodels, (1,)).item() for _ in range(self.num_layers)]
            for q in range(self.num_layers):
                submodel_idx = self.submodel_idx[q]
                submodel = self.UList[submodel_idx]
                submodel_conv_layers = self.get_conv_layers(submodel) #Select each layer randomly from the sub-models
                with torch.no_grad():
                    unet_conv_layers[q].weight.copy_(submodel_conv_layers[q].weight) #copy the weights to the base U-net model
                    unet_conv_layers[q].bias.copy_(submodel_conv_layers[q].bias)
            y_hat = self.net(x) #model output using selected layers
            y_pred_probs = F.softmax(y_hat,dim=1)
            #Collect n predictions in stack
            if n==0:
                stack = y_pred_probs.unsqueeze(0).cpu()
            else:
                stack = torch.cat((stack, y_pred_probs.unsqueeze(0).cpu()), dim =0)
        U = torch.var(stack, dim=0, keepdim=False) #calculate variance of n predictions
        y_hat = torch.mean(stack, dim=0, keepdim=False) #calculate mean of n predictions
        y=(torch.squeeze(y, 1)).long()
        return y_hat.cuda(), y.cuda(), U.cuda()
    
    def infer_batch_step1(self, batch):
        x, y = self.prepare_batch(batch)
        x = x.float().cuda()    
        unet_conv_layers = self.get_conv_layers(self.net)
        self.num_layers = len(unet_conv_layers)
        self.submodel_idx = [torch.randint(0, args.num_submodels, (1,)).item() for _ in range(self.num_layers)]
        for q in range(self.num_layers):
            submodel_idx = self.submodel_idx[q]
            submodel = self.UList[submodel_idx]
            submodel_conv_layers = self.get_conv_layers(submodel) #Select each layer randomly from the sub-models
            with torch.no_grad():
                unet_conv_layers[q].weight.copy_(submodel_conv_layers[q].weight)#copy the weights to the base U-net model
                unet_conv_layers[q].bias.copy_(submodel_conv_layers[q].bias)
        y_hat = self.net(x) #model output using selected layers
        y=(torch.squeeze(y, 1)).long()
        return y_hat, y
    
    def infer_batch_step2(self, batch):
        x, y = self.prepare_batch(batch)
        x = x.float().cuda()    
        unet_conv_layers = self.get_conv_layers(self.net)
        self.num_layers = len(unet_conv_layers)
        for n in range(self.samples):
            self.submodel_idx = [torch.randint(0, self.num_submodels, (1,)).item() for _ in range(self.num_layers)]
            for q in range(self.num_layers):
                submodel_idx = self.submodel_idx[q]
                submodel = self.UList[submodel_idx]
                submodel_conv_layers = self.get_conv_layers(submodel) #Select each layer randomly from the sub-models
                with torch.no_grad():
                    unet_conv_layers[q].weight.copy_(submodel_conv_layers[q].weight)#copy the weights to the base U-net model
                    unet_conv_layers[q].bias.copy_(submodel_conv_layers[q].bias)
            y_hat = self.net(x) #model output using selected layers
            y_pred_probs = F.softmax(y_hat,dim=1)
            #Collect n predictions in stack
            if n==0:
                 stack = y_pred_probs.unsqueeze(0).cpu()
            else:
                 stack = torch.cat((stack, y_pred_probs.unsqueeze(0).cpu()), dim =0)

        U = torch.var(stack, dim=0, keepdim=False) #calculate variance of n predictions
        y=(torch.squeeze(y, 1)).long() #calculate mean of n predictions
        return y_hat.cuda(), y.cuda(), U
        
    def training_step(self, batch, batch_idx):
        if self.train_step == 1:
            y_hat, y = self.infer_batch_step1(batch)
            opt = self.optimizers() 
            opt.zero_grad()
            loss = self.criterion(y_hat, y)
            self.manual_backward(loss)
            opt.step()
        else:
            y_hat, y, U = self.infer_batch_step2(batch)
            opt = self.optimizers() 
            opt.zero_grad()
            loss = self.criterion(y_hat, y, U)
            self.manual_backward(loss)
            opt.step()
            
        #Update weights and bias of layers of respective sub-models after training
        unet_conv_layers = self.get_conv_layers(self.net)
        for q in range(self.num_layers):
            submodel_idx = self.submodel_idx[q]
            submodel = self.UList[submodel_idx]
            submodel_conv_layers = self.get_conv_layers(submodel)
            with torch.no_grad():
                submodel_conv_layers[q].weight.copy_(unet_conv_layers[q].weight)
                submodel_conv_layers[q].bias.copy_(unet_conv_layers[q].bias)
        
        self.losses.append(loss.item())
        self.log('train_loss', loss.item(), prog_bar=True, logger=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=16)
        return loss
    
    def validation_step(self, batch, batch_idx):
        if self.train_step == 1:
            y_hat, y = self.infer_batch_step1(batch)
            val_loss = self.criterion(y_hat, y)
        else:
            y_hat, y, U = self.infer_batch_step2(batch)
            val_loss = self.criterion(y_hat, y, U)
        
        self.val_losses.append(val_loss.item())
        self.log('val_loss', val_loss.item(), logger=True, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=16)
        return val_loss
    
    def on_train_epoch_end(self):
        self.avg_train_loss.append((sum(self.losses) / len(self.losses)))
        self.losses=[]
        sch = self.lr_schedulers()
        sch.step(self.trainer.callback_metrics["train_loss"])
        
    
    def on_validation_epoch_end(self):
        self.avg_val_loss.append((sum(self.val_losses) / len(self.val_losses)))
        self.val_losses=[]

    def post_process(self, shape, var, img):
        crop = tio.CropOrPad(shape[1:]) #crop image and variance to original resolution
        img = crop(img)
        var = crop(var)
        connect_comp = KeepLargestConnectedComponent() #remove smaller segments
        img = connect_comp(img)
        return var, img

    
    def test_step(self, batch, batch_idx):
        y_hat, y, U = self.infer_test_batch(batch)
        test_loss = self.criterion(y_hat, y, U)
        self.avg_test_loss.append(test_loss.item())
        y_hat = y_hat.argmax(dim=1, keepdim=False) #Convert logits to classes 0 and 1
        U = torch.gather(U, dim=1, index=y_hat.unsqueeze(1)) #Collect pixel-wise variance
        self.avg_var.append(torch.mean(U))
        batch['shape'] = [int(x.cpu().item())  for x in batch['shape']]
        U, y_hat = self.post_process(batch['shape'], U.squeeze(1), y_hat.squeeze(1))
        y_hat = y_hat.squeeze(0).cpu().numpy().astype(np.int16)
        U = U.squeeze(0).cpu().numpy().astype(np.float32)
        filename = batch['name'][0]
        affine = batch['affine'].cpu().squeeze(0).numpy()
        nib_img = nib.Nifti1Image(y_hat, affine) #convert into NIfTI images
        nib_var = nib.Nifti1Image(U, affine)
        nib.save(nib_img, self.save_path + '/mean/' + filename)
        nib.save(nib_var, self.save_path + '/variance/' + filename)
        logging.info(f'{filename} : {self.avg_var[-1]}')

    def save_models(self, path, lr, optim, loss, epoch):
        model_dict = {}
        #Save sub-models
        for i, submodel in enumerate(self.UList):
            model_dict[f'model{i}_state_dict'] = submodel.state_dict()
            config_optimizer = self.configure_optimizers()
            model_dict[f'optimizer{i}_state_dict'] = config_optimizer['optimizer'].state_dict()
            model_dict['lr'] = lr
            model_dict['epoch'] = epoch
            model_dict['loss'] = loss
        torch.save(model_dict, path)

        

if __name__== "__main__":
    pl.seed_everything(42, workers=True)
    #parse arguments
    desc = 'Semantic Segmentation of brain MRA using U-net'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--save_path', type=str, default='result')
    parser.add_argument('--num_submodels', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=9)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--train_step', type=int, default=1)
    parser.add_argument('--Lambda', type=float, default=10.0)
    parser.add_argument('--ckpt', type=str, default='/p/scratch/delia-mp/rathore1/trained_models/MRA_ensemble1-lr=0.0001-epoch=200-val_loss=0.03-L=7-M=2')

    args = parser.parse_args()
    torch.set_float32_matmul_precision('medium')
    #Create instance of data module 
    data = CustomDataModule(args.data_path)
    #prepare train, test and validation datasets
    data.prepare_data()
    #define UNet model
    unet=UNet(spatial_dims=3, in_channels=1, out_channels=2, channels=(32, 64, 128, 256), strides =(2, 2, 2), act=Act.LEAKYRELU)
    tb_logger = pl_loggers.TensorBoardLogger(args.save_path + '/logs')
    epochs = 200
    #Loss functions
    if args.train_step == 1:
        Loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9]))
    else:
        Loss_fn = losses.uncertainty_loss(weight=torch.tensor([0.1, 0.9]), Lambda=args.Lambda)
    #Create instance of Ensemble for first training step
    ensemble = Ensemble(
        net=unet,
        criterion=Loss_fn,
        learning_rate=0.0001,
        optimizer_class=torch.optim.Adam,
        save_path=args.save_path,
        num_submodels=args.num_submodels,
        num_layers=args.num_layers,
        samples=args.samples,
        train_step=1,
        ckpt = args.ckpt,
    )

    #Intitalise pytorch Lightning trainer for first training step
    trainer = pl.Trainer(accelerator='gpu', devices = 2,
        #precision= 16 if torch.cuda.is_available() else 32,
        precision = 32,
        callbacks=[RichProgressBar(leave=True)],
        logger=tb_logger,
        max_epochs=epochs
    )
    #create file for logging workflow
    logging.basicConfig(filename = args.save_path + '/workflow.log', filemode='w',
                    level = logging.INFO,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    #Start training step1
    logging.info(f"Training step{args.train_step} started")    
    trainer.fit(ensemble, datamodule=data)
    logging.info(f"Training ended")
    logging.info(f"gpu used {torch.cuda.max_memory_allocated(device=None)} memory")
    logging.info(f"Saving models......")
    #Save sub-models
    save_path = args.save_path + f'MRA_ensemble{ensemble.train_step}-lr=0.0001-epoch={epochs}-val_loss={ensemble.avg_val_loss[-1]:.2f}-L={ensemble.num_layers}-M={ensemble.num_submodels}-T{ensemble.samples}-W{args.Lambda}'
    ensemble.save_models(save_path, ensemble.lr, ensemble.optimizer_class,ensemble.avg_train_loss[-1], epochs)
    #Create instance of Ensemble for second training step
    ensemble = Ensemble(
        net=unet,
        criterion=losses.uncertainty_loss(weight=torch.tensor([0.1, 0.9]), Lambda=args.Lambda),
        learning_rate=0.0001,
        optimizer_class=torch.optim.Adam,
        save_path=args.save_path,
        num_submodels=args.num_submodels,
        num_layers=args.num_layers,
        samples=args.samples,
        train_step=2,
        ckpt = save_path,
    )
    #Intitalise pytorch Lightning trainer for second training step
    trainer = pl.Trainer(accelerator='gpu', devices = 2,
        precision = 32,
        callbacks=[RichProgressBar(leave=True)],
        logger=tb_logger,
        max_epochs=100
    )
    #Start training step2
    logging.info('Training step 2 started')
    trainer.fit(ensemble, datamodule=data)
    logging.info(f"Training ended")
    logging.info(f"gpu used {torch.cuda.max_memory_allocated(device=None)} memory")
    logging.info(f'Number of submodels : {ensemble.num_submodels}')
    logging.info(f'Number of layers : {ensemble.num_layers}')
    logging.info(f'Number of iterations : {ensemble.samples}')
    logging.info(f'Lambda value : {args.Lambda}')
    logging.info(f'Validation loss : {ensemble.avg_val_loss[-1]:.2f}')
    param = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logging.info(f'Number of parameters : {param * ensemble.num_submodels}')
    #Start testing
    logging.info('testing started')
    trainer_test=pl.Trainer()
    trainer_test.test(ensemble, datamodule=data)
    logging.info('testing ended')
    
    #Plot avg variance
    avg_var = torch.tensor(ensemble.avg_var, device =  'cpu')
    img = range(1, len(avg_var)+1)
    fig, ax = plt.subplots()
    ax.plot(img, avg_var, 'b',marker='o')
    ax.set_title('Average Variance of Test Set')
    ax.set_xlabel('Image Index')
    ax.set_ylabel('Variance')
    ax.set_yscale('log')
    plt.savefig(args.save_path + '/avg_var.png')
    plt.show()
    plt.close()
    
    #Plot training and validation loss
    ep = range(1, len(ensemble.avg_train_loss)+1)
    plt.plot(ep, ensemble.avg_train_loss, 'b', label='Training Loss')
    plt.plot(ep, ensemble.avg_val_loss[1:], 'r', label='Validation Loss')
    plt.yscale('log') 
    plt.title('Training and Validation Loss ')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig(args.save_path + f'/losses-step{ensemble.train_step}-L{ensemble.num_layers}-M{ensemble.num_submodels}-T{ensemble.samples}-W{args.Lambda}.png')
    plt.close()

    #Plot test loss
    ep = range(1, len(ensemble.avg_test_loss)+1)
    plt.plot(ep, ensemble.avg_test_loss, 'b', label='Test Loss')
    plt.yscale('log') 
    plt.title('Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig(args.save_path + '/baseline-L4-test_loss.png')
    plt.close()

