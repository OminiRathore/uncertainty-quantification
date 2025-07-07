#!/usr/bin/env python
# coding: utf-8
import argparse
import numpy as np
from dataset import CustomDataModule
import torch
import torch.nn as nn
import monai
from monai.networks.nets import UNet
from monai.networks.layers.factories import Act
from monai.transforms import KeepLargestConnectedComponent
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchio as tio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import RichProgressBar
monai.config.print_config()
monai.utils.set_determinism()
import nibabel as nib
import logging

class Baseline(pl.LightningModule):
    def __init__(self, net, criterion, learning_rate, optimizer_class, save_path):
        super().__init__()
        self.lr = learning_rate
        self.net = net
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.avg_train_loss=[]
        self.avg_val_loss=[]
        self.all_dices=[]
        self.losses=[]
        self.val_losses=[]
        self.avg_test_loss=[]
        self.save_path = save_path
   
    def forward(self, x):
        return self._model(x)
    
    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr, weight_decay=1e-5)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.2, patience = 15, threshold= 0.005, threshold_mode = 'abs' ),
            "monitor": "train_loss",
            }
        }

    def prepare_batch(self, batch):
        return batch['image'][tio.DATA], batch['label'][tio.DATA]
    
    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)
        x = x.float().cuda()
        y_hat = self.net(x)
        y=(torch.squeeze(y, 1)).long()
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat,y)
        self.losses.append(loss.item())
        self.log('train_loss', loss.item(), prog_bar=True, logger=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=16)
        return loss
    
    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        val_loss = self.criterion(y_hat,y)
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
    
    def test_step(self, batch, batch_idx):
            y_hat, y = self.infer_batch(batch)
            test_loss = model.criterion(y_hat, y)
            self.avg_test_loss.append(test_loss.item())
            y_hat = y_hat.argmax(dim=1, keepdim=False) # convert logits to classes 0 and 1
            batch['shape'] = [int(x.cpu().item())  for x in batch['shape']]
            crop = tio.CropOrPad(batch['shape'][1:]) #crop output to original shape
            y_hat = crop(y_hat.cpu().int())
            connect_comp = KeepLargestConnectedComponent() #remove small segments
            y_hat = connect_comp(y_hat)
            y_hat = y_hat.squeeze(0).numpy().astype(np.int16)
            filename = batch['name'][0]
            affine = batch['affine'].cpu().squeeze(0).numpy()
            nib_img = nib.Nifti1Image(y_hat, affine) #convert into NIfTI image
            nib.save(nib_img, self.save_path + '/output_label/' + filename) #save outputs
            
        
        

if __name__== "__main__":
    pl.seed_everything(42, workers=True)
    desc = 'Semantic Segmentation of brain MRA using U-net'
    #parse arguments
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--save_path', type=str, default='result')
    args = parser.parse_args()
    torch.set_float32_matmul_precision('medium')
    #Create instance of data module 
    data = CustomDataModule(args.data_path)
    #prepare train, test and validation datasets
    data.prepare_data()
    #define UNet model
    unet=UNet(spatial_dims=3, in_channels=1, out_channels=2, channels=(32, 64, 128, 256), strides =(2, 2, 2), act=Act.LEAKYRELU)
    tb_logger = pl_loggers.TensorBoardLogger(args.save_path + '/logs')
    #Create instance of Baseline
    model = Baseline(
        net=unet,
        criterion=nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9])),
        learning_rate=0.0001,
        optimizer_class=torch.optim.Adam,
        save_path=args.save_path
    )
    #save model checkpoints every 50 epochs
    checkpoint_callback = ModelCheckpoint(every_n_epochs=49, save_top_k=1, monitor="val_loss", mode="min", dirpath= args.save_path + '/trained_models', filename='MRA baseline-64-L4-{epoch}-{val_loss:.2f}')
    #Intitalise pytorch Lightning trainer
    trainer = pl.Trainer(accelerator='gpu', devices = 1,
    precision=16 if torch.cuda.is_available() else 32,
    callbacks=[checkpoint_callback, RichProgressBar(leave=True)],
    logger=tb_logger,
    max_epochs=200
    )
    trainer.logger._default_hp_metric = False
    #create file for logging workflow
    logging.basicConfig(filename = args.save_path + '/workflow-baseline-64-L4.log', filemode='w',
                level = logging.INFO, 
                format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    trainer.logger._default_hp_metric = False
    #Start training
    logging.info(f"Training started")    
    trainer.fit(model, datamodule=data)
    logging.info(f"Training ended")
    
    logging.info(f"gpu used {torch.cuda.max_memory_allocated(device=None)} memory")
    logging.info(f'Validation loss : {model.avg_val_loss[-1]:.2f}')
    param = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logging.info(f'Number of parameters : {param}')
    trainer_test=pl.Trainer(accelerator='gpu', devices = 1)
    #uncomment following commenting to test using pre-trained model
    #checkpoint="/mnt/big_disk/o.rathore/trained_models/bigpatch_MRA -epoch=48-val_loss=0.11.ckpt" 
    #model = Model.load_from_checkpoint(checkpoint, net=unet, criterion=nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0])))

    #Start testing
    logging.info("Inference started")
    trainer_test.test(model, datamodule=data)
    logging.info("Inference ended")
    
    
    #Plot test loss
    ep = range(1, len(model.avg_test_loss)+1)
    plt.plot(ep, model.avg_test_loss, 'b', label='Test Loss')
    plt.yscale('log') 
    plt.title('Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig(args.save_path + '/baseline-L4-test_loss.png')
    plt.close()

    #Plot training and validation loss
    ep = range(1, len(model.avg_train_loss)+1)
    plt.plot(ep, model.avg_train_loss, 'b', label='Training Loss')
    plt.plot(ep, model.avg_val_loss[1:], 'r', label='Validation Loss')
    plt.title('Training and Validation Loss ')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig(args.save_path + '/baseline-64-L4-losses.png')
    plt.close()
