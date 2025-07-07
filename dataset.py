#!/usr/bin/env python
# coding: utf-8

from torch.utils.data import DataLoader
from pathlib import Path
import torchio as tio
import numpy as np
import pytorch_lightning as pl

class CustomDataModule(pl.LightningDataModule):
    
    def __init__(self, data_path, patch_size=None):
        super().__init__()
        self.data_path = Path(data_path)
        self.train_subjects = []
        self.val_subjects = []
        self.test_subjects = []
        self.preprocess = None
        self.transform = None
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.label_test_paths=None
        self.patch_size = [128, 128, 80] #patch-size for patch based training
        self.queue_length = 300
        self.samples_per_volume = 10 #default no. of patches to extract from each volume
        self.sampler = tio.data.LabelSampler(patch_size=self.patch_size) #yields patches whose center value is greater than 0 in the label_name
        

    def get_max_shape(self, subjects):
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.spatial_shape for s in dataset])
        return shapes.max(axis=0)

    def download_data(self):
        def get_niis(d):
            return sorted(p for p in d.glob('*.nii*'))


        image_training_paths = get_niis(self.data_path /'train_input')
        label_training_paths = get_niis(self.data_path /'train_label')
        image_val_paths = get_niis(self.data_path / 'val_input')
        label_val_paths = get_niis(self.data_path / 'val_label')
        image_test_paths = get_niis(self.data_path / 'test_input')
        self.label_test_paths = get_niis(self.data_path / 'test_label')

        return image_training_paths, label_training_paths, image_val_paths, label_val_paths, image_test_paths, self.label_test_paths


    def prepare_data(self):
        image_training_paths, label_training_paths, image_val_paths, label_val_paths, image_test_paths, label_test_paths = self.download_data()
        
        self.train_subjects = []
        for image_path, label_path in zip(image_training_paths, label_training_paths):

            subject = tio.Subject(
                image=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path)
            )
            self.train_subjects.append(subject)

        self.val_subjects = []
        for image_path, label_path in zip(image_val_paths, label_val_paths ):

            subject = tio.Subject(
                image=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path)
            )
            self.val_subjects.append(subject)
        
        self.test_subjects = []
        for image_path, label_path in zip(image_test_paths, label_test_paths):
            subject = tio.Subject(image=tio.ScalarImage(image_path),label=tio.LabelMap(label_path), name=((Path(label_path).stem).split('.'))[0], affine=tio.LabelMap(label_path).affine,shape=tio.ScalarImage(image_path).shape)
            self.test_subjects.append(subject)



    def get_preprocessing_transform(self):
        preprocess = tio.Compose([
            tio.RescaleIntensity((0, 1))  #Normalize the images
        ])

        test_preprocess=tio.Compose([
            tio.RescaleIntensity((0, 1)),  #Normalize the images
            tio.EnsureShapeMultiple(16, method = 'pad')  # testing is performed on whole images, ensure correct shape for U-Net
            ])
        
        return preprocess, test_preprocess
    
    def get_augmentation_transform(self):
        #Augmetations based on nnU-Net 
        
        augment = tio.Compose([
            tio.RandomFlip(axes=[0,1],flip_probability=0.5 ),
            tio.RandomGamma(p=0.5),
            tio.RandomMotion(p=0.1),
            ])
        
        
        return augment   

    def setup(self, stage=None):
        self.preprocess, test_preprocess = self.get_preprocessing_transform()
        augment = self.get_augmentation_transform()
        self.transform = tio.Compose([self.preprocess, augment])
        self.train_set = tio.SubjectsDataset(self.train_subjects, transform=self.transform)
        self.val_set = tio.SubjectsDataset(self.val_subjects, transform=self.preprocess)
        self.test_set = tio.SubjectsDataset(self.test_subjects, transform=test_preprocess)
   
    
    def train_dataloader(self):
        
        patches_queue = tio.Queue(
        self.train_set,
        self.queue_length,
        self.samples_per_volume,
        self.sampler,
        num_workers=4)

        patches_loader = DataLoader(patches_queue, batch_size=16, num_workers=0,)
        
        return patches_loader
        
    def val_dataloader(self):
        
        patches_queue = tio.Queue(
        self.val_set,
        self.queue_length,
        self.samples_per_volume,
        self.sampler,
        num_workers=4)

        patches_loader = DataLoader(patches_queue, batch_size=16, num_workers=0,)
            
        return patches_loader
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=1)

    


 
    
    
        
