from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import re
from torchvision.transforms.functional import InterpolationMode
import torchvision

class Boiling(data.Dataset):
    """Dataset class for the BRATS dataset."""

    def __init__(self, image_dir, mode, domain,in_sequence = True,image_size =256):
        """Initialize and Load the Boiling dataset."""
        self.image_dir = image_dir
        # self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self.domain = domain
        self.in_sequence = in_sequence
        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []
        self.load_data()
        # self.prepare_time_indices()
        # self.sorted_alphanumeric()
        # self.get_frame_no()

        

        if mode == 'train':
            self.num_images = len(self.train_dataset)

        elif mode == 'val':
            self.num_images = len(self.val_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def get_frame_no(self,filename):
        match = re.search(r'\d+', filename)

        # Checking if a number was found
        if match:
            frame_no = match.group()
            return int(frame_no)
            print("Extracted number:", frame_no)
        else:
            print("No frame_no found in the file name.")

    def sorted_alphanumeric(self,data):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(data, key=alphanum_key)

    def prepare_time_indices(self,index,data_len,in_sequence):
        if in_sequence:
            if index <= (data_len-3):
                i = index
            else:
                i = index - 2

            j = random.randint(i+1, data_len-2)
            k = random.randint(j+1, data_len-1)
        elif not in_sequence:
            excluded_numbers = []
            i = index
            excluded_numbers.append(i)

            j = random.randint(0, data_len-1)
            while j in excluded_numbers:
                j = random.randint(0, data_len-1)
            excluded_numbers.append(j)

            k = random.randint(0, data_len-1)
            while k in excluded_numbers or k>j>i:
                k = random.randint(0, data_len-1)
        
        return i,j,k

    def load_data(self):
        """Load dataset"""
        
        test_pos = self.sorted_alphanumeric(glob(os.path.join(self.image_dir, 'test', f'domain{self.domain}', '*jpg')))

        for filename in test_pos:
            if self.domain == "A":
                self.test_dataset.append([filename, [0]])
            else:
                self.test_dataset.append([filename, [1]])

        # Load val dataset
        val_pos = self.sorted_alphanumeric(glob(os.path.join(self.image_dir, 'val', f'domain{self.domain}', '*jpg')))

        for filename in val_pos:
            if self.domain == "A":
                self.val_dataset.append([filename, [0]])
            else:
                self.val_dataset.append([filename, [1]])

        # Load train dataset
        train_pos = self.sorted_alphanumeric(glob(os.path.join(self.image_dir, 'train', f'domain{self.domain}', '*jpg')))


        for filename in train_pos:
            if self.domain == "A":
                self.train_dataset.append([filename, [0]])
            else:
                self.train_dataset.append([filename, [1]])

        print('Finished loading the Boiling dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""

        if self.mode == 'train':
            dataset = self.train_dataset
        elif self.mode == 'val':
            dataset = self.val_dataset
        else:
            dataset = self.test_dataset  

        i,j,k = self.prepare_time_indices(index = index,data_len = len(dataset),in_sequence = self.in_sequence)


        transform = []
        if self.mode == 'train':
            transform.append(T.RandomHorizontalFlip())
        # transform.append(T.CenterCrop(crop_size))
        transform.append(T.Grayscale(num_output_channels=1))  # Convert to grayscale
        # transform.append(T.Resize(image_size))
        transform.append(T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.LANCZOS))

        transform.append(T.ToTensor())
        # transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform.append(T.Normalize(mean=(0.5,), std=(0.5,)))
        transform = T.Compose(transform)

        filename1, label1 = dataset[i]
        filename2, label2 = dataset[j]
        filename3, label3 = dataset[k]

        frame_no = [self.get_frame_no(filename1),self.get_frame_no(filename2),self.get_frame_no(filename3)]

        frame_no1 = self.get_frame_no(filename1)
        frame_no2 = self.get_frame_no(filename2)
        frame_no3 = self.get_frame_no(filename3)

        file_names = [filename1,filename2,filename3]
        # labels = [label1,label2,label3]

        label = label1 # doesn't matter which one you take becuase they all have the same domain label since they are from the same domian

        image1 = transform(Image.open(filename1))
        image2 = transform(Image.open(filename2))
        image3 = transform(Image.open(filename3))

        seq_img = torch.cat((image1, image2, image3), dim=0)
        seq_label = 1 if i<j<k else 0
        
        self.in_sequence = not(self.in_sequence) # flip the flag to balance the data_loading

        # return seq_img,seq_label,file_names,labels
        return seq_img, torch.FloatTensor(label), seq_label, file_names
        # return self.transform(seq_img), torch.FloatTensor(seq_label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, image_size=256, 
               batch_size=16, dataset='Boiling', mode='train', num_workers=1, domain = "A"):


    dataset = Boiling(image_dir, mode, domain)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  # shuffle=(mode=='train'),
                                  shuffle= True,
                                  num_workers=num_workers)

    return data_loader

