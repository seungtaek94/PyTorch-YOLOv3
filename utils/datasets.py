import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        #print(h_factor, w_factor)
        
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            #boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 16))
            
            #print(boxes.shape)
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            
            # Extract Axle cord for uppadded + upscaled image
            axle = []
            for i in range(6, len(boxes)):
                if i % 2 == 0:
                    axle.append(w_factor * box[:, i])
                else:
                    axle.append(h_factor * box[:, i])
            '''            
            xAxle1 = w_factor * boxes[:, 6]
            yAxle1 = h_factor * boxes[:, 7]
            xAxle2 = w_factor * boxes[:, 8]
            yAxle2 = h_factor * boxes[:, 9]
            xAxle3 = w_factor * boxes[:, 10]
            yAxle3 = h_factor * boxes[:, 11]
            xAxle4 = w_factor * boxes[:, 12]
            yAxle4 = h_factor * boxes[:, 13]
            xAxle5 = w_factor * boxes[:, 14]
            yAxle5 = h_factor * boxes[:, 15]
            '''
            
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            
            for i in range(len(axle)):
                if i % 2 == 0:
                    axle[i] += pad[0]
                else:
                    axle[i] += pad[2]
                    
            '''
            xAxle1 += pad[0]
            yAxle1 += pad[2]
            xAxle2 += pad[0]
            yAxle2 += pad[2]
            xAxle3 += pad[0]
            yAxle3 += pad[2]
            xAxle4 += pad[0]
            yAxle4 += pad[2]
            xAxle5 += pad[0]
            yAxle5 += pad[2]
            '''
            
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h
            
            # Returns axle(xAxle1, yAxle1, xAxle2, yAxle2, xAxle3, yAxle3, xAxle4, yAxle4, xAxle5, yAxle5)
            for i in range(6, len(boxes)):
                if i % 2 == 0:
                    boxes[:, i] = axle[i-6] / padded_w
                else:
                    boxes[:, i] = axle[i-6] / padded_h
            '''        
            boxes[:, 6] = xAxle1 / padded_w
            boxes[:, 7] = yAxle1 / padded_h
            boxes[:, 8] = xAxle2 / padded_w
            boxes[:, 9] = yAxle2 / padded_h
            boxes[:, 10] = xAxle3 / padded_w
            boxes[:, 11] = yAxle3 / padded_h
            boxes[:, 12] = xAxle4 / padded_w
            boxes[:, 13] = yAxle4 / padded_h
            boxes[:, 14] = xAxle5 / padded_w
            boxes[:, 15] = yAxle5 / padded_h
            '''
            
            #print(len(boxes))
            #targets = torch.zeros((len(boxes), 6))
            targets = torch.zeros((len(boxes), 17))
            targets[:, 1:] = boxes
            
            #print(targets)

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
