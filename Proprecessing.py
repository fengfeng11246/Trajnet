import torch
import numpy as np
from torch.autograd import Variable
import math

import argparse
import os
import time
import pickle
import subprocess
import random



class Proprecessing():
    # class for data augmentation

    def __init__(self, data,data_x,data_y,peds,methods):

        self.base_train_path = 'data/train/'
        self.base_validation_path = 'data/validation/'

        # list of angles will be use for rotation
        self.angles = list(range(0, 360, 30))
        self.dataloader = data

    def data_pre(self,data,data_x,data_y,peds,methods):
        for method in methods:
            if method == 1:
                data[:,3],data[:,2] = self.dataset_min_max_normalization(data,data_x,data_y)
            elif method == 2:
                data[:,3],data[:,2] = self.dataset_mean_max_normalization(self,data,data_x,data_y)
            elif method == 6:
                data[:,3],data[:,2] = self.dataset_z_standardized(self,data,data_x,data_y)

            elif method == 3 or 4 or 5:
                npeds = peds.shape[0]
                first_values_dict = []
                data_new =[]
                for i in range(npeds):
                    pts = data[np.where(data[:, 1] == peds[i])]
                    xys = pts[:, 2:]
                    x_data = pts[:, 3]
                    y_data = pts[:, 2]
                    # vs = np.sqrt(np.sum(np.square(xys[1:, :] - xys[:-1, :]), axis=1))
                    # s[i] = np.average(vs)
                    if method == 3:
                        x,y = self.dataset_rotation(self,xys,data_x,data_y)
                        new = [pts[:, 0:1],y,x]
                        data_new.append(new)
                    elif method ==4:
                        x,y = self.dataset_reverse(self,xys,data_x,data_y)
                        new = [pts[:, 0:1], y, x]
                        data_new.append(new)
                    elif method == 5:
                        data[:,3],data[:,2],first_values_dict = self.dataset_offset(self, xys, data_x, data_y)
                        first_values_dict.append(first_values_dict)

                data.append(data_new)

        return data





    def dataset_min_max_normalization(self,data,data_x,data_y):#[0,1]
        print(1)
        data_x = (data_x-np.min(data_x))/(np.max(data_x)- np.min(data_x))
        data_y =  (data_y-np.min(data_y))/(np.max(data_y)- np.min(data_y))
        return data_x,data_y
    def dataset_mean_max_normalization(self,data,data_x,data_y):#[-1,1]
        print(2)
        data_x = (data_x-np.mean(data_x))/(np.max(data_x)- np.min(data_x))
        data_y =  (data_y-np.mean(data_y))/(np.max(data_y)- np.min(data_y))
        return data_x,data_y
    def dataset_z_standardized(self,data,data_x,data_y):#mean = 0 var = 1
        print(6)
        data_x = (data_x - np.mean(data_x)) / np.std(data_x)
        data_y = (data_y - np.mean(data_y)) / np.std(data_y)
        return data_x, data_y
    def dataset_rotation(self,data,data_x,data_y):
        print(3)
        """
                Rotate a point counterclockwi se by a given angle around a given origin.

                The angle should be given in radians.
                """
        #p1 = reference_point
        p1 = (0,1)
        p2 = (data_x[0].data.numpy(), data_y[0].data.numpy())
        angle = (np.arctan2(*p1[::-1])-np.arctan2(*p2[::-1]))%(2*np.pi)
        origin = (0, 0)
        for ind in range(len(data)):
            point = data[ind]
            rotated_point = rotate(origin, point, angle)
            data_x[ind] = rotated_point[0]
            data_y[ind] = rotated_point[1]
        return data_x,data_y


    def dataset_reverse(self,data,data_x,data_y):
        print(4)
        reverse_data = data.clone()
        length = len(data)
        for ind, frame in enumerate(data):

            reverse_data[len(data)-ind-1,:] = data[ind,:]
        return reverse_data[:,0],reverse_data[:,1]

    def dataset_offset(self,data,data_x,data_y):
        print(6)
        first_values_dict = data[0, :]
        vectorized_x_seq = data.clone()
        for ind, frame in enumerate(data):

            vectorized_x_seq[ind, :] = data[ind,:] - first_values_dict

        return vectorized_x_seq[:,0], vectorized_x_seq[:,1],first_values_dict
    def dataset_offset_retrurn(self,data,data_x,data_y):
        print(6)
        first_values_dict = data[0, :]
        vectorized_x_seq = data.clone()
        for ind, frame in enumerate(data):
            vectorized_x_seq[ind, :] = data[ind, :] + first_values_dict

        return vectorized_x_seq[:,0], vectorized_x_seq[:,1],first_values_dict
    def rotate(origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        # return torch.cat([qx, qy])
        return [qx, qy]






