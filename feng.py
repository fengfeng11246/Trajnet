import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as girdspec
import imageio
import os
import time
from scipy.stats import gaussian_kde
import matplotlib.image as img
from PIL import Image

path = '/home/fengfeng/share/StanfordDroneDataset-master/deathCircle/video0/annotations.txt'
start_time = time.time()
data = np.genfromtxt(path)

x = (data[0:10000,1]+data[0:10000,3])/2
y = (data[0:10000,2]+data[0:10000,4])/2
# x = (data[:,1]+data[:,3])/2
# y = (data[:,2]+data[:,4])/2
# # Make the plot
# xy = np.vstack([x,y])
# print('1')
# z = gaussian_kde(xy) (xy)
# print('2')
# idx = z.argsort()
# x, y, z = x[idx], y[idx], z[idx]
imgP=Image.open('/home/fengfeng/share/social-lstm-master/data/validation/stanford/0.png')
# #fig.figimage(imgP)   #背景图片
# #ax1 = fig.add_axes(axis)
a=int(x.max()-x.min())
b= int (y.max()-y.min())
imgP.resize((1500,1500))
plt.imshow(imgP)

# plt.scatter(x, y, c=z,s=0.05, alpha=0.4)
# plt.colorbar()
# plt.show()
# plt.close()

xy = np.vstack([x,y])
z = gaussian_kde(xy) (xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
z = (z-z.min())/(z.max()-z.min())
# Make the plot
# indx =  np.where(z>=0.00000005)
# x2 = x[indx]
# y2 = y[indx]
# z2 = z[indx]
#plt.pcolormesh(x2, y2, z2.reshape(x2.shape))
#plt.scatter(x2,y2,c=z2, s=0.05, alpha=0.4)

cm = plt.cm.get_cmap('jet')
plt.scatter(x,y,c=z, s=0.05, alpha=0.4,cmap = cm)
plt.colorbar()

plt.savefig('test.png',bbox_inches = 'tight',dpi = 700)
plt.show()