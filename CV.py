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

def CV(x_seq, obj_seq,pre_seq,method):
    ret_x_seq =np.zeros(shape=(obj_seq + pre_seq, 2))

    ret_x_seq[:obj_seq, :] = x_seq[:obj_seq, :]

    next_x = x_seq[obj_seq - 1 , 0]
    next_y = x_seq[obj_seq- 1 , 1]
    # delt_x = np.zeros(shape=(obj_seq-1 ,1))
    # delt_y = np.zeros(shape=(obj_seq - 1, 1))
    # for i in range(obj_seq-1):
    #     delt_x[i] = x_seq[i+1, 0] - x_seq[i, 0]
    #     delt_y[i] = x_seq[i+1, 1] - x_seq[i, 1]
    # delt_x[0] = x_seq[obj_seq - 1, 0] - x_seq[obj_seq - 2, 0]
    # delt_y[0] = x_seq[obj_seq - 1, 1] - x_seq[obj_seq - 2, 1]
    #######################
    cal_x = (x_seq[obj_seq-1,0]-x_seq[obj_seq-1-method,0])/method
    cal_y = (x_seq[obj_seq - 1,1] - x_seq[obj_seq - 1 - method,1]) / method
    a_x = (x_seq[obj_seq-1,0]-x_seq[obj_seq-3,0]+x_seq[obj_seq-2,0]-x_seq[obj_seq-4,0])/4
    a_y = (x_seq[obj_seq-1,1]-x_seq[obj_seq-3,1]+x_seq[obj_seq - 2,1] - x_seq[obj_seq - 4,1]) /4
    #####################
    # if method == 1:
    #     cal_x = delt_x[obj_seq-2]
    #     cal_y = delt_y[obj_seq-2]
    # elif method == 2:
    #     cal_x = delt_x[obj_seq - 2]
    #     cal_y = delt_y[obj_seq - 2]
    T=0
    for tstep in range(obj_seq - 1, pre_seq + obj_seq - 1):
        T=T+1
        next_x = next_x+cal_x
        next_y = next_y+cal_y
        ret_x_seq[tstep + 1,  0] = next_x
        ret_x_seq[tstep + 1, 1] = next_y
        # ret_x_seq[tstep + 1, 0] = next_x + cal_x * T + 0.5 * a_x * T * T
        # ret_x_seq[tstep + 1, 1] = next_y + cal_y * T + 0.5 * a_y * T * T
    return ret_x_seq

def Histogram_error(s,fname):
    plt.rc('font', family='Times New Roman',size = '10',)
    kwargs = dict(histtype='bar', alpha=1, normed=0, bins=40)
    plt.hist(s,**kwargs,edgecolor = 'k')
    plt.title(fname)
    plt.xlabel("FDE/m")
    dir_path = './out/speed_hist/'
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    plt.savefig(dir_path+fname+'.png',bbox_inches = 'tight',dpi = 700)
    #plt.savefig('./out/speed_hist/'+fname+'.png',bbox_inches='tight')
    plt.close()



def write_pred_txt(file_name, read_file_path, log_dir_path, log_content,count): # read and log files will be of the same names but in different directories
    #file_name =
    log_file = open(os.path.join(log_dir_path, file_name,), 'w')   # log_content: [np.array(2, T), np.array(2, T), ...] in physical coordinates
    line_n = -1+count*20 # line count
    pred_n = -1
    with open(read_file_path, 'r') as f:
        # for line in f:
        #     line_n += 1
        #     if line_n%20 in range(0, 8):  # 0~7 # when in observation sequence
        #         log_file.write(line)
        #         if line_n%20 == 7:
        #             pred_n += 1 # starts from 0
        #     else: # when in prediction sequence
        #         xy_array = log_content[line_n%20,:] # np.array(2, T)
        #         ind = line_n%20 # 0~11
        #         line = line.strip().split(' ')
        #         log_file.write(line[0]+' '+line[1]+' '+str(xy_array[0])+' '+str(xy_array[1])+'\n')
        for ind,line in enumerate(f):
            line_n += 1
            line = line.strip().split(' ')
            log_file.write(line[0]+' '+line[1]+' '+str(log_content[ind,0])+' '+str(log_content[ind,1])+'\n')

    log_file.close()


#access file route
core = '/home/fengfeng/share/Train/'
paths =[]
log_paths = []
filenames = []
folders = os.walk(core)
for path,folder_list,file_list in folders:
    for folder in folder_list:
        if folder != 'test':
            continue
        subfolders = os.walk(core+folder)
        for subpath,subfolder_list,subfile_list in subfolders:
            for subfolder in subfolder_list:
                if subfolder != 'crowds' :
                    continue
                for filename in os.listdir(core+folder+'/'+subfolder):
                    if not filename.endswith(".txt"):
                        continue
                    path =core+folder+'/'+subfolder+'/'+filename
                    log_path = core+'result'+'/'+subfolder+'/'
                    paths.append(path)
                    log_paths.append(log_path)
                    filenames.append(filename)
print(paths)
print(log_paths)
print(filenames)
file_num = len(paths)
progress = 0
sequence = 200
obj_seq = 8
pre_seq = 12
error_all = 0
error_allf = 0
methods = [1]



for method in methods:
    print("***************method:%d**************"%method)
    error_all = 0
    error_allf = 0
    npeds_all = []
    fde_error_all = []
    count_bad = 0
    count_all = 0
    count_good = 0
    for ind,path in enumerate(paths):
        fname = path[path.rfind('/')+1:path.rfind(".")]
        log_path = log_paths[ind]
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        #print(fname)
        progress += 1
        start_time = time.time()
        data = np.genfromtxt(path)
        x = data[:,3]
        y = data[:,2]
        #peds = np.unique(data[:, 1])
        peds = []
        for i in data[:, 1]:
            if not i in peds:
                peds.append(i)
        peds = np.array(peds)
        npeds = peds.shape[0]
        s = np.zeros((npeds,))
        count = -1
        pre_xys_all = []

        for i in range(npeds):
            pts = data[np.where(data[:, 1] == peds[i])]
            xys = pts[:, 2:]
            pre_xys = CV(xys,obj_seq,pre_seq,method)
            pre_xys_all.extend(pre_xys)

            count+=1
        pre_xys_all = np.array(pre_xys_all)
        filename = filenames[ind]
        write_pred_txt(filename, path, log_path, pre_xys_all,count)


    print("*******************************************8**************" )

# for method in methods:
#     print("***************method:%d**************"%method)
#     error_all = 0
#     error_allf = 0
#     npeds_all = []
#     fde_error_all = []
#     count_bad = 0
#     count_all = 0
#     count_good = 0
#     for path in paths:
#         fname = path[path.rfind('/')+1:path.rfind(".")]
#         #print(fname)
#         progress += 1
#         start_time = time.time()
#         data = np.genfromtxt(path)
#         x = data[:,3]
#         y = data[:,2]
#         peds = np.unique(data[:, 1])
#         npeds = peds.shape[0]
#         s = np.zeros((npeds,))
#         error_file = 0
#         error_filef = 0
#         for i in range(npeds):
#             pts = data[np.where(data[:, 1] == peds[i])]
#             xys = pts[:, 2:]
#             pre_xys = CV(xys,obj_seq,pre_seq,method)
#             error = np.sqrt(np.sum(np.square(pre_xys-xys),axis = 1))
#             mean_error =  np.sum(error)/pre_seq
#             fde_error = np.sqrt(np.sum(np.square(pre_xys[-1,:]-xys[-1,:])))
#             if fde_error <= 20:
#                 fde_error_all.append(fde_error)
#             error_file += mean_error
#             error_filef += fde_error
#             if fde_error <= 0.5:
#                 count_good += 1
#             if fde_error >= 5:
#                 count_bad += 1
#             count_all += 1
#         avgerror_file = error_file/npeds
#         avgerror_filef = error_filef / npeds
#         npeds_all.append(npeds)
#         axis = [np.min(fde_error_all), np.max(fde_error_all)]
#
#         # print('error_file_ADE: %0.3f'%avgerror_file)
#         # print('error_file_FDE: %0.3f'%avgerror_filef)
#         error_all += error_file
#         error_allf += error_filef
#     print("count_all",count_all)
#     print("count_good", count_good)
#     print("count_bad", count_bad)
#     print(count_good/count_all)
#     print(count_bad / count_all)
#     #Histogram_error(fde_error_all, 'SDD'+str(method))
#     print("the number of data",np.sum(npeds_all))
#     error_all = error_all/np.sum(npeds_all)
#     error_allf = error_allf/np.sum(npeds_all)
#     print(' The totle error:')
#     print('error_all_ADE: %0.3f'%error_all)
#     print('error_all_FDE: %0.3f'%error_allf)
#     print("*******************************************8**************" )



