import os
import pickle
import numpy as np
import pandas as pd
import random
import torch
import math
from torch.autograd import Variable
import time
from Proprecessing import *
class Data_utils():

    def __init__(self,batch_size,seq_length, num_validation, infer=False):
        '''
        Initialiser function for the DataLoader class
        params:
        batch_size : Size of the mini-batch
        seq_length : Sequence length to be considered
        num_of_validation : number of validation dataset will be used
        infer : flag for test mode
        generate : flag for data generation mode
        forcePreProcess : Flag to forcefully preprocess the data again from csv files
        '''
        # base test files
        base_test_dataset = ['/data/test/biwi/biwi_eth.txt',
                             '/data/test/crowds/crowds_zara01.txt',
                             '/data/test/crowds/uni_examples.txt',
                             '/data/test/stanford/coupa_0.txt',
                             '/data/test/stanford/coupa_1.txt', '/data/test/stanford/gates_2.txt',
                             '/data/test/stanford/hyang_0.txt', '/data/test/stanford/hyang_1.txt',
                             '/data/test/stanford/hyang_3.txt', '/data/test/stanford/hyang_8.txt',
                             '/data/test/stanford/little_0.txt', '/data/test/stanford/little_1.txt',
                             '/data/test/stanford/little_2.txt', '/data/test/stanford/little_3.txt',
                             '/data/test/stanford/nexus_5.txt', '/data/test/stanford/nexus_6.txt',
                             '/data/test/stanford/quad_0.txt', '/data/test/stanford/quad_1.txt',
                             '/data/test/stanford/quad_2.txt', '/data/test/stanford/quad_3.txt'
                             ]
        # base train files
        base_train_dataset = [  # '/data/train/biwi/biwi_hotel.txt',
            #                 '/data/train/crowds/arxiepiskopi1.txt','/data/train/crowds/crowds_zara02.txt',
            #                 '/data/train/crowds/crowds_zara03.txt','/data/train/crowds/students001.txt','/data/train/crowds/students003.txt',
            #                 '/data/train/mot/PETS09-S2L1.txt',
            '/data/train/stanford/bookstore_0.txt'
            , '/data/train/stanford/bookstore_1.txt', '/data/train/stanford/bookstore_2.txt',
            '/data/train/stanford/bookstore_3.txt', '/data/train/stanford/coupa_3.txt',
            '/data/train/stanford/deathCircle_0.txt', '/data/train/stanford/deathCircle_1.txt',
            '/data/train/stanford/deathCircle_2.txt', '/data/train/stanford/deathCircle_3.txt',
            '/data/train/stanford/deathCircle_4.txt', '/data/train/stanford/gates_0.txt',
            '/data/train/stanford/gates_1.txt', '/data/train/stanford/gates_3.txt', '/data/train/stanford/gates_4.txt',
            '/data/train/stanford/gates_5.txt', '/data/train/stanford/gates_6.txt', '/data/train/stanford/gates_7.txt',
            '/data/train/stanford/gates_8.txt', '/data/train/stanford/hyang_4.txt',
            '/data/train/stanford/hyang_5.txt', '/data/train/stanford/hyang_6.txt', '/data/train/stanford/hyang_9.txt',
            '/data/train/stanford/nexus_0.txt', '/data/train/stanford/nexus_1.txt', '/data/train/stanford/nexus_2.txt',
            '/data/train/stanford/nexus_3.txt', '/data/train/stanford/nexus_4.txt', '/data/train/stanford/nexus_7.txt',
            '/data/train/stanford/nexus_8.txt', '/data/train/stanford/nexus_9.txt'
        ]
        # dimensions of each file set
        self.dataset_dimensions = {'biwi': [720, 576], 'crowds': [720, 576], 'stanford': [595, 326], 'mot': [768, 576]}
        # List of data directories where raw data resides
        self.base_train_path = 'data/train/'
        self.base_test_path = 'data/train/'
        self.base_validation_path = 'data/validation/'

        # check infer flag, if true choose test directory as base directory
        if infer is False:
            self.base_data_dirs = base_train_dataset
        else:
            self.base_data_dirs = base_train_dataset


        # if not infer mode, use train dataset
        if infer is False:
            self.data_path = self.base_train_path
        else:
            self.data_path = self.base_test_path




    def Datapath(self,):
        #core = './social-lstm-master/data/'
        paths = []
        folders = self.data_path
        subfolders = os.walk(self.data_path)
        for subpath, subfolder_list, subfile_list in subfolders:
            for subfolder in subfolder_list:
                for filename in os.listdir(self.data_path + '/' + subfolder):
                    if not filename.endswith(".txt"):
                        continue
                    path = self.data_path + '/' + subfolder + '/' + filename
                    paths.append(path)
        print(paths)
        return paths


    # method:1--rotate,2--reverse, 3--offset,4-- normalization, 5--standardzation 6
    def Dataprocess(self,data_path, methods):
        print(1)
        progress = 0
        df_all = pd.DataFrame(columns = ['frame_ID','pred_ID','y','x'])
        data_all = []
        for path in data_path:
            fname = path[path.rfind('/') + 1:path.rfind(".")]
            print(fname)
            progress += 1
            start_time = time.time()
            data = np.genfromtxt(path)
            x = data[:, 3]
            y = data[:, 2]
            peds = np.unique(data[:, 1])
            npeds = peds.shape[0]
            s = np.zeros((npeds,))
            pre = Proprecessing(data,x,y,peds,methods)
            data_new = pre.data_pre(data,x,y,peds,methods)
            #df = pd.DataFrame(data_new,columns = ['frame_ID','pred_ID','y','x'])
            #df_c = pd.merge(df,df_all)
            data_all.extend(data_new)
        return data_all
        #return df_all


























    def next_batch(self):
        '''
        Function to get the next batch of points
        '''
        # Source data
        x_batch = []
        # Target data
        y_batch = []
        # Dataset data
        d = []

        # pedlist per sequence
        numPedsList_batch = []

        # pedlist per sequence
        PedsList_batch = []

        # return target_id
        target_ids = []

        # Iteration index
        i = 0
        while i < self.batch_size:
            # Extract the frame data of the current dataset
            frame_data = self.data[self.dataset_pointer]
            numPedsList = self.numPedsList[self.dataset_pointer]
            pedsList = self.pedsList[self.dataset_pointer]
            # Get the frame pointer for the current dataset
            idx = self.frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            if idx + self.seq_length - 1 < len(frame_data):
                # All the data in this sequence
                seq_source_frame_data = frame_data[idx:idx + self.seq_length]
                seq_numPedsList = numPedsList[idx:idx + self.seq_length]
                seq_PedsList = pedsList[idx:idx + self.seq_length]
                seq_target_frame_data = frame_data[idx + 1:idx + self.seq_length + 1]

                # Number of unique peds in this sequence of frames
                x_batch.append(seq_source_frame_data)
                y_batch.append(seq_target_frame_data)
                numPedsList_batch.append(seq_numPedsList)
                PedsList_batch.append(seq_PedsList)
                # get correct target ped id for the sequence
                target_ids.append(
                    self.target_ids[self.dataset_pointer][math.floor((self.frame_pointer) / self.seq_length)])
                self.frame_pointer += self.seq_length

                d.append(self.dataset_pointer)
                i += 1

            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                self.tick_batch_pointer(valid=False)

        return x_batch, y_batch, d, numPedsList_batch, PedsList_batch, target_ids

    def next_valid_batch(self):
        '''
        Function to get the next Validation batch of points
        '''
        # Source data
        x_batch = []
        # Target data
        y_batch = []
        # Dataset data
        d = []

        # pedlist per sequence
        numPedsList_batch = []

        # pedlist per sequence
        PedsList_batch = []
        target_ids = []

        # Iteration index
        i = 0
        while i < self.batch_size:
            # Extract the frame data of the current dataset
            frame_data = self.valid_data[self.valid_dataset_pointer]
            numPedsList = self.valid_numPedsList[self.valid_dataset_pointer]
            pedsList = self.valid_pedsList[self.valid_dataset_pointer]

            # Get the frame pointer for the current dataset
            idx = self.valid_frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            if idx + self.seq_length < len(frame_data):
                # All the data in this sequence
                # seq_frame_data = frame_data[idx:idx+self.seq_length+1]
                seq_source_frame_data = frame_data[idx:idx + self.seq_length]
                seq_numPedsList = numPedsList[idx:idx + self.seq_length]
                seq_PedsList = pedsList[idx:idx + self.seq_length]
                seq_target_frame_data = frame_data[idx + 1:idx + self.seq_length + 1]

                # Number of unique peds in this sequence of frames
                x_batch.append(seq_source_frame_data)
                y_batch.append(seq_target_frame_data)
                numPedsList_batch.append(seq_numPedsList)
                PedsList_batch.append(seq_PedsList)
                # get correct target ped id for the sequence
                target_ids.append(
                    self.target_ids[self.dataset_pointer][math.floor((self.valid_frame_pointer) / self.seq_length)])
                self.valid_frame_pointer += self.seq_length

                d.append(self.valid_dataset_pointer)
                i += 1

            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                self.tick_batch_pointer(valid=True)

        return x_batch, y_batch, d, numPedsList_batch, PedsList_batch, target_ids

    def tick_batch_pointer(self, valid=False):
        '''
        Advance the dataset pointer
        '''

        if not valid:

            # Go to the next dataset
            self.dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.dataset_pointer >= len(self.data):
                self.dataset_pointer = 0
            print("*******************")
            print("now processing: %s" % self.get_file_name())
        else:
            # Go to the next dataset
            self.valid_dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.valid_frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.valid_dataset_pointer >= len(self.valid_data):
                self.valid_dataset_pointer = 0
            print("*******************")
            print("now processing: %s" % self.get_file_name(pointer_type='valid'))

    def reset_batch_pointer(self, valid=False):
        '''
        Reset all pointers
        '''
        if not valid:
            # Go to the first frame of the first dataset
            self.dataset_pointer = 0
            self.frame_pointer = 0
        else:
            self.valid_dataset_pointer = 0
            self.valid_frame_pointer = 0

    def switch_to_dataset_type(self, train=False, load_data=True):
        # function for switching between train and validation datasets during training session
        print('--------------------------------------------------------------------------')
        if not train:  # if train mode, switch to validation mode
            if self.additional_validation:
                print("Dataset type switching: training ----> validation")
                self.orig_seq_lenght, self.seq_length = self.seq_length, self.orig_seq_lenght
                self.data_dirs = self.validation_dataset
                self.numDatasets = len(self.data_dirs)
                if load_data:
                    self.load_preprocessed(self.data_file_vl, True)
                    self.reset_batch_pointer(valid=False)
            else:
                print("There is no validation dataset.Aborted.")
                return
        else:  # if validation mode, switch to train mode
            print("Dataset type switching: validation -----> training")
            self.orig_seq_lenght, self.seq_length = self.seq_length, self.orig_seq_lenght
            self.data_dirs = self.train_dataset
            self.numDatasets = len(self.data_dirs)
            if load_data:
                self.load_preprocessed(self.data_file_tr)
                self.reset_batch_pointer(valid=False)
                self.reset_batch_pointer(valid=True)

    def convert_proper_array(self, x_seq, num_pedlist, pedlist):
        # converter function to appropriate format. Instead of direcly use ped ids, we are mapping ped ids to
        # array indices using a lookup table for each sequence -> speed
        # output: seq_lenght (real sequence lenght+1)*max_ped_id+1 (biggest id number in the sequence)*2 (x,y)

        # get unique ids from sequence
        unique_ids = pd.unique(np.concatenate(pedlist).ravel().tolist()).astype(int)
        # create a lookup table which maps ped ids -> array indices
        lookup_table = dict(zip(unique_ids, range(0, len(unique_ids))))

        seq_data = np.zeros(shape=(self.seq_length, len(lookup_table), 2))

        # create new structure of array
        for ind, frame in enumerate(x_seq):
            corr_index = [lookup_table[x] for x in frame[:, 0]]
            seq_data[ind, corr_index, :] = frame[:, 1:3]

        return_arr = Variable(torch.from_numpy(np.array(seq_data)).float())

        return return_arr, lookup_table

    def add_element_to_dict(self, dict, key, value):
        # helper function to add a element to dictionary
        dict.setdefault(key, [])
        dict[key].append(value)

    def get_dataset_path(self, base_path, f_prefix):
        # get all datasets from given set of directories
        dataset = []
        dir_names = unique_list(self.get_all_directory_namelist())
        for dir_ in dir_names:
            dir_path = os.path.join(f_prefix, base_path, dir_)
            file_names = get_all_file_names(dir_path)
            [dataset.append(os.path.join(dir_path, file_name)) for file_name in file_names]
        return dataset

    def get_file_name(self, offset=0, pointer_type='train'):
        # return file name of processing or pointing by dataset pointer
        if pointer_type is 'train':
            return self.data_dirs[self.dataset_pointer + offset].split('/')[-1]

        elif pointer_type is 'valid':
            return self.data_dirs[self.valid_dataset_pointer + offset].split('/')[-1]

    def create_folder_file_dict(self):
        # create a helper dictionary folder name:file name
        self.folder_file_dict = {}
        for dir_ in self.base_data_dirs:
            folder_name = dir_.split('/')[-2]
            file_name = dir_.split('/')[-1]
            self.add_element_to_dict(self.folder_file_dict, folder_name, file_name)

    def get_directory_name(self, offset=0):
        # return folder name of file of processing or pointing by dataset pointer
        folder_name = self.data_dirs[self.dataset_pointer + offset].split('/')[-2]
        return folder_name

    def get_directory_name_with_pointer(self, pointer_index):
        # get directory name using pointer index
        folder_name = self.data_dirs[pointer_index].split('/')[-2]
        return folder_name

    def get_all_directory_namelist(self):
        # return all directory names in this collection of dataset
        folder_list = [data_dir.split('/')[-2] for data_dir in (self.base_data_dirs)]
        return folder_list

    def get_file_path(self, base, prefix, model_name='', offset=0):
        # return file path of file of processing or pointing by dataset pointer
        folder_name = self.data_dirs[self.dataset_pointer + offset].split('/')[-2]
        base_folder_name = os.path.join(prefix, base, model_name, folder_name)
        return base_folder_name

    def get_base_file_name(self, key):
        # return file name using folder- file dictionary
        return self.folder_file_dict[key]

    def get_len_of_dataset(self):
        # return the number of dataset in the mode
        return len(self.data)

    def clean_test_data(self, x_seq, target_id, obs_lenght, predicted_lenght):
        # remove (pedid, x , y) array if x or y is nan for each frame in observed part (for test mode)
        for frame_num in range(obs_lenght):
            nan_elements_index = np.where(np.isnan(x_seq[frame_num][:, 2]))

            try:
                x_seq[frame_num] = np.delete(x_seq[frame_num], nan_elements_index[0], axis=0)
            except ValueError:
                print("an error has been occured")
                pass

        for frame_num in range(obs_lenght, obs_lenght + predicted_lenght):
            nan_elements_index = x_seq[frame_num][:, 0] != target_id

            try:
                x_seq[frame_num] = x_seq[frame_num][~nan_elements_index]

            except ValueError:
                pass

    def clean_ped_list(self, x_seq, pedlist_seq, target_id, obs_lenght, predicted_lenght):
        # remove peds from pedlist after test cleaning
        target_id_arr = [target_id]
        for frame_num in range(obs_lenght + predicted_lenght):
            pedlist_seq[frame_num] = x_seq[frame_num][:, 0]

    def write_to_file(self, data, base, f_prefix, model_name):
        # write all files as txt format
        self.reset_batch_pointer()
        for file in range(self.numDatasets):
            path = self.get_file_path(f_prefix, base, model_name, file)
            file_name = self.get_file_name(file)
            self.write_dataset(data[file], file_name, path)

    def write_dataset(self, dataset_seq, file_name, path):
        # write a file in txt format
        print("Writing to file  path: %s, file_name: %s" % (path, file_name))
        out = np.concatenate(dataset_seq, axis=0)
        np.savetxt(os.path.join(path, file_name), out, fmt="%1d %1.1f %.3f %.3f", newline='\n')

    def write_to_plot_file(self, data, path):
        # write plot file for further visualization in pkl format
        self.reset_batch_pointer()
        for file in range(self.numDatasets):
            file_name = self.get_file_name(file)
            file_name = file_name.split('.')[0] + '.pkl'
            print("Writing to plot file  path: %s, file_name: %s" % (path, file_name))
            with open(os.path.join(path, file_name), 'wb') as f:
                pickle.dump(data[file], f)

    def get_frame_sequence(self, frame_lenght):
        # begin and end of predicted fram numbers in this seq.
        begin_fr = (self.frame_pointer - frame_lenght)
        end_fr = (self.frame_pointer)
        frame_number = self.orig_data[self.dataset_pointer][begin_fr:end_fr, 0].transpose()
        return frame_number

    def get_id_sequence(self, frame_lenght):
        # begin and end of predicted fram numbers in this seq.
        begin_fr = (self.frame_pointer - frame_lenght)
        end_fr = (self.frame_pointer)
        id_number = self.orig_data[self.dataset_pointer][begin_fr:end_fr, 1].transpose()
        id_number = [int(i) for i in id_number]
        return id_number

    def get_dataset_dimension(self, file_name):
        # return dataset dimension using dataset file name
        return self.dataset_dimensions[file_name]


def CV(x_seq, Pedlist, args, true_x_seq, true_Pedlist, dimensions, dataloader, look_up,
       num_pedlist, is_gru, grid=None):
    '''
    The sample function
    params:
    x_seq: Input positions
    Pedlist: Peds present in each frame
    args: arguments
    net: The model
    true_x_seq: True positions
    true_Pedlist: The true peds present in each frame
    saved_args: Training arguments
    dimensions: The dimensions of the dataset
    target_id: ped_id number that try to predict in this sequence
    '''
    # Number of peds in the sequence
    numx_seq = len(look_up)

    ret_x_seq = Variable(torch.zeros(args.obs_length + args.pred_length, numx_seq, 2))

    # Initialize the return data structure
    if args.use_cuda:
        ret_x_seq = ret_x_seq.cuda()
    ret_x_seq[:args.obs_length, :, :] = x_seq.clone()

    # assign last position of observed data to temp
    # temp_last_observed = ret_x_seq[args.obs_length-1].clone()
    # ret_x_seq[args.obs_length-1] = x_seq[args.obs_length-1]

    # For the predicted part of the trajectory
    next_x = x_seq[args.obs_length - 1, :, 0]
    next_y = x_seq[args.obs_length - 1, :, 1]
    for tstep in range(args.obs_length - 1, args.pred_length + args.obs_length - 1):
        delt_x = x_seq[args.obs_length - 1, :, 0] - x_seq[args.obs_length - 2, :, 0]
        delt_y = x_seq[args.obs_length - 1, :, 1] - x_seq[args.obs_length - 2, :, 1]

        next_x = next_x + delt_x
        next_y = next_y + delt_y
        ret_x_seq[tstep + 1, :, 0] = next_x
        ret_x_seq[tstep + 1, :, 1] = next_y

    return ret_x_seq




