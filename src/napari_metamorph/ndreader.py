# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:58:05 2023

@author: Jean
"""
import os
from skimage.io import imread
import numpy as np
from dask import delayed
import dask.array as da
import re

class NdExp:
    '''This class is creating an object which contains all parameters of the .nd file from Metamorph. 
    it requires two other classes (below): WavePointCollection and Event
    it also contains the function build_layers which returns a proper list of layers to be read by napari'''
    
    def __init__(self):
        self.is_nd = False
        self.bad_nd_type = False
        self.do_timelapse = False
        self.do_stage = False
        self.do_wave = False
        self.do_wave_points_collection = False
        self.do_z_series = False
        self.wave_in_file_name = False
        self.n_time_points = 1
        self.n_stage_positions = 1
        self.n_wavelengths = 1
        self.n_wave_points_collection = 0
        self.n_z_steps = 0
        self.n_events = 0
        self.z_step_size = np.double(0.0)
        self.origin = ''
        self.destination = ''
        self.version = ''
        self.description = ''
        self.start_time = ''
        self.image_name = ''
        self.extension_for_stack = ".tif"
        self.position_name = []
        self.wave_name = []
        self.file_name_list = []
        self.wave_points_collection = []
        self.event = []
        self.wave_do_z = []
        self.file_name = ""
        self.n_digits = 1
        self.nd_dictionary = dict()
        self.image_layers=[]
        self.folder_path=''

    def nd_reader(self, path):
        '''Takes a path and fills the object with good attributes'''
        dictionary = dict()
        try:
            with open(path, 'r') as file:
                for line in file:
                    if len(line) > 1:
                        pre_name = line.rstrip().split(', ')
                        if pre_name[0][-1] == ',':
                            pre_name = line.rstrip().split(',')
                        if pre_name[0][0] == '"' and pre_name[0][-1] == '"':
                            name = pre_name[0]
                            if len(pre_name) > 2:
                                elem = ','.join(pre_name[1:])
                            else:
                                elem = pre_name[-1]
                            if name == '"WavePointsCollected"':
                                if name in dictionary:
                                    wp = dictionary[name]
                                    dictionary[name] = wp + [elem]
                                else:
                                    dictionary[name] = [elem]
                            else:
                                dictionary[name] = elem
                        else:
                            if '"Description"' in dictionary:
                                description = dictionary['"Description"']
                                dictionary['"Description"'] = description + '\n' + line.rstrip()
            self.nd_dictionary = dictionary
            f=re.split('/|\\\\',path)
            self.origin = '\\'.join(f[:-1]) 
            self.file_name = f[-1].split('.nd')[0]
            self.version = dictionary['"NDInfoFile"']
            self.description = dictionary['"Description"']
            self.start_time = (dictionary['"StartTime1"'])
            self.do_timelapse = dictionary['"DoTimelapse"'] == "TRUE"
            if self.do_timelapse:
                self.n_time_points = int(dictionary['"NTimePoints"'])
            self.do_stage = dictionary['"DoStage"'] == "TRUE"
            if self.do_stage:
                self.n_stage_positions = int(dictionary['"NStagePositions"'])
                self.position_name = []
                for k in range(1, self.n_stage_positions + 1):
                    self.position_name += [dictionary['"Stage' + str(k) + '"'][1:-2]]
            self.do_wave = dictionary['"DoWave"'] == "TRUE"
            self.n_wavelengths = max(int(dictionary['"NWavelengths"']),1)
            if self.do_wave:
                self.wave_name = []
                self.wave_do_z = []
                for k in range(1, self.n_wavelengths + 1):
                    self.wave_name += [dictionary['"WaveName' + str(k) + '"'].split('Position')[-1].strip('"')]
                    self.wave_do_z += [dictionary['"WaveDoZ' + str(k) + '"'] == 'TRUE']
            self.do_wave_points_collection = '"WavePointsCollected"' in dictionary
            if self.do_wave_points_collection:
                self.n_wave_points_collection = len(dictionary['"WavePointsCollected"'])
                self.wave_points_collection = []
                for k in range(self.n_wave_points_collection):
                    pc = dictionary['"WavePointsCollected"'][k].split(',')
                    vpc = WavePointsCollection(len(pc) - 1)
                    vpc.wave=int(pc[0])
                    for j in range(1, len(pc)):
                        vpc.time_points[j - 1]= int(pc[j])
                    self.wave_points_collection+=[vpc]
            self.do_z_series = dictionary['"DoZSeries"'] == "TRUE"
            if self.do_z_series:
                self.n_z_steps = int(dictionary['"NZSteps"'])
                self.z_step_size = float(dictionary['"ZStepSize"'])
            self.wave_in_file_name = dictionary['"WaveInFileName"'] == "TRUE"
            self.n_events = int(dictionary['"NEvents"'])
            if self.n_events > 0:
                self.events = []
                for i in range(self.n_events):
                    ev = Event()
                    arg = dictionary['"Event"' + str(i + 1)].split(',')
                    ev.Event(arg[0][1, len(arg[0]) - 1], arg[1][1, len(arg[1]) - 1], arg[2][1, len(arg[2]) - 1],
                             int(arg[3]), int(arg[4]), int(arg[5]), arg[6] == '"TRUE"', float(arg[7]))
                    self.events.append(ev)
            
            self.folder_path=self.origin+'\\'
            
        except:
            pass
    
    def build_layers(self,pos):
        '''Takes a position and returs the proper stack of images, containing all different wavelenghts
        output is a list or tuples containing layers for napari'''
        self.image_layers=[]

        if self.do_timelapse:
            for wave in range(self.n_wavelengths):
                self.build_file_list(pos,wave)   
                da_stack=read_stack([self.folder_path+filename for filename in self.file_name_list])
                # optional kwargs for the corresponding viewer.add_* method
                add_kwargs = {'name':(self.wave_name[wave] if self.do_wave else 'no_name')}
                layer_type = "image"  # optional, default is "image"
                self.image_layers+=[(da_stack, add_kwargs, layer_type)]
        else:
            for wave in range(self.n_wavelengths):
                filename=''.join([self.file_name,
                    ('_w'+str(wave + 1) if self.do_wave else ''),
                    (self.wave_name[wave] if self.do_wave and self.wave_in_file_name else ''),
                    ('_s' + str(pos) if self.do_stage else ''),
                    ('.tif' if (not self.do_wave or not self.wave_do_z[wave]) and (self.do_wave or not self.do_z_series) else self.extension_for_stack)])
                data=imread(self.folder_path+filename)
                add_kwargs = {'name':(self.wave_name[wave] if self.do_wave else 'no_name')}
                layer_type = "image"  # optional, default is "image"
                self.image_layers+=[(data, add_kwargs, layer_type)]

    def build_file_list(self, pos, wave):
        begin=1
        end=self.n_time_points
        restrict_to_collected = False
        collection = 0
        n_points = 0
        index = 0
        time = 0
    
        if self.do_wave_points_collection:
            for index in range(len(self.wave_points_collection)):
                if self.wave_points_collection[index].wave == wave + 1:
                    restrict_to_collected = True
                    collection = index
    
            n_points = 0
    
            for index in range(self.wave_points_collection[collection].size):
                time = self.wave_points_collection[collection].time_points[index]
                if begin <= time <= end:
                    n_points += 1
    
        if not restrict_to_collected:
            n_points = end - begin + 1
    
        self.file_name_list = [None] * n_points
        self.check_padding()
    
        if not restrict_to_collected:
            for index in range(begin, end + 1):
                self.file_name_list[index - begin] = ''.join([self.file_name,
                                                              ('_w'+str(wave + 1) if self.do_wave else ''),
                                                              (self.wave_name[wave] if self.do_wave and self.wave_in_file_name else ''),
                                                              ('_s' + str(pos + 1) if self.do_stage else ''),
                                                              '_t' + self.pad(index, self.n_digits),
                                                              ('.tif' if (not self.do_wave or not self.wave_do_z[wave]) and (self.do_wave or not self.do_z_series) else self.extension_for_stack)])
        else:
            index = 0
    
            for time in range(self.wave_points_collection[collection].size):
                curr_time = self.wave_points_collection[collection].time_points[time]
                if begin <= curr_time <= end:
                    self.file_name_list[index] = ''.join([self.file_name,
                                                        ('_w'+str(wave + 1) if self.do_wave else ''),
                                                        (self.wave_name[wave] if self.do_wave and self.wave_in_file_name else ''),
                                                        ('_s' + str(pos + 1) if self.do_stage else ''),
                                                        '_t' + self.pad(curr_time, self.n_digits),
                                                        ('.tif' if (not self.do_wave or not self.wave_do_z[wave]) and (self.do_wave or not self.do_z_series) else self.extension_for_stack)])
                    index+=1
    def pad(self, what, n):
        out = str(what)
        while len(out) < n:
            out = '0' + out
        return out


    def check_padding(self):
        file_list = os.listdir(self.origin)
        self.n_digits = len(str(file_list[0]))
    
        for file_name in file_list:
            start = file_name.rfind("_t") + 2
            end = file_name.rfind(".")
            
            if start != 1 and end != -1:
                self.n_digits = min(self.n_digits, len(file_name[start:end]))


class WavePointsCollection:
    def __init__(self, size=0):
        """Creates a new instance of WavePointsCollection to store the indexes of images
        collected during Multi-Dimensional Acquisition when not all time points were collected.
        """
        self.size = size
        self.wave = 0
        self.time_points = [0] * size
        
class Event:
    '''event created by a user during ann acquisition in Metamorph'''
    
    def __init__(self, type, comment, time, nb1, nb2, nb3, bool_val, color):
        self.type = type
        self.comment = comment
        self.time = time
        self.nb1 = nb1
        self.nb2 = nb2
        self.nb3 = nb3
        self.bool_val = bool_val
        self.color = color

    def __str__(self):
        return (
            f"Event(type={self.type}, comment={self.comment}, time={self.time}, "
            f"nb1={self.nb1}, nb2={self.nb2}, nb3={self.nb3}, "
            f"bool_val={self.bool_val}, color={self.color})"
        )
    

def read_stack(filenames):
    '''function lazy imread, taken from napari tutorial. used in NdExp.build_stacks'''
    sample = imread(filenames[0])
    
    lazy_imread = delayed(imread)  # lazy reader
    lazy_arrays = [lazy_imread(fn) for fn in filenames]
    dask_arrays = [
        da.from_delayed(delayed_reader, shape=sample.shape, dtype=sample.dtype)
        for delayed_reader in lazy_arrays
    ]
    # Stack into one large dask.array
    da_stack = da.stack(dask_arrays, axis=0)
    da_stack.shape 
    return da_stack

#test of the file_name_list given by the function
# =============================================================================
# exp=NdExp()
# 
# exp.nd_reader(r"G:/optorhoa/230707_rpe1_optoprg_doublephen/multicell.nd")
# 
# exp.build_file_list(1,0)
# 
# exp.file_name_list
# =============================================================================
