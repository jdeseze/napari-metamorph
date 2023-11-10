"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import numpy as np
from skimage.io import imread
from dask import delayed
import dask.array as da


def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    if not path.endswith(".nd"):
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path
    
    nd_exp=NdExp()
    
    nd_exp.nd_reader(paths[0])
    
    nd_exp.build_layers(1)
    
    if nd_exp.do_stage:
        pass

    return nd_exp.image_layers

#function lazy imread, taken from napari tutorial. used in NdExp.build_stacks
def read_stack(filenames):
    sample = imread(filenames[0])
    
    lazy_imread = delayed(imread)  # lazy reader
    lazy_arrays = [lazy_imread(fn) for fn in filenames]
    dask_arrays = [
        da.from_delayed(delayed_reader, shape=sample.shape, dtype=sample.dtype)
        for delayed_reader in lazy_arrays
    ]
    # Stack into one large dask.array
    da_stack = da.stack(dask_arrays, axis=0)
    da_stack.shape  # (nfiles, nz, ny, nx)
    return da_stack

#%%CLASS nd_reader

import numpy as np
import os

class NdExp:
    
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
            f=path.split('/')
            origin = '\\'.join(f[:-1])
            self.origin = origin 
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
            
            self.folder_path='\\'.join(path.split('/')[0:-1])+'\\'
            
        except:
            pass
    
    def build_layers(self,pos):
        self.image_layers=[]

        if self.do_timelapse:
            for wave in range(self.n_wavelengths):
                self.build_file_list(1,wave)   
                da_stack=read_stack([self.folder_path+filename for filename in self.file_name_list])#imread(filename)
                # optional kwargs for the corresponding viewer.add_* method
                add_kwargs = {'name':(self.wave_name[wave] if self.do_wave else 'no_name')}
                layer_type = "image"  # optional, default is "image"
                self.image_layers+=[(da_stack, add_kwargs, layer_type)]
        else:
            for wave in range(self.n_wavelengths):
                filename=''.join([self.file_name,
                    ('_w'+str(wave + 1) if self.do_wave else ''),
                    (self.wave_name[wave] if self.do_wave and self.wave_in_file_name else ''),
                    ('_s' + str(pos + 1) if self.do_stage else ''),
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


#for test r'G:\optorhoa\230711_rpe1_optocontrol_intensitymeaurement\multicell.nd'

# =============================================================================
# exp=NdExp()
# 
# exp.nd_reader(r'G:/optorhoa/211019_RPE1_optoRhoA_FRAP/cells.nd')
# exp.nd_reader(r'G:/optorhoa/201208_RPE_optoRhoA_PAKiRFP/cell2s_50msact_1.nd')
# exp.build_file_list(1,3)
# print(exp.file_name_list)
# =============================================================================

#%% Class WavePointsCollection

class WavePointsCollection:
    def __init__(self, size=0):
        """Creates a new instance of WavePointsCollection to store the indexes of images
        collected during Multi-Dimensional Acquisition when not all time points were collected.
        """
        self.size = size
        self.wave = 0
        self.time_points = [0] * size



#%%
# =============================================================================
# 
# 
# if self.nd.do_timelapse:
#     gd.addMessage("Timelapse detected, {} timepoints.".format(self.nd.n_time_points))
#     gd.addNumericField("First timepoint: ", 1.0, 0)
#     gd.addNumericField("Last timepoint: ", float(self.nd.n_time_points), 0)
#     gd.addMessage("")
# 
# if self.nd.do_wave:
#     gd.addChoice("{} wavelengths: ".format(self.nd.n_wavelengths), self.nd.wave_name, self.nd.wave_name[0])
# else:
#     gd.addMessage("1 wavelength")
# 
# if self.nd.do_stage:
#     gd.addChoice("{} positions: ".format(self.nd.n_stage_positions), self.nd.position_name, self.nd.position_name[0])
# else:
#     gd.addMessage("1 position")
# 
# if self.nd.do_timelapse:
#     gd.addCheckbox("All_timepoints", True)
# 
# if self.nd.do_wave:
#     gd.addCheckbox("All_wavelengths", True)
# 
# if self.nd.do_stage:
#     gd.addCheckbox("All_positions", True)
# 
# if self.nd.do_z_series:
#     gd.addMessage("")
#     gd.addMessage("Z series detected, {} slices.".format(self.nd.n_z_steps))
#     gd.addChoice("Do projection, Method: ", self.proj_meth, "Maximum")
#     gd.addNumericField("Top slice: ", 1.0, 0)
#     gd.addNumericField("Bottom slice: ", float(self.nd.n_z_steps), 0)
# 
# def get_filename():
#     
#     if self.do_timelapse:
#         start_time = 1
#         stop_time = float(self.n_time_point)
#     
#     if self.do_wave:
#         waves = self.wave_name
#     
#     if self.do_stage:
#         self.position_selected = gd.getNextChoiceIndex()
#     
#     if self.do_timelapse:
#         #if I add a choice of the times to look at. otherwise True
#         self.timepoint_choice = True
#     
#     if self.do_wave:
#         #if I add a choice of the wavelengths to look at. otherwise True
#         self.wave_choice = True
#     
#     if self.do_stage:
#         #choice of the stage position to be added in the widget
#         self.position_choice = 1
#     
#     if self.do_z_series:
#         #to be checked, for the moment it's wrong
#         self.proj_meth_int = gd.getNextChoiceIndex()
#         self.start_slice = int(gd.getNextNumber())
#         self.end_slice = int(gd.getNextNumber())
#         if self.end_slice < self.start_slice:
#             self.start_slice, self.end_slice = self.end_slice, self.start_slice
#     
#     name = sd.getFileName()
#     
#             self.destination = os.path.join(self.destination, name + "_")
# 
# 
# 
# 
# else:
# gd = GenericDialog("!!! Warning !!!")
# if not (self.file_ext.lower() == "nd" and (self.nd is not None and not self.nd.is_nd and self.nd.version is None)):
#     gd.addMessage("This is not a ND file:\nProceed anyway ?")
# 
# if self.nd is not None and not self.nd.is_nd and self.nd.version is not None:
#     gd.addMessage("{}: ND file format not yet supported:\nProceed anyway ?".format(self.nd.version))
# 
# gd.showDialog()
# 
# if not gd.wasCanceled():
#     BatchProjector(self.origin)
# 
# 
# from ij import IJ
# from ij.io import Opener, FileSaver
# from ij.process import ImageProcessor
# from ij.ImagePlus import ImagePlus
# from ij.process import ImageStack
# import os
# 
# 
# def build_stacks(self, path, rec_pos, pos2rec, rec_wave, wave2rec, rec_time, start_time, stop_time, start_slice, stop_slice, proj, convert):
#     self.destination = path
#     begin_pos = 0
#     end_pos = max(1, self.n_stage_positions)
# 
#     if rec_pos:
#         begin_pos = max(pos2rec, 0)
#         end_pos = min(pos2rec + 1, self.n_stage_positions + 1)
# 
#     begin_wave = 0
#     end_wave = max(1, self.n_wavelengths)
# 
#     if rec_wave:
#         begin_wave = max(wave2rec, 0)
#         end_wave = min(wave2rec + 1, self.n_wavelengths + 1)
# 
#     begin_time = 1
#     end_time = self.n_time_points
# 
#     if rec_time:
#         begin_time = max(start_time, 1)
#         end_time = min(stop_time + 1, self.n_time_points)
# 
#     nb_files = (end_pos - begin_pos) * (end_wave - begin_wave) * (end_time - begin_time)
#     curr_file = 1
# 
#     if self.do_z_series:
#         tmp = os.listdir(self.origin)
# 
#         for wave in range(len(tmp)):
#             if tmp[wave].lower().endswith("stk"):
#                 self.extension_for_stack = ".stk"
#                 break
# 
#     for pos in range(begin_pos, end_pos):
#         for wave in range(begin_wave, end_wave):
#             self.build_file_list(pos, wave, begin_time, end_time)
#             curr_img = None
#             stack = ImageStack()
# 
#             for time in range(len(self.file_name_list)):
#                 IJ.showStatus(f"Processing image {curr_file}/{nb_files}")
#                 file_exists = os.path.exists(os.path.join(self.origin, self.file_name_list[time]))
# 
#                 if file_exists:
#                     curr_img = Opener().openImage(self.origin, self.file_name_list[time])
# 
#                 if file_exists:
#                     if stack.getSize() < 1:
#                         stack = ImageStack(curr_img.getWidth(), curr_img.getHeight())
# 
#                     if curr_img.getNSlices() > 1:
#                         slice = self.do_proj(curr_img, start_slice, stop_slice, proj)
#                     else:
#                         slice = curr_img
# 
#                     if slice is not None and convert:
#                         slice.setProcessor("slice", slice.getProcessor().convertToByte(True))
# 
#                     stack.addSlice(self.file_name_list[time], slice.getProcessor())
#                     curr_img.flush()
#                     slice.flush()
#                 else:
#                     IJ.log(f"File {self.file_name_list[time]} is missing.")
# 
#             if stack.getSize() == 0:
#                 IJ.log(f"Stack {self.file_name}" +
#                        (f"_{self.position_name[pos]}" if self.do_stage else "") +
#                        (f"_{self.wave_name[wave]}" if self.do_wave else "") +
#                        " could not be built.")
#             else:
#                 save_name = (f"{self.file_name}" +
#                              (f"_{self.position_name[pos]}" if self.do_stage else "") +
#                              (f"_{self.wave_name[wave]}" if self.do_wave else "")).replace(".", "_") + ".tif"
# 
#                 result = ImagePlus(save_name, stack)
#                 result.setProperty("Info", self.description)
# 
#                 if stack.getSize() > 1:
#                     FileSaver(result).saveAsTiffStack(os.path.join(self.destination, save_name))
#                 elif stack.getSize() == 1:
#                     FileSaver(result).saveAsTiff(os.path.join(self.destination, save_name))
# 
# 
# 
# =============================================================================
