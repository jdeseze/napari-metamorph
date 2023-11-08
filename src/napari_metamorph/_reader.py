"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import numpy as np
from skimage.io import imread


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
    
    nd_exp=Nd_exp()
    nd_exp.nd_reader(paths[0])
    
    # load all files into array
    #arrays = [np.load(_path) for _path in paths]
    # stack arrays into single array
    #data = np.squeeze(np.stack(arrays))
    path=paths[0]
    filename='\\'.join(path.split('/')[0:-1]+'multicell_w1TIRF 561_s1.tif'
    data=imread(filename)
    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {}

    layer_type = "image"  # optional, default is "image"
    return [(data, add_kwargs, layer_type)]

#%%CLASS nd_reader

class Nd_exp :
    
    def __init(self):
        self.isND = False
        self.badNDType = False
        self.DoTimelapse = False
        self.DoStage = False
        self.DoWave = False
        self.DoWavePointsCollection = False
        self.DoZSeries = False
        self.WaveInFileName = False
        self.NTimePoints = 1
        self.NStagePositions = 1
        self.NWavelengths = 1
        self.NWavePointsCollection = 0
        self.NZSteps = 0
        self.NEvents = 0
        self.ZStepSize = np.double(0.0)
        self.origin = ''
        self.destination = ''
        self.Version = ''
        self.Description = ''
        self.StartTime = ''
        self.ImageName = ''
        self.ExtensionForStack = ".tif"
        self.PositionName = []
        self.WaveName = []
        self.FileNameList = []
        self.WavePointsCollection = []
        self.Event = []
        self.WaveDoZ = []
        self.FileName = ""
        self.nDigits = 1
        self.nd_dictionnary = dict()

    def nd_reader(self,path):
        
        dictionnary = dict()
        try :
            with open(path,'r') as file:
                for line in file :
                    if len(line)>1:
                        prename=line.rstrip().split(', ')
                        if prename[0][-1]==',': #if it’s empty
                            prename=line.rstrip().split(',')
                        if prename[0][0]=='"' and prename[0][-1] =='"':
                            name = prename[0]
                            if len(prename)>2:
                                elem = ','.join(prename[1:])
                            else :
                                elem = prename[-1]
                            if name =='"WavePointsCollected"':
                                if name in dictionnary :
                                    wp=dictionnary[name]
                                    dictionnary[name] = wp+[elem]
                                else:
                                    dictionnary[name] = [elem]
                            else :
                                dictionnary[name]=elem
                        else:
                            if '"Description"' in dictionnary :
                                description = dictionnary['"Description"']
                                dictionnary['"Description"']=description +'\n'+line.rstrip()
            self.nd_dictionnary = dictionnary
            f = path.split('/')
            origin = '/'+f[0]+'/'
            for k in range(1,len(f)-1):
                origin = origin+f[k]+'/'
            self.origin = origin 
            self.FileName = f[-1].split('.nd')[0]
            self.Version = dictionnary['"NDInfoFile"']
            self.Description = dictionnary['"Description"'] #code made for version 2.0
            self.StartTime = (dictionnary['"StartTime1"'])
            self.DoTimelapse = dictionnary['"DoTimelapse"']=="TRUE"
            if self.DoTimelapse :
                self.NTimePoints= int(dictionnary['"NTimePoints"'])
            self.DoStage = dictionnary['"DoStage"']=="TRUE"
            if self.DoStage :
                self.NStagePositions= int(dictionnary['"NStagePositions"'])
                self.PositionName = []
                for k in range(1,self.NStagePositions+1):
                    self.PositionName+=[dictionnary['"Stage'+str(k)+'"'][1:-2]]
            self.DoWave = dictionnary['"DoWave"']=="TRUE"
            self.NWavelengths = int(dictionnary ['"NWavelengths"'])
            if self.DoWave :
                self.WaveName=[]
                self.WaveDoZ = []
                for k in range(1,self.NWavelengths+1):
                    self.WaveName+=[dictionnary['"WaveName'+str(k)+'"'].split('Position')[-1]]
                    self.WaveDoZ+=[dictionnary['"WaveDoZ'+str(k)+'"']=='TRUE']
            self.DoWavePointsCollection = '"WavePointsCollected"'in dictionnary
            if self.DoWavePointsCollection:
                self.NWavePointsCollection = len(dictionnary['"WavePointsCollected"'])
                self.WWavePointsCollection = []
                for k in range(self.NWavePointsCollection):
                    pc = dictionnary['"WavePointsCollected"'][k].split(',')
                    Vpc = WavePointsCollection()
                    Vpc.WavePointsCollection(len(pc)-1)
                    Vpc.setWave(int(pc[0]))
                    for j in range(1,len(pc)):
                        Vpc.setTimePoint(j-1, int(pc[j]))
                    self.WWavePointsCollection.append(Vpc)
            self.DoZSeries = dictionnary['"DoZSeries"']=="TRUE"
            if self.DoZSeries:
                self.NZSteps = int(dictionnary ['"NZSteps"'])
                self.ZStepSize = float(dictionnary ['"ZStepSize"'])
            self.WaveInFileName = dictionnary['"WaveInFileName"']=="TRUE"
            self.NEvents = int(dictionnary['"NEvents"'])
            if self.NEvents > 0: #This part may be not functionnal since I did not test it
                self.Events = []
                for i in range(self.NEvents):
                    ev = Event()
                    arg = dictionnary['"Event"' + str(i + 1)].split(',')
                    ev.Event(arg[0][1,len(arg[0])-1], arg[1][1, len(arg[1])-1], arg[2][1, len(arg[2])-1], int(arg[3]), int(arg[4]), int(arg[5]), arg[6]=='"TRUE"', float(arg[7]))
                    self.Events.append(ev)
        except:
            pass
