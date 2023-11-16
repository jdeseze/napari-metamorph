"""
This module contains four napari widgets declared in
different ways:

- a pure Python function flagged with `autogenerate: true`
    in the plugin manifest. Type annotations are used by
    magicgui to generate widgets for each parameter. Best
    suited for simple processing tasks - usually taking
    in and/or returning a layer.
- a `magic_factory` decorated function. The `magic_factory`
    decorator allows us to customize aspects of the resulting
    GUI, including the widgets associated with each parameter.
    Best used when you have a very simple processing task,
    but want some control over the autogenerated widgets. If you
    find yourself needing to define lots of nested functions to achieve
    your functionality, maybe look at the `Container` widget!
- a `magicgui.widgets.Container` subclass. This provides lots
    of flexibility and customization options while still supporting
    `magicgui` widgets and convenience methods for creating widgets
    from type annotations. If you want to customize your widgets and
    connect callbacks, this is the best widget option for you.
- a `QWidget` subclass. This provides maximal flexibility but requires
    full specification of widget layouts, callbacks, events, etc.

References:
- Widget specification: https://napari.org/stable/plugins/guides.html?#widgets
- magicgui docs: https://pyapp-kit.github.io/magicgui/

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING
from qtpy.QtWidgets import QComboBox,QFileDialog,QMessageBox,QVBoxLayout,QCheckBox,QPushButton, QWidget
import os
import numpy as np
from skimage.io import imread
from dask import delayed
import dask.array as da
from .ndreader import NdExp
import napari

class NdReaderWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        #set the vertical layout
        self.setLayout(QVBoxLayout())
        
        #add the 'Select Folder' button
        select_folder = QPushButton(self)
        select_folder.setText('Select Folder')
        self.layout().addWidget(select_folder)
        select_folder.clicked.connect(self.choose_exp)
        
        #add the dropdown list to select the experiment
        self.list_ndfiles=QComboBox(self)
        self.layout().addWidget(self.list_ndfiles)
        self.list_ndfiles.currentTextChanged.connect(self.load_ndfile)
        
        #add the dropdown button to select the cell 
        self.cell_nb=QComboBox(self.list_ndfiles)
        self.layout().addWidget(self.cell_nb)
        
        #button to decide whether you keep the layers or not
        self.keep_layers=QCheckBox()
        self.keep_layers.setText('Keep layers')
        self.layout().addWidget(self.keep_layers)
        
        #button to launch the loading of the experiment
        load=QPushButton(self)
        load.setText('Load images')
        self.layout().addWidget(load)
        load.clicked.connect(self.load_images)
        
        #button to launch the loading of the next position of the experiment
        load_next=QPushButton(self)
        load_next.setText('Load next position')
        self.layout().addWidget(load_next)
        load_next.clicked.connect(self.load_next_pos)

    '''function to choose the directory and have the experiments in the folder being displayed'''
    def choose_exp(self):
        dbox = QFileDialog(self)
        #dbox.setDirectory('F:/optorhoa')   
        dbox.setFileMode(QFileDialog.Directory)          
        if dbox.exec_():
            self.folder = dbox.selectedFiles()
        for folder_path in self.folder:
             filenames=[f for f in os.listdir(folder_path) if f.endswith('.nd')]  
        self.list_ndfiles.clear()
        if len(filenames)==0:
            mess=QMessageBox(self)
            mess.setText('No .nd file in this directory')
            self.layout().addWidget(mess) 
        else:
            self.list_ndfiles.addItems(filenames)
        
        self.load_ndfile()
            

    def load_ndfile(self):
        self.cell_nb.clear()
        
        try:
            #create the NdReader object containing the experiments
            self.nd_exp=NdExp()
            self.nd_exp.nd_reader(os.path.join(self.folder[0],self.list_ndfiles.currentText()))
        except:
            mess=QMessageBox(self)
            mess.setText('Unable to load experiment, or no .nd file in the folder')
            self.layout().addWidget(mess) 
        
        if self.nd_exp.do_stage:
            #add positions to the cell_nb list
            self.cell_nb.addItems(list(map(str,range(1,self.nd_exp.n_stage_positions+1))))        
        
    def load_images(self):
        #if he finds a shape layer, he keeps it
        is_shape=False
        for layer in self.viewer.layers:
            if type(layer)==napari.layers.shapes.shapes.Shapes:
                shape=layer
                is_shape=True
                
                
        #clear all layers
        if self.keep_layers.checkState()==0:
            self.viewer.layers.clear()
        
        #check whether there are different position or not
        try:
            pos=int(self.cell_nb.currentText())
        except:
            pos=0
        self.nd_exp.build_layers(pos)

        try:
            for image_layer in self.nd_exp.image_layers:
                #add each wavelength
                self.viewer.add_image(image_layer[0],**image_layer[1]) 
                self.viewer.layers[-1].reset_contrast_limits()
        except:
            mess=QMessageBox(self)
            mess.setText('Cannot read images. They could be no complete timepoints, or the images corresponding to the .nd file are not in the folder')
            self.layout().addWidget(mess) 
        
        #add the shape layer on top 
        if is_shape:
            self.viewer.add_layer(shape)

        self.viewer.reset_view()
        self.viewer.grid.enabled=True
        
    def load_next_pos(self):
        
        try:
            self.cell_nb.setCurrentIndex(self.cell_nb.currentIndex()+1)
            self.load_images()
        except:
            mess=QMessageBox(self)
            mess.setText('It is the last position')
            self.layout().addWidget(mess) 
        
        self.viewer.reset_view()
        self.viewer.grid.enabled=True

#%%
# =============================================================================
# # Uses the `autogenerate: true` flag in the plugin manifest
# # to indicate it should be wrapped as a magicgui to autogenerate
# # a widget.
# def threshold_autogenerate_widget(
#     img: "napari.types.ImageData",
#     threshold: "float", 
# ) -> "napari.types.LabelsData":
#     return img_as_float(img) > threshold
# 
# 
# # the magic_factory decorator lets us customize aspects of our widget
# # we specify a widget type for the threshold parameter
# # and use auto_call=True so the function is called whenever
# # the value of a parameter changes
# @magic_factory(
#     threshold={"widget_type": "FloatSlider", "max": 1}, auto_call=True
# )
# def threshold_magic_widget(
#     img_layer: "napari.layers.Image", threshold: "float"
# ) -> "napari.types.LabelsData":
#     return img_as_float(img_layer.data) > threshold
# 
# 
# 
# # if we want even more control over our widget, we can use
# # magicgui `Container`
# class ImageThreshold(Container):
#     def __init__(self, viewer: "napari.viewer.Viewer"):
#         super().__init__()
#         self._viewer = viewer
#         # use create_widget to generate widgets from type annotations
#         self._image_layer_combo = create_widget(
#             label="Image", annotation="napari.layers.Image"
#         )
#         self._threshold_slider = create_widget(
#             label="Threshold", annotation=float, widget_type="FloatSlider"
#         )
#         self._threshold_slider.min = 0
#         self._threshold_slider.max = 1
#         # use magicgui widgets directly
#         self._invert_checkbox = CheckBox(text="Keep pixels below threshold")
# 
#         # connect your own callbacks
#         self._threshold_slider.changed.connect(self._threshold_im)
#         self._invert_checkbox.changed.connect(self._threshold_im)
# 
#         # append into/extend the container with your widgets
#         self.extend(
#             [
#                 self._image_layer_combo,
#                 self._threshold_slider,
#                 self._invert_checkbox,
#             ]
#         )
# 
#     def _threshold_im(self):
#         image_layer = self._image_layer_combo.value
#         if image_layer is None:
#             return
# 
#         image = img_as_float(image_layer.data)
#         name = image_layer.name + "_thresholded"
#         threshold = self._threshold_slider.value
#         if self._invert_checkbox.value:
#             thresholded = image < threshold
#         else:
#             thresholded = image > threshold
#         if name in self._viewer.layers:
#             self._viewer.layers[name].data = thresholded
#         else:
#             self._viewer.add_labels(thresholded, name=name)
# =============================================================================
