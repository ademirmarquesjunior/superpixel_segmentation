# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 15:47:11 2021

@author: adeju
"""

# from PyQt5 import sip
from PyQt5 import uic, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
import sys
import os
import cv2
import superpixels as sp
import numpy as np

from datetime import datetime

# from skimage.segmentation import slic
# from PIL import Image

# from skimage.segmentation import mark_boundaries

# os.chdir(sys._MEIPASS)

time_now = str(datetime.now())
log = open('log'+time_now.replace(':', '_')+'.txt', 'w')
_ = log.write('Started at ' + str(datetime.now()) + '\n')

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('untitled.ui', self)
        
        self.mask = []
        
        
        self.mask_state = []
        self.state = 1
        self.history = ''
        
        self.original = []
        # self.mask_class = []

        self.superpixel = []
        self.superpixel_A = []
        self.superpixel_V = []

        self.active_mode = 'paint'
        self.active_class = 1
        self.active_color = [0, 0, 255]
        
        self.tolerance = 20
        self.opacity = 0.8

        self.first_click = []

        self.init_Ui()
        self.show()

    def init_Ui(self):
        # Menu
        self.actionOpen_Image.triggered.connect(self.open_image)
        self.actionOpen_Superpixel_file.triggered.connect(self.open_superpixel)
        self.actionSave_mask.triggered.connect(self.save_mask)
        self.actionSLIC.triggered.connect(lambda: self.generate_superpixels('SLIC'))

        # Navigation buttons
        self.undoButton.clicked.connect(self.undo_state)
        self.redoButton.clicked.connect(self.redo_state)
        self.zoomIn.clicked.connect(lambda: self.view_zoom(0.2))
        self.zoomOut.clicked.connect(lambda: self.view_zoom(-0.2))

        # Function buttons
        self.paintSuperpixel.clicked.connect(lambda: self.set_mode('paint'))
        self.eraseSuperpixel.clicked.connect(lambda: self.set_mode('erase'))
        self.leastPath.clicked.connect(lambda: self.set_mode('path'))
        self.regionGrowing.clicked.connect(lambda: self.set_mode('region'))
        self.forestSegmentation.clicked.connect(lambda: self.set_mode('forest'))
        self.dynamicSegmentation.clicked.connect(lambda: self.set_mode('dynamic'))
        
        #Sliders
        self.sliderTolerance.valueChanged.connect(self.set_tolerance)
        self.sliderOpacity.valueChanged.connect(self.set_opacity)

        # Scene definitions
        self.pixmap = self.PixmapItem(self)
        self.scene = QtWidgets.QGraphicsScene(self)
        self.scene.addItem(self.pixmap)
        self.graphicsView.setScene(self.scene)
        
        
    def generate_superpixels(self, method):
        value, done = QtWidgets.QInputDialog.getInt(
           self, 'Generate SLIC superpixels', 'Enter quantity of superpixles:')

        if done:
            try:
                if method=='SLIC':
                    self.superpixel = slic(self.original, value)
                else:
                    self.superpixel = slic(self.original, value)
                self.superpixel_A = sp.superpixels_to_graph(self.superpixel)
                labImage = cv2.cvtColor(self.original, cv2.COLOR_BGR2Lab)
                self.superpixel_V = sp.computeSuperpixelColor(labImage,
                                                              self.superpixel)
                self.mask_class = np.full((self.superpixel.max()+1), 0)
                # visited = np.full((self.superpixel.max()+1), False)
                self.statusBar().showMessage('Superpixel processing done.')
                log.write('Superpixels processes at ' + str(datetime.now()) + '\n')
                print('Superpixel data processing done.')
            except Exception as e:
                print(e)

        
        
        

    def set_mode(self, mode):
        self.active_mode = mode

    def view_zoom(self, scale):
        self.graphicsView.scale(1+scale, 1+scale)


    def add_state(self):  
        if self.state != np.shape(self.mask_state)[0] - 1:
            
            self.mask_state = np.copy(self.mask_state[0: self.state+1])
            self.state += 1

        if np.size(np.shape(self.mask)) == 3:
            self.mask_state = np.concatenate((self.mask_state,
                                          np.reshape(self.mask, (1, np.shape(self.mask)[0], np.shape(self.mask)[1], np.shape(self.mask)[2]))),
                                         axis=0)
        if np.size(np.shape(self.mask)) == 2:
            self.mask_state = np.concatenate((self.mask_state,
                                          np.reshape(self.mask, (1, np.shape(self.mask)[0], np.shape(self.mask)[1]))),
                                         axis=0)
        self.state = np.shape(self.mask_state)[0] - 1

        return
    
    
    def undo_state(self): 
        self.state += -1
        
        if self.state >= 0:
            self.mask = np.copy(self.mask_state[self.state])
            self.update_canvas()
        else:
            self.state = 0
        return
    
    
    def redo_state(self):
        self.state += 1
        
        try:
            self.mask = np.copy(self.mask_state[self.state])
            self.update_canvas()
        except:
            self.state -= 1
        return

    def array_to_QPixmap(self, image):
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImg = QImage(image.data, width, height, bytesPerLine,
                      QImage.Format_RGB888).rgbSwapped()
        return qImg

    def open_image(self):
        path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open image', '',
                                                     'All Files (*.*)')
        
        if path != ('', ''):
            print("File path : " + path[0])
            
            file = open(path[0], 'rb').read()
            image = cv2.imdecode(np.frombuffer(file, np.uint8), 1)
            _ = log.write('Opened '+ path[0] + ' at ' + str(datetime.now()) + '\n')
            
            # image = cv2.imread(path[0])
            self.original = image
            self.mask = np.zeros((np.shape(image)[0], np.shape(image)[1],3), dtype=np.uint8)

            self.mask_state = np.full((1,  np.shape(image)[0], np.shape(image)[1], 3),  np.nan, dtype=np.uint8)
            self.mask_state[0] = self.mask
            
            self.state = 0
            
            

            # self.mask_class = np.full((np.shape(self.mask)[0],
            #                             np.shape(self.mask)[1]), -1)
            
            
            self.update_canvas()
            # self.load_Image_item(path[0])

    def open_superpixel(self):
        path = QtWidgets.QFileDialog.getOpenFileName(self,
                                                     'Open Superpixel file', '',
                                                     'All Files (*.*)')
        if path != ('', ''):
            try:
                self.superpixel = cv2.imread(path[0], -1)
                _ = log.write('Opened '+ path[0] + ' at ' + str(datetime.now()) + '\n')
                self.superpixel_A = sp.superpixels_to_graph(self.superpixel)
                labImage = cv2.cvtColor(self.original, cv2.COLOR_BGR2Lab)
                self.superpixel_V = sp.computeSuperpixelColor(labImage,
                                                              self.superpixel)
                self.mask_class = np.full((self.superpixel.max()+1), 0)
                # visited = np.full((self.superpixel.max()+1), False)
                self.statusBar().showMessage('Superpixel processing done.')
                print('Superpixel data processing done.')
                _ = log.write('Superpixels processes at ' + str(datetime.now()) + '\n')
            except Exception as e:
                print(e)
                
    def save_mask(self):
        path = QtWidgets.QFileDialog.getSaveFileName(self, 'Save mask', '', '*.png')
        
        if path != ('', ''):
            try:
                cv2.imwrite(path[0], self.mask)
                _ = log.write('Saved '+ path[0] + ' at ' + str(datetime.now()) + '\n')
            except Exception as e:
                print(path)
                print(e)

    def paint_superpixel(self, pos):
        x, y = pos
        
        try:
            self.mask[np.where((self.superpixel ==
                                 self.superpixel[x, y]))] = self.active_color
            
        except Exception as e:
            print(e)
            
        self.update_canvas()
        
    def erase_superpixel(self, pos):
        x, y = pos
        
        try:
            self.mask[np.where((self.superpixel ==
                                 self.superpixel[x, y]))] = [0,0,0]
            
        except Exception as e:
            print(e)
            
        self.update_canvas()
        
    def set_tolerance(self):
        self.tolerance = self.sliderTolerance.value()
        self.toleranceValueText.setText(str(self.tolerance))
        
    def set_opacity(self):
        self.opacity = self.sliderOpacity.value()/100
        self.opacityValueText.setText(str(self.sliderOpacity.value()))
        self.update_canvas()
        
    def update_canvas(self):
        temp = cv2.addWeighted(self.original, 1, self.mask, self.opacity, 0.0)
        q_img = self.array_to_QPixmap(temp)
        pixmap_image = QPixmap.fromImage(q_img)
        self.pixmap.setPixmap(pixmap_image)

    def regiongrowing_paint(self, pos, threshold, method="forest"):
        x, y = pos
        
        if method == "forest":
            backtrack = sp.superpixel_forest_segmentation(self.superpixel_A,
                                                          self.superpixel[x, y],
                                                          self.superpixel_V,
                                                          threshold)
        elif method == "region":
            backtrack = sp.superpixel_region_growing(self.superpixel_A,
                                                     self.superpixel[x, y],
                                                     self.superpixel_V,
                                                     threshold, np.full((self.superpixel.max()+1), False))
        try:
            for i in backtrack:
                self.mask[np.where((self.superpixel == i))] = self.active_color
        except Exception:
            print("Error")
        self.update_canvas()

    def dynamic_forest(self, pos, first_pos, path_cost, threshold, event):
        x, y = pos
        x0, y0 = first_pos

        if pos != first_pos:
            backtrack, path_cost = sp.forest_segmentation2(self.superpixel_A,
                                                           self.superpixel[x0, y0],
                                                           self.superpixel[x, y],
                                                           self.superpixel_V,
                                                           path_cost)
            # back = self.history_class.pop()
            # self.mask = np.copy(self.original)
            for target in backtrack:
                self.mask[np.where((self.superpixel == target))] = self.active_color

            self.update_canvas()

        return path_cost
    
    
    def least_path(self, pos, first_pos):
        x, y = pos
        x0, y0 = first_pos

        if pos != first_pos:
            backtrack = sp.least_path(self.superpixel_A, 
                                      self.superpixel[x0, y0],
                                      self.superpixel[x, y],
                                      self.superpixel_V)
            
            self.mask = np.copy(self.mask_state[self.state])
            for target in backtrack:
                self.mask[np.where((self.superpixel == target))] = self.active_color

            self.update_canvas()
            
        return
    



    class PixmapItem(QtWidgets.QGraphicsPixmapItem):
        def __init__(self, outer):
            self.outer = outer
            self.path_cost = []
            super().__init__()

        def mousePressEvent(self, event):
            pos = [int(event.pos().y()), int(event.pos().x())]
            if self.outer.active_mode == 'paint':
                self.outer.paint_superpixel(pos)
                
            if self.outer.active_mode == 'erase':
                self.outer.erase_superpixel(pos)
                
            if self.outer.active_mode == 'path':
                self.outer.first_click = pos
                self.path_cost = self.outer.least_path(pos, self.outer.first_click)
                
            if self.outer.active_mode == 'region':
                self.outer.regiongrowing_paint(pos, self.outer.tolerance, 'region')
                
            if self.outer.active_mode == 'forest':
                self.outer.regiongrowing_paint(pos, self.outer.tolerance, 'forest')
                
            if self.outer.active_mode == 'dynamic':
                self.outer.first_click = pos
                self.path_cost = self.outer.dynamic_forest(pos,
                                                           self.outer.first_click,
                                                           None, None, 0)
            # super(self.PixmapItem, self).mousePressEvent(event)

        def mouseMoveEvent(self, event):
            pos = [int(event.pos().y()), int(event.pos().x())]
            if self.outer.active_mode == 'paint':
                self.outer.paint_superpixel(pos)
                
            if self.outer.active_mode == 'erase':
                self.outer.erase_superpixel(pos)
                
            if self.outer.active_mode == 'path':
                self.path_cost = self.outer.least_path(pos, self.outer.first_click)
            
            if self.outer.active_mode == 'region':
                self.outer.regiongrowing_paint(pos, self.outer.tolerance, 'region')
            
            if self.outer.active_mode == 'forest':
                self.outer.regiongrowing_paint(pos, self.outer.tolerance, 'forest')
            
            if self.outer.active_mode == 'dynamic':
                self.path_cost = self.outer.dynamic_forest(pos,
                                                           self.outer.first_click,
                                                           self.path_cost,
                                                           None, 1)
            
            # super(self.PixmapItem, self).mouseMoveEvent(event)

        def mouseReleaseEvent(self, event):
            pos = [int(event.pos().y()), int(event.pos().x())]
            
            if self.outer.active_mode == 'path':
                self.path_cost = self.outer.least_path(pos, self.outer.first_click)
            
            if self.outer.active_mode == 'dynamic':
                self.path_cost = self.outer.dynamic_forest(pos,
                                                           self.outer.first_click,
                                                           self.path_cost,
                                                           None, 2)
                
            self.outer.add_state()

            # super(self.PixmapItem, self).mouseReleaseEvent(event)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Ui()
    sys.exit(app.exec_())

log.close()
