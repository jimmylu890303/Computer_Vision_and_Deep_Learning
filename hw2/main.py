import sys
from PyQt5.QtWidgets import  QApplication, QFileDialog, QMainWindow, QGraphicsPixmapItem, QGraphicsScene
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap, QPainter, QImage
from PyQt5.QtCore import Qt
from utility import *

class UI(QMainWindow):
    def __init__(self):
        super(UI,self).__init__()
        loadUi(".\hw2.ui",self)
        self.video_path = ""
        self.image_path = ""

        self.load_img_btn.clicked.connect(self.loadImage)
        self.load_video_btn.clicked.connect(self.loadVideo)
        self.BackgroundSubstraction_btn.clicked.connect(lambda:BackgroundSubstraction(self.video_path))
        self.Preprocess_btn.clicked.connect(lambda:preprocess(self.video_path))
        self.video_tracking_btn.clicked.connect(lambda:VideoTracking(self.video_path))

    def loadVideo(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self,'Open Files','','*.mp4')  
    def loadImage(self):
        self.image_path, _ = QFileDialog.getOpenFileName(self,'Open Files','','*.png *.jpg')
       
app = QApplication(sys.argv)
MainWindow = UI()
MainWindow.show()
sys.exit(app.exec_())
