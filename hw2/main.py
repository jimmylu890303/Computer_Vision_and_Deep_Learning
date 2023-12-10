import sys
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import  QApplication, QFileDialog, QMainWindow, QGraphicsPixmapItem, QGraphicsScene,QGraphicsView
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap, QPainter, QImage
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt,QBuffer
from utility import *
from PIL import Image,ImageQt

class Canvas(QtWidgets.QLabel):

    def __init__(self):
        super().__init__()
        pixmap = QtGui.QPixmap(400, 300)
        pixmap.fill(Qt.black)
        self.setPixmap(pixmap)

        self.last_x, self.last_y = None, None
        self.pen_color = QtGui.QColor('#FFFFFF')

    def set_pen_color(self, c):
        self.pen_color = QtGui.QColor(c)

    def mouseMoveEvent(self, e):
        
        if self.last_x is None: # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return # Ignore the first time.

        painter = QtGui.QPainter(self.pixmap())
        p = painter.pen()
        p.setWidth(12)
        p.setColor(self.pen_color)
        painter.setPen(p)
        painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        painter.end()
        self.update()
        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None
    def get_image(self):
        return self.pixmap().toImage()

class UI(QMainWindow):
    def __init__(self):
        super(UI,self).__init__()
        loadUi(".\hw2.ui",self)
        self.video_path = ""
        self.image_path = ""
        self.image_q5__path = ""
        self.initPaint()

        self.load_img_btn.clicked.connect(self.loadImage)
        self.load_video_btn.clicked.connect(self.loadVideo)
        self.BackgroundSubstraction_btn.clicked.connect(lambda:BackgroundSubstraction(self.video_path))
        self.Preprocess_btn.clicked.connect(lambda:preprocess(self.video_path))
        self.video_tracking_btn.clicked.connect(lambda:VideoTracking(self.video_path))
        self.PCA_btn.clicked.connect(lambda:PCA_SOL(self.image_path))
        self.show_model_btn.clicked.connect(Show_Model_Summary)
        self.show_acc_loss_btn.clicked.connect(self.Show_Accuracy_and_Loss)
        self.predict_mnist_btn.clicked.connect(self.predict)
        self.reset_btn.clicked.connect(self.reset)
        self.load_q5Img_btn.clicked.connect(self.load_show_image)
        self.show_resnet50_btn.clicked.connect(show_ResNet50_Model_Summary)
        self.show_q5_img_btn.clicked.connect(show_class_image)
        self.Show_Comprasion_btn.clicked.connect(show_accurancy_comparison)
        self.resnet50_predict_btn.clicked.connect(self.predictCatDog)

    def load_show_image(self):
        self.image_q5__path, _ = QFileDialog.getOpenFileName(self,'Open Files','','*.png *.jpg')
        self.resnet_predict.setText(f'Predict = ')
        # Clear scene
        self.input_img.setScene(None)
        scene = QGraphicsScene()
        small_pixmap = QPixmap(self.image_q5__path)
        large_image = QImage(224, 224, QImage.Format_ARGB32)
        large_image.fill(0)
        painter = QPainter(large_image)
        painter.setRenderHint(QPainter.SmoothPixmapTransform) 
        painter.drawImage(0, 0, small_pixmap.toImage().scaled(224, 224))
        painter.end()
        # add to widget
        pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(large_image))
        scene.addItem(pixmap_item)
        self.input_img.setScene(scene)
    def loadVideo(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self,'Open Files','','*.mp4')  
    def loadImage(self):
        self.image_path, _ = QFileDialog.getOpenFileName(self,'Open Files','','*.png *.jpg')
    def Show_Accuracy_and_Loss(self):
      
        image1_path = 'model/loss_plot.png'
        image2_path = 'model/accuracy_plot.png'
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)
        Hori = np.concatenate((image1, image2), axis=0)
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.imshow(image1)
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(image2)
        plt.axis('off') 
        plt.show() 
        
        if type(self.paint_view)==QGraphicsView:
            return
        view = QGraphicsView()
        scene = QGraphicsScene()
        height, width, channel = Hori.shape
        bytesPerLine = 3 * width
        qImg = QImage(Hori.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        small_pixmap = QPixmap.fromImage(qImg)
        # create large img 256*192
        large_image = QImage(400, 300, QImage.Format_ARGB32)
        large_image.fill(0)
        # scale small img to large
        painter = QPainter(large_image)
        painter.setRenderHint(QPainter.SmoothPixmapTransform) 
        painter.drawImage(0, 0, small_pixmap.toImage().scaled(600, 300, aspectRatioMode=Qt.KeepAspectRatio))
        painter.end()
        # add to widget
        pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(large_image))
        scene.addItem(pixmap_item)
        view.setScene(scene)
        index = self.sol4_layout.indexOf(self.paint_view)
        self.sol4_layout.insertWidget(index, view)
        self.paint_view.deleteLater()  # Delete the old paint_view
        self.paint_view = view  # Assign canvas to paint_view
    def predict(self):
        img = self.paint_view.get_image()
        pil_image = ImageQt.fromqimage(img)
        predict = predict_mnist(pil_image)
        self.mnist_predict.setText(f' Predict = {predict}')
    def reset(self):
        self.mnist_predict.setText(f' Predict =')
        self.initPaint()
    def initPaint(self):
        self.canvas = Canvas()
        # Add the Canvas instance to the layout replacing paint_view
        index = self.sol4_layout.indexOf(self.paint_view)
        self.sol4_layout.insertWidget(index, self.canvas)
        self.paint_view.deleteLater()  # Delete the old paint_view
        self.paint_view = self.canvas  # Assign canvas to paint_view
    def predictCatDog(self):
        label = predict_resnet50(self.image_q5__path)
        self.resnet_predict.setText(f'Predict = {label}')
    
app = QApplication(sys.argv)
MainWindow = UI()
MainWindow.show()
sys.exit(app.exec_())
