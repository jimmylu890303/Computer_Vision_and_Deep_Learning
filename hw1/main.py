import sys
from PyQt5.QtWidgets import  QApplication, QFileDialog, QMainWindow, QGraphicsPixmapItem, QGraphicsScene
from PyQt5.uic import loadUi
from utility import *
from PyQt5.QtGui import QPixmap, QPainter, QImage
from PyQt5.QtCore import Qt

class UI(QMainWindow):
    def __init__(self):
        super(UI,self).__init__()
        loadUi("./UI.ui",self)
        self.folder_path = ""
        self.imageL_path = ""
        self.imageR_path = ""
        self.image1_path = ""
        self.image2_path = ""
        self.image_q5__path = ""
        self.intrinsic_mat = []
        self.distortion_mat = []
        self.load_folder_btn.clicked.connect(self.openFolder)
        self.load_imageL_btn.clicked.connect(lambda: self.openImage(0))
        self.load_imageR_btn.clicked.connect(lambda: self.openImage(1))
        self.load_image1_btn.clicked.connect(lambda: self.openImage2(0))
        self.load_image2_btn.clicked.connect(lambda: self.openImage2(1))
        self.q5_load_img_btn.clicked.connect(lambda: self.openImage_and_show())
        self.find_corners_btn.clicked.connect(self.find_and_draw_corners)
        self.find_intrinsic_btn.clicked.connect(self.find_Intrinsic_Matrix)
        self.find_extrinsic_btn.clicked.connect(self.find_Extrinsic_Matrix)
        self.find_distortion_btn.clicked.connect(self.Find_Distortion_Matrix)
        self.show_q1result_btn.clicked.connect(self.Show_Undistorted_Result)
        self.show_words_hori_btn.clicked.connect(self.Show_Word_on_chessboard)
        self.show_words_vert_btn.clicked.connect(self.Show_Word_ver_on_chessboard)
        self.Stereo_Disparity_Map_btn.clicked.connect(lambda: Stereo_Disparity_Map(self.imageL_path,self.imageR_path))
        self.find_keypoints_btn.clicked.connect(lambda:find_keypoints(self.image1_path))
        self.match_keypoints_btn.clicked.connect(lambda:match_keypoints(self.image1_path,self.image2_path))
        self.show_augmented_img_btn.clicked.connect(lambda:Show_Augmented_images())
        self.show_model_summary_btn.clicked.connect(lambda:Show_Model_Summary())
        self.show_acc_loss_btn.clicked.connect(lambda:Show_Accuracy_and_Loss())
        self.Inference_btn.clicked.connect(self.Inference)
        self.spinBox.setRange(1,15)
       
    def openFolder(self):
        self.folder_path = QFileDialog.getExistingDirectory()
    def openImage(self,type):
        if type==0:
            self.imageL_path, _ = QFileDialog.getOpenFileName(self,'Open Files','','*.png *.jpg')  
        else:
            self.imageR_path, _ = QFileDialog.getOpenFileName(self,'Open Files','','*.png *.jpg')
    def openImage2(self,type):
        if type==0:
            self.image1_path, _ = QFileDialog.getOpenFileName(self,'Open Files','','*.png *.jpg')  
        else:
            self.image2_path, _ = QFileDialog.getOpenFileName(self,'Open Files','','*.png *.jpg')
    def openImage_and_show(self):  
            self.predict_text.setText("Predict = ")
            self.image_q5__path, _ = QFileDialog.getOpenFileName(self,'Open Files','','*.png *.jpg')
            # Clear scene
            self.input_img.setScene(None)
            scene = QGraphicsScene()
            # load 32*32 small img
            small_pixmap = QPixmap(self.image_q5__path)
            # create large img 256*192
            large_image = QImage(256, 192, QImage.Format_ARGB32)
            large_image.fill(0)
            # scale small img to large
            painter = QPainter(large_image)
            painter.setRenderHint(QPainter.SmoothPixmapTransform) 
            painter.drawImage(32, 0, small_pixmap.toImage().scaled(192, 192, aspectRatioMode=Qt.KeepAspectRatio))
            painter.end()
            # add to widget
            pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(large_image))
            scene.addItem(pixmap_item)
            self.input_img.setScene(scene)
    def find_and_draw_corners(self):
        find_and_draw_corners(self.folder_path)
    def find_Intrinsic_Matrix(self):
        self.intrinsic_mat = find_Intrinsic_Matrix(self.folder_path)
    def find_Extrinsic_Matrix(self):
        find_Extrinsic_Matrix(self.folder_path,self.spinBox.value())
    def Find_Distortion_Matrix(self):
        self.distortion_mat = Find_Distortion_Matrix(self.folder_path)
    def Show_Undistorted_Result(self):
        Show_Undistorted_Result(self.folder_path,self.intrinsic_mat,self.distortion_mat)
    def Show_Word_on_chessboard(self):
        Show_Word_on_chessboard(self.folder_path,self.question2_textEdit.toPlainText())
    def Show_Word_ver_on_chessboard(self):
        Show_Word_ver_on_chessboard(self.folder_path,self.question2_textEdit.toPlainText())
    def Inference(self):
        predict = Inference(self.image_q5__path)
        self.predict_text.setText("Predict = " + predict)


app = QApplication(sys.argv)
MainWindow = UI()
MainWindow.show()
sys.exit(app.exec_())
