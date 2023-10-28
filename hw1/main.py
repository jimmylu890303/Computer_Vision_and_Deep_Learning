import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QMainWindow
from PyQt5.uic import loadUi
from utility import *


class UI(QMainWindow):
    def __init__(self):
        super(UI,self).__init__()
        loadUi("./hw1/UI.ui",self)
        self.folder_path = ""
        self.imageL_path = ""
        self.imageR_path = ""
        self.intrinsic_mat = []
        self.distortion_mat = []
        self.load_folder_btn.clicked.connect(self.openFolder)
        self.load_imageL_btn.clicked.connect(lambda: self.openImage(0))
        self.load_imageR_btn.clicked.connect(lambda: self.openImage(1))
        self.load_image1_btn.clicked.connect(lambda: self.openImage(0))
        self.load_image2_btn.clicked.connect(lambda: self.openImage(1))
        self.find_corners_btn.clicked.connect(self.find_and_draw_corners)
        self.find_intrinsic_btn.clicked.connect(self.find_Intrinsic_Matrix)
        self.find_extrinsic_btn.clicked.connect(self.find_Extrinsic_Matrix)
        self.find_distortion_btn.clicked.connect(self.Find_Distortion_Matrix)
        self.show_q1result_btn.clicked.connect(self.Show_Undistorted_Result)
        self.show_words_hori_btn.clicked.connect(self.Show_Word_on_chessboard)
        self.show_words_vert_btn.clicked.connect(self.Show_Word_ver_on_chessboard)
        self.Stereo_Disparity_Map_btn.clicked.connect(lambda: Stereo_Disparity_Map(self.imageL_path,self.imageR_path))
        self.find_keypoints_btn.clicked.connect(lambda:find_keypoints(self.imageL_path))
        self.match_keypoints_btn.clicked.connect(lambda:match_keypoints(self.imageL_path,self.imageR_path))
        self.spinBox.setRange(1,15)

       
    def openFolder(self):
        self.folder_path = QFileDialog.getExistingDirectory()
    def openImage(self,type):
        if type==0:
            self.imageL_path, _ = QFileDialog.getOpenFileName(self,'Open Files','','*.png *.jpg')  
        else:
            self.imageR_path, _ = QFileDialog.getOpenFileName(self,'Open Files','','*.png *.jpg')  
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


app = QApplication(sys.argv)
MainWindow = UI()
MainWindow.show()
sys.exit(app.exec_())
