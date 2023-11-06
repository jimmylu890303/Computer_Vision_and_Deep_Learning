import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image 
import torchvision.transforms as transforms 
from torchsummary import summary
from torchvision import models
import torch
import torch.nn.functional as F
def get_parameters(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path,filename))
        if img is not None:
            images.append(img)
    # 3-dimension coordinate in real world
    objectPoints = []
    objp = np.zeros((8 * 11, 3), np.float32)
    k=0
    for i in range(8):
        for j in range(11):
            objp[k] = [j,i,0]
            k=k+1
    # 2-dimension coordinate in image plane
    imagePoints = []
    # find all corner points in images
    for img in images:
        winSize = (5, 5)               
        zeroZone = (-1, -1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret,corners = cv2.findChessboardCorners(gray_img, (11, 8))
        corners = cv2.cornerSubPix(gray_img, corners, winSize, zeroZone, criteria)
        imagePoints.append(corners)
        objectPoints.append(objp)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints,(2048, 2048), None, None)

    return mtx, dist, rvecs, tvecs
# 1.1 sol.
def find_and_draw_corners(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path,filename))
        if img is not None:
            images.append(img)
    
    for img in images:
        winSize = (5, 5)               
        zeroZone = (-1, -1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret,corners = cv2.findChessboardCorners(gray_img, (11, 8))
        corners = cv2.cornerSubPix(gray_img, corners, winSize, zeroZone, criteria)

        cv2.drawChessboardCorners(img, (11,8),corners,ret)
        
    for img in images:
        cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
        cv2.resizeWindow("output", 500, 500) 
        cv2.imshow('output',img)
        cv2.waitKey(1000)        
# 1.2 sol
def find_Intrinsic_Matrix(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path,filename))
        if img is not None:
            images.append(img)
    # 3-dimension coordinate in real world
    objectPoints = []
    objp = np.zeros((8 * 11, 3), np.float32)
    k=0
    for i in range(8):
        for j in range(11):
            objp[k] = [j,i,0]
            k=k+1
    # 2-dimension coordinate in image plane
    imagePoints = []
    # find all corner points in images
    for img in images:
        winSize = (5, 5)               
        zeroZone = (-1, -1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret,corners = cv2.findChessboardCorners(gray_img, (11, 8))
        corners = cv2.cornerSubPix(gray_img, corners, winSize, zeroZone, criteria)
        imagePoints.append(corners)
        objectPoints.append(objp)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints,(2048, 2048), None, None)
    print("Intrinsic:")
    print(mtx)
    return mtx
# 1.3 sol.
def find_Extrinsic_Matrix(folder_path,image):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path,filename))
        if img is not None:
            images.append(img)
    # 3-dimension coordinate in real world
    objectPoints = []
    objp = np.zeros((8 * 11, 3), np.float32)
    k=0
    for i in range(8):
        for j in range(11):
            objp[k] = [j,i,0]
            k=k+1
    # 2-dimension coordinate in image plane
    imagePoints = []
    # find all corner points in images
    for img in images:
        winSize = (5, 5)               
        zeroZone = (-1, -1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret,corners = cv2.findChessboardCorners(gray_img, (11, 8))
        corners = cv2.cornerSubPix(gray_img, corners, winSize, zeroZone, criteria)
        imagePoints.append(corners)
        objectPoints.append(objp)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints,(2048, 2048), None, None)
    R = cv2.Rodrigues(rvecs[image-1])
    ext = np.hstack((R[0], tvecs[image-1]))
    print("Extrinsic:")
    print(ext)
# 1.4 sol.
def Find_Distortion_Matrix(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path,filename))
        if img is not None:
            images.append(img)
    # 3-dimension coordinate in real world
    objectPoints = []
    objp = np.zeros((8 * 11, 3), np.float32)
    k=0
    for i in range(8):
        for j in range(11):
            objp[k] = [j,i,0]
            k=k+1
    # 2-dimension coordinate in image plane
    imagePoints = []
    # find all corner points in images
    for img in images:
        winSize = (5, 5)               
        zeroZone = (-1, -1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret,corners = cv2.findChessboardCorners(gray_img, (11, 8))
        corners = cv2.cornerSubPix(gray_img, corners, winSize, zeroZone, criteria)
        imagePoints.append(corners)
        objectPoints.append(objp)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints, imagePoints,(2048, 2048), None, None)
    print("Distortion:")
    print(dist)
    return dist
# 1.5 sol
def Show_Undistorted_Result(folder_path):
    mtx, dist, rvecs, tvecs = get_parameters(folder_path)
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path,filename))
        if img is not None:
            images.append(img)
    undistort_images = []
    for img in images:
        undistort_image = cv2.undistort(img,mtx,dist) 
        undistort_images.append(undistort_image)
    
            
    for i in range(len(images)):
        cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
        cv2.resizeWindow("output", 1000, 500) 
        Hori = np.concatenate((images[i], undistort_images[i]), axis=1) 
        
        cv2.imshow('output',Hori)
        cv2.waitKey(1000)        
# 2.1 sol.
def Show_Word_on_chessboard(folder_path,input_str):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path,filename))
        if img is not None:
            images.append(img)

    # 1. Calibrate 5 images to get intrinsic, distortion and extrinsic parameters
    Intrinsic_Mat,Distortion_Mat, rvecs, tvecs = get_parameters(folder_path)
    
    # 2. Show the words on every image
    input_str = input_str.upper()
    file_path = folder_path+'/Q2_lib/alphabet_lib_onboard.txt'
    fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
    anchor = [[7,5,0],[4,5,0],[1,5,0],[7,2,0],[4,2,0],[1,2,0]]
    for i in range(len(images)):
        # 2.1 draw every char on images
        for pos,char in enumerate(input_str,0):
            ch = fs.getNode(char).mat() 
            lines,p1,p2 = ch.shape
            # get line points of 3D real world 
            for j in range(lines):
                ch[j,0] = anchor[pos] + ch[j,0]
                ch[j,1] = anchor[pos] + ch[j,1]
            ch=ch.reshape(-1,3).astype(np.float32)
            # get line poins of 2D image plane
            imgpts, jac = cv2.projectPoints(ch, rvecs[i], tvecs[i], Intrinsic_Mat, Distortion_Mat)
            # draw lines of each char on image
            for point_idx in range(lines*2):
                if(point_idx % 2==0):
                    cv2.line(images[i],tuple(imgpts[point_idx].ravel().astype(np.int32)),tuple(imgpts[point_idx+1].ravel().astype(np.int32)),color=(0, 0, 255),thickness=10)
        # 2.2 print result of each image 
        cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
        cv2.resizeWindow("output", 500, 500) 
        cv2.imshow('output',images[i])
        cv2.waitKey(1000)
    fs.release()
# 2.2 sol.
def Show_Word_ver_on_chessboard(folder_path,input_str):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path,filename))
        if img is not None:
            images.append(img)

    # 1. Calibrate 5 images to get intrinsic, distortion and extrinsic parameters
    Intrinsic_Mat,Distortion_Mat, rvecs, tvecs = get_parameters(folder_path)
    
    # 2. Show the words on every image
    input_str = input_str.upper()
    file_path = folder_path+'/Q2_lib/alphabet_lib_vertical.txt'
    fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
    anchor = [[7,5,0],[4,5,0],[1,5,0],[7,2,0],[4,2,0],[1,2,0]]
    for i in range(len(images)):
        # 2.1 draw every char on images
        for pos,char in enumerate(input_str,0):
            ch = fs.getNode(char).mat() 
            lines,p1,p2 = ch.shape
            # get line points of 3D real world 
            for j in range(lines):
                ch[j,0] = anchor[pos] + ch[j,0]
                ch[j,1] = anchor[pos] + ch[j,1]
            ch=ch.reshape(-1,3).astype(np.float32)
            # get line poins of 2D image plane
            imgpts, jac = cv2.projectPoints(ch, rvecs[i], tvecs[i], Intrinsic_Mat, Distortion_Mat)
            # draw lines of each char on image
            for point_idx in range(lines*2):
                if(point_idx % 2==0):
                    cv2.line(images[i],tuple(imgpts[point_idx].ravel().astype(np.int32)),tuple(imgpts[point_idx+1].ravel().astype(np.int32)),color=(0, 0, 255),thickness=10)
        # 2.2 print result of each image 
        cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
        cv2.resizeWindow("output", 500, 500) 
        cv2.imshow('output',images[i])
        cv2.waitKey(1000)
    fs.release()
# 3.1 sol.
def onClick(event,x,y,flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        disparity,img_right = param
        if(disparity[y][x]==0):
            print("Failure case")
        else:
            x1 = x - disparity[y][x]
            y1 = y
            dist = disparity[y][x]
            print(f'({x1}, {y1}),dis:{dist: .2f}')
            cv2.circle(img_right, (int(x1), y1), 20, (0, 255, 0), -1)
            cv2.imshow('ImageR',img_right)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
def Stereo_Disparity_Map(img1_path,img2_path):
    img_left = cv2.imread(img1_path)
    img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right = cv2.imread(img2_path)
    img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
    disparity = stereo.compute(img_left_gray, img_right_gray)
    # Convert to gray_img
    disparity_visual = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    # show ImageL
    cv2.namedWindow("ImageL", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("ImageL", 480, 270) 
    cv2.setMouseCallback("ImageL", onClick,(disparity,img_right))
    cv2.imshow('ImageL',img_left)
    # show ImageR
    cv2.namedWindow("ImageR", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("ImageR", 480, 270) 
    cv2.imshow('ImageR',img_right)
    # show disparity
    cv2.namedWindow("disparity", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("disparity", 480, 270) 
    cv2.imshow('disparity',disparity_visual)
    cv2.waitKey(0)       
    cv2.destroyAllWindows()
# 4.1 sol.
def find_keypoints(img_path):
    image = cv2.imread(img_path)
    sift = cv2.SIFT_create() 
    keypoints, discriptors = sift.detectAndCompute(image,None)
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    output = cv2.drawKeypoints(gray_image,keypoints,None,color=(0,255,0))

    cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("output", 500, 500) 
    cv2.imshow('output',output)
    cv2.waitKey(0)
# 4.2 sol.
def match_keypoints(img1_path,img2_path):
    sift = cv2.SIFT_create()

    image1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
    
    bf = cv2.BFMatcher()

    # KNN match
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    # Get best match
    best_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            best_matches.append(m)
    # draw image
    match_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, best_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("output", 1000, 500) 
    cv2.imshow('output',match_image)
    cv2.waitKey(0)
# 5.1 
def Show_Augmented_images():
        train_transform = transforms.Compose([
        transforms.RandomAutocontrast(p=0.5),
        transforms.ColorJitter(saturation=0.2, brightness=0.2),
        transforms.RandomGrayscale(p=0.3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20)
        ])
        folder_path='./Dataset_CvDl_Hw1\Q5_image\Q5_1'
        images = []
        filenames = []
        for filename in os.listdir(folder_path):
            img = Image.open(os.path.join(folder_path,filename))
            if img is not None:
                # Data Augmenatatin
                img = train_transform(img)
                images.append(img)
                filenames.append(filename.split(".")[0])
        
        plt.figure(figsize=(10, 10))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            img = images[i]
            plt.imshow(img)
            title = filenames[i]
            plt.title(title)
            plt.axis('off')
        plt.show()
# 5.2 sol.
def Show_Model_Summary():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.vgg19_bn(num_classes=10)
    model.to(device)
    summary(model, (3, 32, 32), -1, device)
# 5.3 sol.
def Show_Accuracy_and_Loss():
    image1_path = './model\\result_picture\\loss_plot.png'
    image2_path = './model\\result_picture\\accuracy_plot.png'
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    Hori = np.concatenate((image1, image2), axis=1)
    cv2.imshow('output',Hori)
    cv2.waitKey(0)     
# 5.4 sol.
def Inference(image_path):
        classes = ['airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck']
        # define model
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = models.vgg19_bn(num_classes=10)
        weight = torch.load('./model\\vgg19_bn.pth')
        model.load_state_dict(weight)
        model.eval()
        model.to(device)    
        # load image
        transform = transforms.Compose([
        transforms.ToTensor()
        ])
        image = Image.open(image_path)
        image = transform(image)
        image = image.unsqueeze(0)
        image = image.to(device)
        # Inference
        with torch.no_grad():
            output = model(image)
            probabilities = F.softmax(output,1)
            _, predicts = torch.max(output, 1)
            predict_class = classes[predicts]
            output_np = probabilities.cpu().detach().numpy().ravel()
            plt.figure(figsize=(10, 10))
            plt.bar(range(len(output_np)), output_np)
            plt.xticks(range(len(output_np)), classes) 
            plt.xlabel('Class')
            plt.ylabel('Probability')
            plt.title('Probability of each class')
            plt.show()
        return predict_class