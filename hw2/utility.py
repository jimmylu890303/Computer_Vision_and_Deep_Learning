import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torchvision.transforms as transforms 
from torchsummary import summary
from torchvision import models
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.datasets.folder import default_loader
import os
import random
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vgg_model = models.vgg19_bn(num_classes=10)
weight = torch.load('.\model\\vgg19bn_best.pth')
vgg_model.load_state_dict(weight)
vgg_model.eval()
vgg_model.to(device)
resnet50_model=models.resnet50(num_classes=2)
res_weight = torch.load('.\model\\resnet50_RandomEarsing.pth')
resnet50_model.load_state_dict(res_weight)
resnet50_model.eval()
resnet50_model.to(device)
# 1.1 Sol.
def BackgroundSubstraction(video_path):

    cap = cv2.VideoCapture(video_path)
    if cap.isOpened() == False:
        print('ERROR ! Open Video Failed !')
        return
    # 創建背景減除器
    history = 500  # 歷史記錄數
    dist2Threshold = 400  # 距離閾值
    subtractor = cv2.createBackgroundSubtractorKNN(history, dist2Threshold, detectShadows=True)

    while (cap.isOpened()):
        ret, frame = cap.read()
        if(ret==False):
           break
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        mask = subtractor.apply(blurred_frame)
        mask_result = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        result = cv2.bitwise_and(frame, frame, mask=mask)
        imgs = np.hstack([frame,mask_result,result])
        cv2.imshow("BackgroundSubstraction", imgs)
        cv2.waitKey(1)

# 2.1 SOL 
def detect_and_mark_point(frame):
    # 轉換為灰度影像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 設定 goodFeaturesToTrack 參數
    max_corners = 1
    quality_level = 0.3
    min_distance = 7
    block_size = 7

    # 執行 goodFeaturesToTrack 檢測特徵點
    corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance, blockSize=block_size)

    if corners is not None:
        # 將特徵點座標轉為整數
        corners = np.intp(corners)
        # 繪製紅色十字標記在特徵點位置
        for corner in corners:
            x, y = corner.ravel()
            cv2.line(frame, (x, y - 10), (x, y + 10), (0, 0, 255), 4)
            cv2.line(frame, (x - 10, y), (x + 10, y ), (0, 0, 255), 4)
    return frame
def preprocess(video_path):
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened() == False:
        print('ERROR ! Open Video Failed !')
        return
    # 讀取第一幀
    ret, first_frame = cap.read()
    # 檢測並標記點
    marked_frame = detect_and_mark_point(first_frame)
    # 顯示結果
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("Result", 1000, 500) 
    cv2.imshow('Result', marked_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 2.2 Sol.
def VideoTracking(video_path):
    cap = cv2.VideoCapture(video_path)
    # optical flow 參數
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # 抓取第一個frame
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    # 設定 goodFeaturesToTrack 參數
    max_corners = 1
    quality_level = 0.3
    min_distance = 7
    block_size = 7
    # 執行 goodFeaturesToTrack 檢測特徵點
    p0 = cv2.goodFeaturesToTrack(old_gray, max_corners, quality_level, min_distance, blockSize=block_size)
    
    # 創建空白圖畫
    mask = np.zeros_like(old_frame)
    while(1):
        ret,frame = cap.read()
        if ret==False:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 計算 optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), (0,100,255), 4)
            cv2.line(frame, (int(a),int(b)+10), (int(a),int(b)-10), (0, 0, 255), 4)
            cv2.line(frame, (int(a)+10,int(b)), (int(a)-10,int(b)), (0, 0, 255), 4)
        img = cv2.add(frame,mask)
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL) 
        cv2.resizeWindow("Result", 1000, 500) 
        cv2.imshow('Result',img)
        cv2.waitKey(1)

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
    
# 3.1 Sol.
def PCA_SOL(image_path):

    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Normalize Image
    gray_image_normalized = gray_image / 255.0

    # Find min components n s.t mse<0.3
    min_dimension = min(gray_image.shape)
    n = 1
    reconstruction_error = float('inf')
    while n <= min_dimension:
        pca = PCA(n_components=n)
        transformed = pca.fit_transform(gray_image_normalized)
        reconstructed = pca.inverse_transform(transformed)
        reconstructed_image = reconstructed.reshape(gray_image.shape)
        mse = np.mean((gray_image - reconstructed_image*255) ** 2)
        if mse <= 3.0:
            reconstruction_error = mse
            break
        n += 1
    print(f"Reconstruct Image with {n} components")

    # Reconstructed Image
    pca = PCA(n_components=n)
    transformed = pca.fit_transform(gray_image_normalized)
    reconstructed = pca.inverse_transform(transformed)
    reconstructed_image = reconstructed.reshape(gray_image.shape)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Gray Image')
    plt.imshow(gray_image, cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title(f'Using {n} Components Reconstructed Image\nMSE: {reconstruction_error:.4f}')
    plt.imshow(reconstructed_image, cmap='gray')
    plt.axis('off')
    plt.show()

# 4.1 Sol.
def Show_Model_Summary():
    summary(vgg_model, (3, 32, 32), -1, device)

# 4.3 Sol.
def predict_mnist(img):
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

    img = transform(img)
    img = img.to(device)  

    output = vgg_model(img.unsqueeze(0))  
    probabilities = F.softmax(output,1)
    _, predicted = torch.max(output, 1)
    predicted_label = classes[predicted]
    output_np = probabilities.cpu().detach().numpy().ravel()
    plt.figure(figsize=(10, 10))
    plt.bar(range(len(output_np)), output_np)
    plt.xticks(range(len(output_np)), classes) 
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.title('Probability of each class')
    plt.show()

    return predicted_label

# 5.1 Sol.
class CVDL(Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        # classes為class種類 class_to_idx為該class的index
        self.classes, self.class_to_idx = self._find_classes(root)
        self.imgs = self._make_dataset(root, self.class_to_idx)

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self, dir, class_to_idx):
        # (img_path,label)
        images = []
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(dir, target_class)
            if not os.path.isdir(target_dir):
                continue

            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, class_index)
                    images.append(item)
        return images
    def get_class_data(self, class_name):
        if class_name not in self.class_to_idx:
            raise ValueError(f"Class '{class_name}' not found in the dataset.")
        
        class_index = self.class_to_idx[class_name]
        class_data = [(path, target) for path, target in self.imgs if target == class_index]
        return class_data
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
def show_class_image():
    classes=['Cat','Dog']
    transform = train_transform = transforms.Compose([
                 transforms.Resize((224,224))])
    dataset = CVDL(root='.\inference_dataset',transform=transform)
    cats = dataset.get_class_data('Cat')
    dogs = dataset.get_class_data('Dog')
    num_of_cats = len(cats)
    num_of_dogs = len(dogs)
    idx_of_cat = random.randint(0, num_of_cats-1)
    idx_of_dog = random.randint(0, num_of_dogs-1)
    cat_Imgpath, cat_label = cats[idx_of_cat]
    dog_Imgpath, dog_label = dogs[idx_of_dog]
    cat_img = Image.open(cat_Imgpath)
    cat_img = transform(cat_img)
    dog_img = Image.open(dog_Imgpath)
    dog_img = transform(dog_img)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.title(classes[cat_label])
    plt.imshow(cat_img)
    plt.axis('off')
    plt.subplot(122)
    plt.title(classes[dog_label])
    plt.imshow(dog_img)
    plt.axis('off') 
    plt.show() 

# 5.2 Sol.
def show_ResNet50_Model_Summary():
    summary(resnet50_model, (3, 224, 224), -1, device)

# 5.3 Sol
def show_accurancy_comparison():
    img = Image.open('.\model\Accurancy_Comparison.png')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# 5.4 Sol.
def predict_resnet50(img_path):
    classes = ['Cat','Dog']

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path)
    img = transform(img)
    img = img.to(device)  

    output = resnet50_model(img.unsqueeze(0))  
    _, predicted = torch.max(output, 1)
    predicted_label = classes[predicted]
    return predicted_label