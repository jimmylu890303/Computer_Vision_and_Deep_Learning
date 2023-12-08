import cv2
import numpy as np

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
    # 釋放資源與關閉視窗
    cap.release()
    cv2.destroyAllWindows()
    print('Done')

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