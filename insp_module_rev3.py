"""更新履歴"""                     
"""rev1 23/06/25 作成"""
"""rev2 23/07/05 作成 template matchingをtmpとtestで逆転"""
"""rev3 23/07/08 作成 スプリットデータで区切る前にフィルターをかける"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from sklearn.linear_model import LinearRegression
from sklearn import svm
import pickle
import gc

"""定数"""
class insp:

    #--------------#
    #  定数初期化  #
    #--------------#
    def __init__(self):
        self.CHIP_THRESHOLD = 50 #チップの明るさ
        self.CHIP_IMAGE_SIZE = 1500 #チップイメージの大きさ
        self.CHIP_SIZE_CHK_1 = 1000 #チップ輪郭認識時のサイズ規格MIN
        self.CHIP_SIZE_CHK_2 = 2000 #チップ輪郭認識時のサイズ規格MAX
        self.PADDING_X = 10 #チップ最外周からパディングするピクセル(X方向)
        self.PADDING_Y = 10 #チップ最外周からパディングするピクセル(Y方向)
        self.TEST_PADDING_X = 2 #チップ最外周からパディングするピクセル(X方向)
        self.TEST_PADDING_Y = 2 #チップ最外周からパディングするピクセル(Y方向)
        self.ONE_TEST_SIZE=100 #1回でテストするサイズ
        self.ERODE_NUM = 10
        self.DILATE_NUM = 10
        self.NU = 0.001
        self.GAMMA = 0.001
        self.STD_COEFF = 6
        self.STD_TYPE = "FIX"
        self.BRIGHTNESS_MATCH = 1 #良品画像と輝度を合わせる
        self.BILATERAL_SIZE = 9 #bilateral filterの程度
        self.UNSHARP_SIZE = 1 #bilateral filterの程度
        self.ADJUST_ALPHA = 2.5 #画像明るさ変更係数
        self.ADJUST_BETA = 0.0 #画像明るさ変更切片
        self.SOBEL_K = 3 #SOBELフィルタのパラメータ
        self.sigmoid_coeff = 1.0 #sigmoidフィルタの係数
        self.sigmoid_std = 128.0 #sigmoidフィルタの中心点
        self.theta_list=[]
        self.FILTER_LIST=[]

        self.GOOD_SAMPLE_FOLDER="C:/workspace/insp_by_oneclasssvm_ver2/good_sample/" #良品画像が入ったフォルダ
        self.PARENT_IMG_FILE="C:/workspace/insp_by_oneclasssvm_ver2/00000AA.JPG" #良品画像が入ったフォルダ
        self.TEST_SAMPLE_FOLDER="C:/workspace/insp_by_oneclasssvm_ver2/test_sample/" #テスト画像が入ったフォルダ

        #outputFolder
        self.GOOD_RESULT_FILE = "C:/workspace/insp_by_oneclasssvm_ver2/good_data.csv"
        self.OUTPUT_RESULT_FILE = "C:/workspace/insp_by_oneclasssvm_ver2/result_data.csv"
        self.OUTPUT_ALL_IMAGE = "C:/workspace/insp_by_oneclasssvm_ver2/all_image/"
        self.OUTPUT_FAIL_IMAGE = "C:/workspace/insp_by_oneclasssvm_ver2/fail_image/"
        self.OUTPUT_GOOD_IMAGE = "C:/workspace/insp_by_oneclasssvm_ver2/good_image/"
        self.SPLIT_DATA_FILE = "C:/workspace/insp_by_oneclasssvm_ver2/rect_0614.csv"
        self.PICKLE_MODEL_FILE = "C:/workspace/insp_by_oneclasssvm_ver2/model.pickle"
        self.PICKLE_SCORE_FILE = "C:/workspace/insp_by_oneclasssvm_ver2/score.pickle"

        #矩形リストの初期値
        #0->top, 1->right, 2->bottom, 3->left
        self.rectangle_point=[[[2300,1450],[2800,1750]],[[3300,2250],[3500,2900]],[[2350,3000],[2700,3250]],[[1750,2000],[2000,2300]]]

    #--------------#
    #     関数     #
    #--------------#
    #設定ファイルの読み込み
    def read_setting_file(self,TypeName,LotNo,test_address,parameter_address,result_address):

        #各アドレスの作成
        self.TEST_SAMPLE_FOLDER = test_address.replace("%LOT%",LotNo).replace("%PRODUCT%",TypeName) #テストサンプル置き場
        parameter_folder = parameter_address.replace("%LOT%",LotNo).replace("%PRODUCT%",TypeName) #パラメーターファイル置き場
        result_folder = result_address.replace("%LOT%",LotNo).replace("%PRODUCT%",TypeName) #結果ファイル置き場

        #各フォルダの設定
        self.GOOD_SAMPLE_FOLDER=parameter_folder+"good_sample/" #良品画像が入ったフォルダ
        self.PARENT_IMG_FILE=parameter_folder+"parent_img.JPG" #良品画像が入ったフォルダ

        #outputFolder
        self.GOOD_RESULT_FILE = parameter_folder + "good_data.csv"
        self.OUTPUT_RESULT_FILE = result_folder + "result_data_"+LotNo+".csv"
        self.OUTPUT_ALL_IMAGE = result_folder + "all_image/"
        self.OUTPUT_FAIL_IMAGE = result_folder + "fail_image/"
        self.OUTPUT_GOOD_IMAGE = result_folder + "good_image/"
        self.SPLIT_DATA_FILE = parameter_folder + "split_data.csv"
        self.PICKLE_MODEL_FILE = parameter_folder + "model.pickle"
        self.PICKLE_SCORE_FILE = parameter_folder + "score.pickle"

        #設定ファイルの読み込み
        setting_file_address = parameter_folder+"setting.txt"
        setting_file = open(setting_file_address,"r")
        setting_line = setting_file.readline()

        while setting_line:
            
            setting_item = setting_line.split(" ")
            
            if setting_item[0] == "CHIP_THRESHOLD":
                self.CHIP_THRESHOLD = int(setting_item[1])
            elif setting_item[0] == "CHIP_IMAGE_SIZE":
                self.CHIP_IMAGE_SIZE = int(setting_item[1])
            elif setting_item[0] == "CHIP_SIZE_CHK_1":
                self.CHIP_SIZE_CHK_1 = int(setting_item[1])
            elif setting_item[0] == "CHIP_SIZE_CHK_2":
                self.CHIP_SIZE_CHK_2 = int(setting_item[1])
            elif setting_item[0] == "PADDING_X":
                self.PADDING_X = int(setting_item[1])
            elif setting_item[0] == "PADDING_Y":
                self.PADDING_Y = int(setting_item[1])
            elif setting_item[0] == "TEST_PADDING_X":
                self.TEST_PADDING_X = int(setting_item[1])
            elif setting_item[0] == "TEST_PADDING_Y":
                self.TEST_PADDING_Y = int(setting_item[1])
            elif setting_item[0] == "ERODE_NUM":
                self.ERODE_NUM = int(setting_item[1])
            elif setting_item[0] == "DILATE_NUM":
                self.DILATE_NUM = int(setting_item[1])
            elif setting_item[0] == "NU":
                self.NU = float(setting_item[1])
            elif setting_item[0] == "GAMMA":
                self.GAMMA = float(setting_item[1])
            elif setting_item[0] == "STD_COEFF":
                self.STD_COEFF = float(setting_item[1])
            elif setting_item[0] == "STD_TYPE":
                self.STD_TYPE = setting_item[1].replace("\n","")
            elif setting_item[0] == "BRIGHTNESS_MATCH":
                self.BRIGHTNESS_MATCH = int(setting_item[1])
            elif setting_item[0] == "BILATERAL_SIZE":
                self.BILATERAL_SIZE = int(setting_item[1])
            elif setting_item[0] == "UNSHARP_SIZE":
                self.UNSHARP_SIZE = int(setting_item[1])
            elif setting_item[0] == "ADJUST_ALPHA":
                self.ADJUST_ALPHA = float(setting_item[1])
            elif setting_item[0] == "ADJUST_BETA":
                self.ADJUST_BETA = float(setting_item[1])
            elif setting_item[0] == "FILTER_LIST" and len(setting_item)>1:
                for filter_name in setting_item[1].split(","):
                    self.FILTER_LIST.append(filter_name.replace("\n",""))
            elif setting_item[0] == "SOBEL_K":
                self.SOBEL_K = int(setting_item[1])
            elif setting_item[0] == "SIGMOID_COEFF":
                self.sigmoid_coeff = float(setting_item[1])
            elif setting_item[0] == "SIGMOID_STD":
                self.sigmoid_std == float(setting_item[1])
            elif setting_item[0] == "RECT_LEFT":
                data = setting_item[1].split(",")
                self.rectangle_point[3][0][0] = int(data[0])
                self.rectangle_point[3][0][1] = int(data[1])
                self.rectangle_point[3][1][0] = int(data[2])
                self.rectangle_point[3][1][1] = int(data[3])
            elif setting_item[0] == "RECT_TOP":
                data = setting_item[1].split(",")
                self.rectangle_point[0][0][0] = int(data[0])
                self.rectangle_point[0][0][1] = int(data[1])
                self.rectangle_point[0][1][0] = int(data[2])
                self.rectangle_point[0][1][1] = int(data[3])
            elif setting_item[0] == "RECT_RIGHT":
                data = setting_item[1].split(",")
                self.rectangle_point[1][0][0] = int(data[0])
                self.rectangle_point[1][0][1] = int(data[1])
                self.rectangle_point[1][1][0] = int(data[2])
                self.rectangle_point[1][1][1] = int(data[3])
            elif setting_item[0] == "RECT_BOTTOM":
                data = setting_item[1].split(",")
                self.rectangle_point[2][0][0] = int(data[0])
                self.rectangle_point[2][0][1] = int(data[1])
                self.rectangle_point[2][1][0] = int(data[2])
                self.rectangle_point[2][1][1] = int(data[3])

            setting_line = setting_file.readline()

        setting_file.close()

        return 0

    #チップの各辺の傾きを求める
    def get_line(self,img,rp):

        #各辺の傾き、切片を入れる変数 
        #0->top, 1->right, 2->bottom, 3->left
        a_b_list = [[0.,0.] for i in range(4)]

        #チップの辺を取得するための2値化レベル
        threshold = self.CHIP_THRESHOLD

        #top->right->bottom->leftでループを回す
        for n,p in enumerate(rp):

            #矩形画像切り出し
            img_slice = img[p[0][1]:p[1][1],p[0][0]:p[1][0]]
            img_slice = img_slice > threshold

            px = []
            py = []

            #topの傾き切片を求める
            if n==0:
                for i in range(p[0][0],p[1][0]):

                    x = i-p[0][0]
                    y = np.argmax(img_slice[:,x]==True)

                    px.append(i)
                    py.append(y+p[0][1])

                px = np.array(px)
                py = np.array(py)

                px = px.reshape(-1,1)
                py = py.reshape(-1,1)

                clf = LinearRegression(fit_intercept=True)
                clf.fit(X=px,y=py)
                y_hat = clf.predict(px)

                diff = py - y_hat

                weight = (np.max(np.abs(diff))-np.abs(diff)).astype("float32").T[0]**2

                clf.fit(X=px,y=py,sample_weight=weight)

                a,b = clf.coef_,clf.intercept_

                a_b_list[n][0] = a
                a_b_list[n][1] = b

            elif n==1:
                for i in range(p[0][1],p[1][1]):
                    y = i-p[0][1]
                    x = len(img_slice[y,:])-np.argmax(img_slice[y,:][::-1]==True)

                    px.append(x+p[0][0])
                    py.append(i)

                px = np.array(px)
                py = np.array(py)

                px = px.reshape(-1,1)
                py = py.reshape(-1,1)

                clf = LinearRegression(fit_intercept=True)
                clf.fit(X=px,y=py)
                y_hat = clf.predict(px)

                diff = py - y_hat

                weight = (np.max(np.abs(diff))-np.abs(diff)).astype("float32").T[0]**2

                clf.fit(X=px,y=py,sample_weight=weight)

                a,b = clf.coef_,clf.intercept_

                a_b_list[n][0] = a
                a_b_list[n][1] = b

            elif n==2:
                for i in range(p[0][0],p[1][0]):

                    x = i-p[0][0]
                    y = len(img_slice[:,x])-np.argmax(img_slice[:,x][::-1]==True)

                    px.append(i)
                    py.append(y+p[0][1])

                px = np.array(px)
                py = np.array(py)

                px = px.reshape(-1,1)
                py = py.reshape(-1,1)

                clf = LinearRegression(fit_intercept=True)
                clf.fit(X=px,y=py)
                y_hat = clf.predict(px)

                diff = py - y_hat

                weight = (np.max(np.abs(diff))-np.abs(diff)).astype("float32").T[0]**2

                clf.fit(X=px,y=py,sample_weight=weight)

                a,b = clf.coef_,clf.intercept_

                a_b_list[n][0] = a
                a_b_list[n][1] = b

            elif n==3:
                for i in range(p[0][1],p[1][1]):
                    y = i-p[0][1]
                    x = np.argmax(img_slice[y,:]==True)

                    px.append(x+p[0][0])
                    py.append(i)

                px = np.array(px)
                py = np.array(py)

                px = px.reshape(-1,1)
                py = py.reshape(-1,1)

                clf = LinearRegression(fit_intercept=True)
                clf.fit(X=px,y=py)
                y_hat = clf.predict(px)

                diff = py - y_hat

                weight = (np.max(np.abs(diff))-np.abs(diff)).astype("float32").T[0]**2

                clf.fit(X=px,y=py,sample_weight=weight)

                a,b = clf.coef_,clf.intercept_

                a_b_list[n][0] = a
                a_b_list[n][1] = b

        return a_b_list

    #画像を回転させる
    def get_rotate_image(self,img,a_b_list):

        #a_b_list
        #0->top, 1->right, 2->bottom, 3->left

        #チップを回転させた画像を取得
        lt= np.zeros(2,dtype=int)
        rt= np.zeros(2,dtype=int)
        rb= np.zeros(2,dtype=int)
        lb= np.zeros(2,dtype=int)

        rectlt= np.zeros(2,dtype=int)
        rectrb= np.zeros(2,dtype=int)

        t = a_b_list[0] 
        r = a_b_list[1] 
        b = a_b_list[2] 
        l = a_b_list[3] 

        #チップの左上の点
        lt[0] = int((l[1]-t[1])/(t[0]-l[0]))
        lt[1] = int(t[0]*lt[0] + t[1])

        #チップの右上の点
        rt[0] = int((r[1]-t[1])/(t[0]-r[0]))
        rt[1] = int(t[0]*rt[0] + t[1])

        #チップの右下の点
        rb[0] = int((r[1]-b[1])/(b[0]-r[0]))
        rb[1] = int(b[0]*rb[0] + b[1])

        #チップの左下の点
        lb[0] = int((l[1]-b[1])/(b[0]-l[0]))
        lb[1] = int(b[0]*lb[0] + b[1])

        #外接四角形の情報 x,y,w,h
        x = min([lt[0],rt[0],rb[0],lb[0]])
        y = min([lt[1],rt[1],rb[1],lb[1]])
        w = max([lt[0],rt[0],rb[0],lb[0]]) - x
        h = max([lt[1],rt[1],rb[1],lb[1]]) - y

        if (w<self.CHIP_SIZE_CHK_1 or w>self.CHIP_SIZE_CHK_2) or (h<self.CHIP_SIZE_CHK_1 or h>self.CHIP_SIZE_CHK_2):
            print("チップの外形がおかしいです")
            sys.exit()

        img_slice = img[y-self.PADDING_Y*2:y+h+self.PADDING_Y*2,x-self.PADDING_X*2:x+w+self.PADDING_X*2]

        center = np.array([x+w/2,y+h/2])

        #左上の角度を求める
        center2chipLT = lt - center
        center2rectLT = np.array([x,y]) - center

        thetaLT = np.arccos(np.dot(center2chipLT,center2rectLT)/(np.linalg.norm(center2chipLT)*np.linalg.norm(center2rectLT)))

        #右下の角度を求める
        center2chipRB = rb - center
        center2rectRB = np.array([x+w,y+h]) - center

        thetaRB = np.arccos(np.dot(center2chipRB,center2rectRB)/(np.linalg.norm(center2chipRB)*np.linalg.norm(center2rectRB)))

        #角度は2つの角度を比較して小さいほうを取る
        #theta = min(thetaLT,thetaRB)
        theta = (thetaLT+thetaRB)/2

        rot_matrix = cv2.getRotationMatrix2D((w/2,h/2),np.degrees(theta),1)

        img_affine = cv2.warpAffine(img_slice,rot_matrix,(w+self.PADDING_X*2,h+self.PADDING_Y*2))

        return img_affine,theta

    #回転したチップ周辺画像からチップ周辺部分だけを抜き出す
    #flag->0 パディング有りの画像を返す、flag->1 パディング無しの画像を返す
    def get_chip_image(self,img_affine,flag):

        #縮小・膨張処理用のカーネル
        kernel = np.ones((5,5),np.uint8)

        #輪郭取得用の二値化処理
        _,img_affine_binary = cv2.threshold(img_affine,self.CHIP_THRESHOLD,255,cv2.THRESH_BINARY)

        img_affine_binary = cv2.erode(img_affine_binary,kernel,iterations=self.ERODE_NUM)
        img_affine_binary = cv2.dilate(img_affine_binary,kernel,iterations=self.DILATE_NUM)

        #findContoursで輪郭取得
        outside_contours, hierarchy = cv2.findContours(img_affine_binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        #取得した輪郭の中からチップ全体の輪郭を探す
        for i, contour in enumerate(outside_contours):

            x,y,w,h = cv2.boundingRect(contour)

            if w<self.CHIP_SIZE_CHK_1 or h<self.CHIP_SIZE_CHK_1:
                continue
           
            #チップ部分をスライスで抜き出し(X,Yそれぞれにパディングを取る)
            img_affine = img_affine[y-self.PADDING_Y:y+h+self.PADDING_Y,x-self.PADDING_X:x+w+self.PADDING_X]

        #フラッグによって返す画像を変える
        if flag==0:
            img_chip = cv2.resize(img_affine,(self.CHIP_IMAGE_SIZE+2*self.PADDING_X,self.CHIP_IMAGE_SIZE+2*self.PADDING_Y))
        elif flag==1:
            #画像部分だけを抜き出し
            img_chip = cv2.resize(img_affine,(self.CHIP_IMAGE_SIZE+2*self.PADDING_X,self.CHIP_IMAGE_SIZE+2*self.PADDING_Y))
            img_chip = img_chip[self.PADDING_X:self.CHIP_IMAGE_SIZE+self.PADDING_X,self.PADDING_Y:self.CHIP_IMAGE_SIZE+self.PADDING_Y]
        else:
            print("get_chip_imageに適切なflagを渡せていません")
            sys.exit()

        return img_chip,x,y,w,h

    #見本画像を取得する
    def get_parent_img(self,parent_img_path):

        parent_img = cv2.imread(parent_img_path,cv2.IMREAD_GRAYSCALE)
        a_b_list = self.get_line(parent_img,self.rectangle_point)
        img_good_affine,_=self.get_rotate_image(parent_img,a_b_list)
        img_good,_,_,_,_ = self.get_chip_image(img_good_affine,0)
           
        return img_good

    #お手本画像との輝度合わせ
    def brightness_match(self,image,data):

        #data[0]->下側の平均 data[1]->上側の平均 data[2]->下側の分散 data[3]->上側の分散

        #大津の2値化の閾値を取得
        rst_test,_ = cv2.threshold(image,0,255,cv2.THRESH_OTSU)

        #閾値下側の平均値と上側の平均値を取得
        lower_mean_test = image[image<rst_test].mean()
        higher_mean_test = image[image>=rst_test].mean()

        #閾値下側の分散と上側の分散を取得
        lower_std_test = image[image<rst_test].std()
        higher_std_test = image[image>=rst_test].std()

        #lower側の輝度合わせ
        image = np.where(image<rst_test,(image-lower_mean_test)/lower_std_test*data[2]+data[0],image)

        #higher側の輝度合わせ
        image = np.where(image>=rst_test,(image-higher_mean_test)/higher_std_test*data[3]+data[1],image)

        #調整
        #image = np.where(image>255,255,image)
        #image = np.where(image<0,0,image)
        image = np.clip(image,0,255)
        image = image.astype(np.uint8)

        return image

    #ノイズ除去用の輝度調整
    def adjust(self,image,alpha,beta):

        adjust_image = alpha*image+beta
        return np.clip(adjust_image,0,255).astype(np.uint8)

    #フィルター処理を実施
    def image_filter(self,image):

        #鮮鋭化処理のカーネル
        k=self.UNSHARP_SIZE
        unsharp_kernel = np.array([[-k/9,-k/9,-k/9],
                                  [-k/9,1+8*k/9,k/9],
                                  [-k/9,-k/9,-k/9]],np.float32)

        img_after_filter = image.copy()

        for filter_name in self.FILTER_LIST:

            if filter_name == "BILATERAL":
                img_after_filter = cv2.bilateralFilter(img_after_filter,d=self.BILATERAL_SIZE,sigmaColor=100,sigmaSpace=10)
            elif filter_name == "UNSHARP":
                img_after_filter = cv2.filter2D(img_after_filter,-1,unsharp_kernel).astype("uint8")
            elif filter_name == "ADJUST":
                img_after_filter = self.adjust(img_after_filter,self.ADJUST_ALPHA,self.ADJUST_BETA)
            elif filter_name == "SOBEL":
                grid_x = cv2.Sobel(img_after_filter,cv2.CV_32F,1,0,self.SOBEL_K)
                grid_y = cv2.Sobel(img_after_filter,cv2.CV_32F,0,1,self.SOBEL_K)
                img_after_filter = np.sqrt(grid_x**2+grid_y**2).astype("uint8")
            elif filter_name == "SIGMOID":
                a = self.sigmoid_coeff
                b = self.sigmoid_std
                img_after_filter = 255/(1+np.exp(-a*(img_after_filter-b)/255))
                img_after_filter = img_after_filter.astype("uint8")
            else:
                print("フィルターの指定が間違っています")
                sys.exit()

        return img_after_filter

    #差分を取得する
    def get_diff_image_list(self,img_good_average,image,split_data,brightness_data,n,mode,img_path):
       
        diff_image_list=[]

        a_b_list = self.get_line(image,self.rectangle_point)
        img_affine,theta= self.get_rotate_image(image,a_b_list)
        img,x,y,w,h = self.get_chip_image(img_affine,0)

        #お手本との輝度合わせを実施する
        if self.BRIGHTNESS_MATCH==1:

            img = (img-np.mean(img))/np.std(img)*brightness_data[1]+brightness_data[0]

            #8bitに調整
            img = np.clip(img,0,255)
            img = img.astype(np.uint8)

        if mode==0:
            #テスト時
            self.theta_list.append([theta,x,y,w,h])
            img_name = self.OUTPUT_ALL_IMAGE+str(n)+".jpg"
            cv2.imwrite(img_name,img)
            print("\r{}枚目 {}を処理中".format(n,img_path),end="")
        elif mode==1:
            #良品学習時
            print("\r{}枚目を処理中".format(n),end="")

        #スプリットする前にフィルターをかける
        if len(self.FILTER_LIST) > 0:
            img = self.image_filter(img)

        #split_dataで画像を切り取っていく
        for s_num,s in enumerate(split_data):
            x = s[0] #5~1495
            y = s[1] #5
            lx = s[2] #60
            ly = s[3] #65

            template = img_good_average[y-self.TEST_PADDING_Y:y+ly+self.TEST_PADDING_Y,x-self.TEST_PADDING_X:x+lx+self.TEST_PADDING_X]
            test_image = img[y:y+ly,x:x+lx]
            
            #テンプレートマッチング
            res = cv2.matchTemplate(template,test_image,cv2.TM_CCOEFF_NORMED)

            #最もマッチングしている部分を取り出し
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            tmp_image_match = template[max_loc[1]:max_loc[1]+ly,max_loc[0]:max_loc[0]+lx]

            #良品とテンプレートにfilterをかける
            #test_image_after_filter = self.image_filter(test_image)
            #template_after_filter = self.image_filter(tmp_image_match)

            img_diff = cv2.absdiff(test_image,tmp_image_match)

            diff_image_list.append((img_diff/255).flatten()) #diff imageを0~1に正規化

        return diff_image_list
     
    def learn_good_feature(self,good_diff_image_list,split_num):

        models=[]

        #train_features -> split_num X 良品見本数
        for i,features in enumerate(good_diff_image_list):
            print("\r{}番目の特徴のモデル作成".format(i+1),end="")
            features = np.array(features)
            model = svm.OneClassSVM(nu=self.NU,gamma=self.GAMMA)
            model.fit(features)
            models.append(model)

        #pickleファイルにモデルを書き出し
        with open(self.PICKLE_MODEL_FILE,mode="wb") as fo:
            pickle.dump(models,fo)

        return models

    #良品学習スキップ時にpickleファイルを読み込む
    def load_model_pickle(self):
        
        with open(self.PICKLE_MODEL_FILE,mode="rb") as fm:
            models = pickle.load(fm)

        return models

    #良品学習スキップ時にpickleファイルを読み込む
    def load_score_pickle(self):
        
        with open(self.PICKLE_SCORE_FILE,mode="rb") as fs:
            scores = pickle.load(fs)

        return scores


    def good_predict(self,models,good_diff_image_list,split_num):

        results = []

        #good_features -> split_num X テスト数
        for i,features in enumerate(good_diff_image_list):
            features = np.array(features)
            result = models[i].score_samples(features)
            results.append(result)

        #pickleファイルに良品スコアを書き出し
        with open(self.PICKLE_SCORE_FILE,mode="wb") as fo:
            pickle.dump(results,fo)

        return results

    def result_predict(self,models,test_diff_image_list,split_num):

        test_features=[[] for i in range(split_num)]
        results = []

        for i,images in enumerate(test_diff_image_list):
            for j,image in enumerate(images):
                test_features[j].append(image.flatten())

        #train_features -> split_num X テスト数
        for i,features in enumerate(test_features):
            features = np.array(features)
            result = models[i].score_samples(features)
            result_list = result.tolist()
            results.append(result_list)
        return results

    def set_standards(self,test_scores):

        standards = []

        for score in test_scores:
            med = np.median(score) #良品学習スコアの内最小値を規格に置く
            std = np.std(score)
            standard = med-self.STD_COEFF*std
            #standard = med-0.04
            standards.append(standard)

        return standards

    def judge_pass_fail(self,standards,test_predictions,split_data,test_num):

        #predictionsはsplit_num X テストサンプル数

        judge_result=[[] for i in range(test_num)] #サンプル数Xsplit_num

        for i,test_prediction in enumerate(test_predictions): #i split_num
            for j,p in enumerate(test_prediction): # 画像数
                if p < standards[i]:
                    judge_result[j].append("1")
                else:
                    judge_result[j].append("0")

        #不良個所に矩形を書く
        for i,result in enumerate(judge_result):
            if "1" in result:
                fail_img = cv2.imread(self.OUTPUT_ALL_IMAGE+str(i+1)+".jpg",cv2.COLOR_GRAY2RGB)
                for j,r in enumerate(result):
                    if r=="1":
                        x1 = split_data[j][0] #矩形左上のx座標
                        y1 = split_data[j][1] #矩形左上のy座標
                        x2 = x1+split_data[j][2] #矩形右下のx座標
                        y2 = y1+split_data[j][3] #矩形右下のy座標
                        cv2.rectangle(fail_img,(x1,y1),(x2,y2),(0,200,0),1)
                        cv2.putText(fail_img,str(j+1),(x1,y1),cv2.FONT_HERSHEY_PLAIN,1.5,(0,200,0),1,cv2.LINE_AA)
                cv2.imwrite(self.OUTPUT_FAIL_IMAGE+str(i+1)+".jpg",fail_img)
            else:
                good_img = cv2.imread(self.OUTPUT_ALL_IMAGE+str(i+1)+".jpg",cv2.COLOR_GRAY2RGB)
                cv2.imwrite(self.OUTPUT_GOOD_IMAGE+str(i+1)+".jpg",good_img)

        return judge_result

    def get_split_data(self):

        split_data=[]
        std_lower=[]

        split_data_file = open(self.SPLIT_DATA_FILE,"r")

        split_data_line = split_data_file.readline()

        while split_data_line:
            d = split_data_line.replace("\n","").split(",")
            split_data.append([int(d[0]),int(d[1]),int(d[2]),int(d[3])])

            if self.STD_TYPE=="FIX":
                std_lower.append(float(d[4]))

            split_data_line = split_data_file.readline()

        split_data_file.close()

        if self.STD_TYPE=="FIX":
            return split_data,std_lower 
        else:
            return split_data

"""メイン"""
if __name__=="__main__":

    insp_test = insp()
    type_name = "ABCDEF"
    lot_no = "test_lot"
    process_type = "0"

    #各辺のポイントを取得するための矩形情報読み込み
    insp_test.read_setting_file(type_name,lot_no)

    #分割情報を取得
    split_data = insp_test.get_split_data()
    split_num = len(split_data)

    """
    良品学習実施
    """
    #初めにお手本画像を作成する:
    img_good_average = insp_test.get_parent_img(insp_test.PARENT_IMG_FILE)
    #明るくする
    img_good_average = insp_test.adjust(img_good_average,alpha=2.5,beta=0)
    good_mean = np.mean(img_good_average)
    good_std = np.std(img_good_average)

    #プロセスタイプによって分岐
    if process_type =="1" or process_type=="2":
        #良品画像読み込み
        print("良品画像読み込み中")
        good_image_list = []

        good_folder_list = os.listdir(insp_test.GOOD_SAMPLE_FOLDER)

        for folder in good_folder_list:

            images = glob.glob(insp_test.GOOD_SAMPLE_FOLDER+folder+"/*AA.JPG")

            for image in images:
                #img_good = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
                good_image_list.append(image)

        #ONE_TEST_SIZE区切りの2次元配列に変換
        good_images=[]
        for i,image_file in enumerate(good_image_list):

            if (i+1)%insp_test.ONE_TEST_SIZE==0:
                tmp.append(image_file)
                good_images.append(tmp)
                continue
            
            if i%insp_test.ONE_TEST_SIZE==0:
                tmp=[]
                tmp.append(image_file)
                continue

            tmp.append(image_file)

        good_images.append(tmp) 

        good_num = len(good_image_list)

        print("良品画像数は{}枚です".format(good_num))

        #for debug
        #cv2.imwrite("./img_good_average.jpg",img_good_average)

        #average画像の各分割のmean,stdを記録
        mean_std = []
        for i,s in enumerate(split_data):
            x = s[0] #5~1495
            y = s[1] #5
            lx = s[2] #60
            ly = s[3] #65

            img_ave = img_good_average[y:y+ly,x:x+lx]
            mean_std.append([np.mean(img_ave),np.std(img_ave)])
     
        #良品画像と平均画像の差分ベクトルを作成
        print("良品の差分ベクトル作成中")
        good_diff_image_list = [[] for s in range(split_num)]
        for i,images in enumerate(good_images):

            good_image_list = []

            #opencvで読み込んだ画像をgood_image_listに入れる(ONE_TEST_SIZE分)
            for j,image in enumerate(images):
                img_good = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
                good_image_list.append(img_good)

            #差分ベクトルをgood_diff_image_listに入れていく(good_diff_image_listはsplit_numのおおきさ)
            for j,image in enumerate(good_image_list):
                diff_image_list = insp_test.get_diff_image_list(img_good_average,image,split_data,i*insp_test.ONE_TEST_SIZE+j+1,1)

                for s in range(split_num):
                    good_diff_image_list[s].append(diff_image_list[s])

           #メモリ開放
            del good_image_list

        #one-class-svmで良品学習
        print("良品学習中")
        models = insp_test.learn_good_feature(good_diff_image_list,split_num)

        #スコア判定規格設定
        print("スコア判定規格設定中")
        good_scores = insp_test.good_predict(models,good_diff_image_list,split_num)

        del good_diff_image_list

        #良品のスコア出力
        output_file = open(insp_test.GOOD_RESULT_FILE,"w")
        output_line = ""
        for i in range(good_num):
            output_line += str(i+1)+","
        output_line += "\n"

        output_file.write(output_line)

        #結果出力
        for scores in good_scores:
            output_line = ""
            for p in scores:
                output_line += str(p) +","
            output_line += "\n"
            output_file.write(output_line)

        output_file.close()

    elif process_type=="0":
        #pickleファイルを読み込んで良品学習をスキップ
        print("モデルpickleファイルを読み込み")
        models = insp_test.load_model_pickle()
        print("スコアpickleファイルを読み込み")
        good_scores = insp_test.load_score_pickle()

    if process_type == "2":
        print("良品学習完了")
        print("正常終了")
        sys.exit()

    #規格設定
    standards = insp_test.set_standards(good_scores)

    del good_scores

    """
    ここからテスト開始
    memory error回避のため100枚ずつテスト
    
    """
    #テストデータ読み込み
    print("テスト実施中")
    test_image_files = glob.glob(insp_test.TEST_SAMPLE_FOLDER+"*AA.JPG")
    test_num = len(test_image_files)
    print("テスト画像数は{}枚です".format(test_num))

    #ONE_TEST_SIZE区切りの2次元配列に変換
    test_images=[]
    for i,image_file in enumerate(test_image_files):

        if (i+1)%insp_test.ONE_TEST_SIZE==0:
            tmp.append(image_file)
            test_images.append(tmp)
            continue
        
        if i%insp_test.ONE_TEST_SIZE==0:
            tmp=[]
            tmp.append(image_file)
            continue

        tmp.append(image_file)

    test_images.append(tmp) 

    #テストメインループ
    predictions = [[] for i in range(split_num)]

    #テスト画像のaffine変換後の画像保存
    if not os.path.exists(insp_test.OUTPUT_ALL_IMAGE):
        os.makedirs(insp_test.OUTPUT_ALL_IMAGE)

    if not os.path.exists(insp_test.OUTPUT_FAIL_IMAGE):
        os.makedirs(insp_test.OUTPUT_FAIL_IMAGE)

    if not os.path.exists(insp_test.OUTPUT_GOOD_IMAGE):
        os.makedirs(insp_test.OUTPUT_GOOD_IMAGE)

    for i,images in enumerate(test_images):

        test_image_list=[] #opencvで開いた画像を入れる
        test_diff_image_list=[] #差分ベクトルを入れる
           
        for j,image in enumerate(images): #imagesはONE_TEST_SIZE枚画像が入っている

            #画像をopencvで読み込み、リストに入れる
            img_test = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
            test_image_list.append(img_test)
        
        for j,image in enumerate(test_image_list):

            diff_image_list = insp_test.get_diff_image_list(img_good_average,image,split_data,i*insp_test.ONE_TEST_SIZE+j+1,0)
            
            test_diff_image_list.append(diff_image_list) #ONE_TEST_SIZE枚分の差分ベクトル

        #テストデータをつかってone-class-svmのスコア取得
        results = insp_test.result_predict(models,test_diff_image_list,split_num)

        for s in range(split_num):
            predictions[s] += results[s]

        #リスト初期化、メモリ開放
        del test_image_list
        del test_diff_image_list

    print("\n画像評価完了")
    print("判定中")

    #良品不良品判定
    judge_results = insp_test.judge_pass_fail(standards,predictions,split_data,test_num)

    #result出力用に転置する
    output_predictions = np.array(predictions).T

    #結果出力csvファイル
    output_file = open(insp_test.OUTPUT_RESULT_FILE,"w")
    #header出力

    output_file.write("\n")
    output_line = "no,p/f,theta,x,y,w,h,"
    for i in range(split_num):
        output_line += str(i+1)+","

    for i in range(split_num):
        output_line += str(i+1)+","

    output_line += "\n"
    output_file.write(output_line)

    #メイン結果出力
    fail_num = 0
    pass_num = 0
    for j,result in enumerate(judge_results): #j 画像数

        if "1" in result:
            output_line = str(j+1)+","+"f"+","+str(insp_test.theta_list[j][0])+","
            fail_num += 1
        else:
            output_line = str(j+1)+","+"p"+","+str(insp_test.theta_list[j])+","
            pass_num += 1

        for r in result:
            output_line += str(r) +","
    
        #score出力
        for p in output_predictions[j]:
            output_line+=str(p)+","
        
        output_line += "\n"
        output_file.write(output_line)


    output_file.close()
    print("良品数は{}、不良品数は{}".format(pass_num,fail_num))
    print("正常終了")

