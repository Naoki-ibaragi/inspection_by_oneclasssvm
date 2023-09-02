"""更新履歴"""                     
"""rev1 23/06/25 作成"""
"""rev2 23/07/05 作成 template matchingをtmpとtestで逆転"""
"""rev3 23/07/08 作成 スプリットデータで区切る前にフィルターをかける"""
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import svm
import pickle
import shutil

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
        self.STD_COEFF = []
        self.EX_COEFF = 20
        self.STD_TYPE = "FIX"
        self.BRIGHTNESS_MATCH = 1 #良品画像と輝度を合わせる
        self.theta_list=[]
        self.FILTER_LIST=[]
        self.FILTER_PARAM = []

        self.GOOD_SAMPLE_FOLDER="C:/workspace/insp_by_oneclasssvm_ver2/good_sample/" #良品画像が入ったフォルダ
        self.PARENT_IMG_FILE="C:/workspace/insp_by_oneclasssvm_ver2/00000AA.JPG" #良品画像が入ったフォルダ
        self.TEST_SAMPLE_FOLDER="C:/workspace/insp_by_oneclasssvm_ver2/test_sample/" #テスト画像が入ったフォルダ

        #outputFolder
        #複数条件に対応できるようにリスト化 23/08/15
        self.GOOD_RESULT_FILE = "C:/workspace/insp_by_oneclasssvm_ver2/good_data.csv"
        self.OUTPUT_RESULT_FILE = "C:/workspace/insp_by_oneclasssvm_ver2/result_data.csv"
        self.OUTPUT_ALL_IMAGE = "C:/workspace/insp_by_oneclasssvm_ver2/all_image/"
        self.OUTPUT_FAIL_IMAGE = "C:/workspace/insp_by_oneclasssvm_ver2/fail_image/"
        self.SPLIT_DATA_FILE = []
        self.PICKLE_MODEL_FILE = ""
        self.PICKLE_SCORE_FILE = ""

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
    
        #ファイルアドレスの設定
        self.PICKLE_MODEL_FILE = parameter_folder
        self.PICKLE_SCORE_FILE = parameter_folder

        #設定ファイルの読み込み
        setting_file_address = parameter_folder+"setting.txt"
        setting_file = open(setting_file_address,"r")
        setting_line = setting_file.readline()

        #フィルターの数
        self.filter_num = 1
        while setting_line:
            
            setting_item = setting_line.replace("\n","").split(" ")

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
            elif setting_item[0] == "ONE_TEST_SIZE":
                self.ONE_TEST_SIZE = int(setting_item[1])
            elif setting_item[0] == "NU":
                self.NU = float(setting_item[1])
            elif setting_item[0] == "GAMMA":
                self.GAMMA = float(setting_item[1])
            elif setting_item[0] == "EX_COEFF":
                self.EX_COEFF = float(setting_item[1])
            elif setting_item[0] == "STD_TYPE":
                self.STD_TYPE = setting_item[1].replace("\n","")
            elif setting_item[0] == "BRIGHTNESS_MATCH":
                self.BRIGHTNESS_MATCH = int(setting_item[1])
            if setting_item[0] == "<FILTER>":
                """FILTER設定を複数条件に対応 23/8/15"""
                """FILTER_LIST -> [[FILTER_LIST-1],[FILTER_LISTS-2],---]"""
                """FILTER_PARAM -> [[[PARAM-1.1],[PARAM-1.2]],[[PARAM-2.1],[PARAM-2.2]]]"""
                while setting_item[0] != "<END>":
                    if setting_item[0] == "FILTER_LIST": #フィルターとパラメータを引っ張る
                        tmp_filter = []
                        tmp_param = []
                        for item in setting_item[1].split(","):
                            if len(item.split("_"))==2:
                                tmp_filter.append(item.split("_")[0])
                                tmp_param.append([int(item.split("_")[1])])
                            elif len(item.split("_"))==3:
                                tmp_filter.append(item.split("_")[0])
                                tmp_param.append([int(item.split("_")[1]),int(item.split("_")[2])])
                            else:
                                print("setting.txtのFILTER PARAMETERの記述が間違っています")
                                sys.exit()
                        self.FILTER_LIST.append(tmp_filter)
                        self.FILTER_PARAM.append(tmp_param)
                    elif setting_item[0] == "SPLIT_DATA": #split_dataのファイル名
                        self.SPLIT_DATA_FILE.append(parameter_folder+setting_item[1])
                    elif setting_item[0] == "STD_COEFF":
                        self.STD_COEFF.append(float(setting_item[1]))
                    setting_line = setting_file.readline()
                    setting_item = setting_line.replace("\n","").split(" ")
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

        #settingファイルをアウトプットフォルダにコピーする
        setting_file_address = parameter_folder+"setting.txt"
        setting_file_address_to = result_folder + "setting.txt"

        #resultフォルダーを作成する
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
            shutil.copy(setting_file_address,setting_file_address_to)
        else:
            return 1

        return

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
            #チップの外形が明らかにおかしい場合処理を止める
            print("チップの外形がおかしいです")
            print("幅{}、高さ{}".format(w,h))
            return 0,0

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
        """複数フィルターに対応できるように変更 23/8/16"""
        images = [] #returnに使うリスト

        for i,filters in enumerate(self.FILTER_LIST):
            img_after_filter = image.copy()
            for j,filter_name in enumerate(filters):
                if filter_name == "BILATERAL":
                    img_after_filter = cv2.bilateralFilter(img_after_filter,d=self.FILTER_PARAM[i][j][0],sigmaColor=100,sigmaSpace=10)
                elif filter_name == "UNSHARP":
                    #鮮鋭化処理のカーネル
                    k=self.FILTER_PARAM[i][j][0]
                    unsharp_kernel = np.array([[-k/9,-k/9,-k/9],
                                            [-k/9,1+8*k/9,k/9],
                                            [-k/9,-k/9,-k/9]],np.float32)
                    img_after_filter = cv2.filter2D(img_after_filter,-1,unsharp_kernel).astype("uint8")
                elif filter_name == "ADJUST":
                    img_after_filter = self.adjust(img_after_filter,self.FILTER_PARAM[i][j][0],self.FILTER_PARAM[i][j][1])
                elif filter_name == "SOBEL":
                    grid_x = cv2.Sobel(img_after_filter,cv2.CV_32F,1,0,self.FILTER_PARAM[i][j][0])
                    grid_y = cv2.Sobel(img_after_filter,cv2.CV_32F,0,1,self.FILTER_PARAM[i][j][0])
                    img_after_filter = np.sqrt(grid_x**2+grid_y**2).astype("uint8")
                elif filter_name == "SIGMOID":
                    img_after_filter = 255/(1+np.exp(-self.FILTER_PARAM[i][j][0]*(img_after_filter-self.FILTER_PARAM[i][j][1])/255))
                    img_after_filter = img_after_filter.astype("uint8")
                else:
                    print("フィルターの指定が間違っています")
                    sys.exit()

            images.append(img_after_filter)

        return images

    #差分を取得する
    def get_diff_image_list(self,img_good_average,image,brightness_data,n,mode,img_path):
        '''FILTERが複数の場合に対応 23/8/16'''
        diff_image_list=[[] for i in range(len(self.FILTER_LIST))] #最終結果を入れるリスト

        a_b_list = self.get_line(image,self.rectangle_point)
        img_affine,theta= self.get_rotate_image(image,a_b_list)
        if img_affine==0:
            #チップの位置補正に失敗した場合はreturn 0する
            return 0

        img,x,y,w,h = self.get_chip_image(img_affine,0)

        #お手本との輝度合わせを実施する
        if self.BRIGHTNESS_MATCH==1:
            img = (img-np.mean(img))/np.std(img)*brightness_data[1]+brightness_data[0]
            img = np.clip(img,0,255) #8bitに調整
            img = img.astype(np.uint8)

        if mode==0:
            #テスト時
            self.theta_list.append([theta,x,y,w,h])
            img_name = self.OUTPUT_ALL_IMAGE+img_path.split("/")[-1].split(".")[0]+".jpg"
            cv2.imwrite(img_name,img)
            print("\r{}枚目 {}を処理中".format(n,img_path),end="")
        elif mode==1:
            #良品学習時
            print("\r{}枚目 {}を処理中".format(n,img_path),end="")

        #スプリットする前にフィルターをかける
        if len(self.FILTER_LIST) > 0:
            img_list = self.image_filter(img)

        #split_dataで画像を切り取っていく
        for i,split_data in enumerate(self.SPLIT_DATA):
            for j,s in enumerate(split_data):
                '''FILTERが複数の場合に対応 23/8/16'''
                x = s[0] #split_data 左上
                y = s[1] #split_data 右上
                lx = s[2] #split_data 幅
                ly = s[3] #split_data 高さ

                template = img_good_average[i][y-self.TEST_PADDING_Y:y+ly+self.TEST_PADDING_Y,x-self.TEST_PADDING_X:x+lx+self.TEST_PADDING_X]
                test_image = img_list[i][y:y+ly,x:x+lx]
                
                #テンプレートマッチング
                res = cv2.matchTemplate(template,test_image,cv2.TM_CCOEFF_NORMED)

                #最もマッチングしている部分を取り出し
                _,_,_,max_loc = cv2.minMaxLoc(res)
                tmp_image_match = template[max_loc[1]:max_loc[1]+ly,max_loc[0]:max_loc[0]+lx]

                img_diff = cv2.absdiff(test_image,tmp_image_match)

                #diff_image_list[i].append((img_diff/255).flatten()) #diff imageを0~1に正規化
                diff_image_list[i].append((img_diff).flatten()) #省メモリのために正規化を後で行う

        return diff_image_list
     
    def learn_good_feature(self,good_diff_image_list):
        """複数フィルターに対応できるように変更 23/08/16"""
        models=[[] for i in range(len(good_diff_image_list))]

        #good_diff_image_list フィルターの種類 X 分割の数 X 良品画像の数
        for i,features_per_filter in enumerate(good_diff_image_list): 
            for j,features in enumerate(features_per_filter):
                print("\r{}番目の分割の{}番目の特徴のモデル作成".format(i+1,j+1),end="")
                features = np.array(features)/255
                model = svm.OneClassSVM(nu=self.NU,gamma=self.GAMMA)
                model.fit(features)
                models[i].append(model)

        #pickleファイルにモデルを書き出し
        for i,model in enumerate(models):
            with open(self.PICKLE_MODEL_FILE+"model_"+str(i+1)+".pickle",mode="wb") as fo:
                pickle.dump(model,fo)

        return models

    #良品のスコアを計算
    def good_predict(self,models,good_diff_image_list):
        """複数フィルターに対応できるように変更 23/08/16"""
        results=[[] for i in range(len(good_diff_image_list))]

        #good_features -> フィルターの数 X 分割数 X 良品画像数
        for i,features_per_filter in enumerate(good_diff_image_list):
            for j,features in enumerate(features_per_filter):
                features = np.array(features)/255
                result = models[i][j].score_samples(features)
                results[i].append(result)

        #pickleファイルに良品スコアを書き出し
        for i,result in enumerate(results):
            with open(self.PICKLE_SCORE_FILE+"score_"+str(i+1)+".pickle",mode="wb") as fo:
                pickle.dump(result,fo)

        return results

    #テスト画像のスコアを計算
    def result_predict(self,models,test_diff_image_list,split_num):
        """複数フィルターに対応できるように変更 23/08/16"""
        results=[[] for i in range(len(test_diff_image_list))]

        #test_features -> フィルター数 X split_num X テスト数
        for i,features_per_filter in enumerate(test_diff_image_list):
            for j,features in enumerate(features_per_filter):
                features = np.array(features)/255
                result = models[i][j].score_samples(features)
                result_list = result.tolist()
                results[i].append(result_list)

        return results

    #良品学習スキップ時にpickleファイルを読み込む
    def load_model_pickle(self):
        """複数フィルターに対応できるように変更 23/08/16"""
        '''Filterの種類が何種類あるかをself.FILTER_LISTから取得'''
        '''そのあとpickleの読込を開始'''
        models = [[] for i in range(len(self.FILTER_LIST))]
        for i in range(len(self.FILTER_LIST)):
            filename = self.PICKLE_MODEL_FILE + "model_" + str(i+1)+".pickle"
            with open(filename,mode="rb") as fm:
                models[i] = pickle.load(fm)

        return models

    #良品学習スキップ時にpickleファイルを読み込む
    def load_score_pickle(self):
        """複数フィルターに対応できるように変更 23/08/16"""
        '''Filterの種類が何種類あるかをself.FILTER_LISTから取得'''
        '''そのあとpickleの読込を開始'''
        scores = [[] for i in range(len(self.FILTER_LIST))]
        for i in range(len(self.FILTER_LIST)):
            filename = self.PICKLE_SCORE_FILE + "score_"+str(i+1)+".pickle"
            with open(filename,mode="rb") as fs:
                scores[i] = pickle.load(fs)

        return scores

    def set_standards(self,test_scores):
        """複数フィルターに対応できるように変更 23/08/16"""
        standards = [[] for i in range(len(self.FILTER_LIST))]

        for i,test_scores_per_filter in enumerate(test_scores):
            for score in test_scores_per_filter:
                med = np.median(score) #良品学習スコアの内最小値を規格に置く
                std = np.std(score)
                standard = med-self.STD_COEFF[i]*std
                standards[i].append(standard)

        return standards

    def judge_pass_fail(self,standards,test_predictions,test_num,test_image_name):
        """複数フィルターに対応できるように変更 23/08/16"""
        #predictionsはフィルター数 X split_num X テストサンプル数

        judge_result=[]
        for i in range(len(test_predictions)):
            judge_result.append([[] for i in range(test_num)]) #フィルター数Xサンプル数のリストを作成

        #judge_result[フィルターの番号][画像番号] に 分割番号ごとに1か0を入れていく
        for i,predictions_per_filter in enumerate(test_predictions): #i フィルターの番号
            for j,predictions in enumerate(predictions_per_filter): #j 分割数
                for k,p in enumerate(predictions): # 画像数
                    if p < standards[i][j]:
                        judge_result[i][k].append("1")
                    else:
                        judge_result[i][k].append("0")

        #どの画像のどの部分が不良かを書き出す
        fail_image_list = [[] for i in range(test_num)] 
        for i,result_per_filter in enumerate(judge_result):
            for j,result in enumerate(result_per_filter):
                if "1" in result:
                    for k,r in enumerate(result):
                        if r=="1":
                            x1 = self.SPLIT_DATA[i][k][0] #矩形左上 X
                            y1 = self.SPLIT_DATA[i][k][1] #矩形右上 Y
                            x2 = x1+self.SPLIT_DATA[i][k][2] #矩形右下 X
                            y2 = y1+self.SPLIT_DATA[i][k][3] #矩形右下 Y
                            fail_image_list[j].append([i,x1,y1,x2,y2])
        
        #Fail画像に矩形を描画する
        #現状3種類までしか対応していない
        color_pallete = [(200,0,0),(0,200,0),(0,0,200)]

        for i,info in enumerate(fail_image_list):
            if len(info)>0:
                fail_img = cv2.imread(self.OUTPUT_ALL_IMAGE+test_image_name[i]+".jpg")
                #fail_img = cv2.cvtColor(fail_img,cv2.COLOR_GRAY2RGBA)
                for rd in info:
                    cv2.rectangle(fail_img,(rd[1],rd[2]),(rd[3],rd[4]),color_pallete[rd[0]],1)

                cv2.imwrite(self.OUTPUT_FAIL_IMAGE+test_image_name[i]+".jpg",fail_img)

        return judge_result

    #スプリットデータ読み込み
    def get_split_data(self):
        """split_dataが複数版に対応 23/8/16"""
        num = len(self.FILTER_LIST)  #split_fileの数 = FILTER_LISTの数
        self.SPLIT_DATA = [[] for i in range(num)]
        std_lower=[[] for i in range(num)] #規格がFIXの時の規格
        for i in range(num):
            #split_data読込
            filename = self.SPLIT_DATA_FILE[i]
            split_data_file = open(filename,"r")
            split_data_line = split_data_file.readline()
            while split_data_line:
                d = split_data_line.replace("\n","").split(",")
                self.SPLIT_DATA[i].append([int(d[0]),int(d[1]),int(d[2]),int(d[3])])

                if self.STD_TYPE=="FIX":
                    std_lower[i].append(float(d[4]))

                split_data_line = split_data_file.readline()

            split_data_file.close()

        if self.STD_TYPE=="FIX":
            return std_lower 
        else:
            return

"""メイン"""
if __name__=="__main__":
    print("no use main")