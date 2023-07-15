import tkinter as tk            # ウィンドウ作成用
from tkinter import ttk
from tkinter import filedialog  # ファイルを開くダイアログ用
from tkinter import messagebox
from PIL import Image, ImageTk  # 画像データ用
import numpy as np              # アフィン変換行列演算用
import os                       # ディレクトリ操作用
import cv2
import sys
from insp_module_rev3 import insp
import glob
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack() 
 
        self.my_title = "OpenCV Tkinter GUI Sample"  # タイトル
        self.back_color = "#FFFFFF"     # 背景色

        # ウィンドウの設定
        self.master.title(self.my_title)    # タイトル
        self.master.geometry("600x400")     # サイズ

        self.pil_image = None           # 表示する画像データ
        self.filename = None            # 最後に開いた画像ファイル名
 
        self.create_menu()   # メニューの作成
        self.create_widget() # ウィジェットの作成

    # -------------------------------------------------------------------------------
    # メニューイベント
    # -------------------------------------------------------------------------------
    def menu_open_clicked(self, event=None):
        # File → Open
        filename = tk.filedialog.askopenfilename(
            filetypes = [("Image file", ".bmp .png .jpg .tif"), ("Bitmap", ".bmp"), ("PNG", ".png"), ("JPEG", ".jpg"), ("Tiff", ".tif") ], # ファイルフィルタ
            initialdir = os.getcwd() # カレントディレクトリ
            )

        # 画像ファイルを設定する
        self.set_image(filename)

    def menu_reload_clicked(self, event=None):
        # File → ReLoad
        self.set_image(self.filename)

    def menu_quit_clicked(self):
        # ウィンドウを閉じる
        self.master.destroy() 

    def menu_save_clicked(self,event=None):
        # 画像を一つ戻す
        self.save() 

    def menu_open_rectfile_clicked(self,event=None):
        # 矩形情報ファイルを開く
        self.open_rectfile()

    def menu_save_rectfile_clicked(self,event=None):
        # 矩形情報ファイルを保存
        self.save_rectfile()

    # -------------------------------------------------------------------------------

    # create_menuメソッドを定義
    def create_menu(self):
        self.menu_bar = tk.Menu(self) # Menuクラスからmenu_barインスタンスを生成
 
        self.file_menu = tk.Menu(self.menu_bar, tearoff = tk.OFF)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)

        self.file_menu.add_command(label="Open", command = self.menu_open_clicked, accelerator="Ctrl+O")
        self.file_menu.add_command(label="ReLoad", command = self.menu_reload_clicked, accelerator="Ctrl+R")
        self.file_menu.add_command(label="Save", command = self.menu_save_clicked, accelerator="Ctrl+S")
        self.file_menu.add_separator() # セパレーターを追加
        self.file_menu.add_command(label="Open RectFile", command = self.menu_open_rectfile_clicked)
        self.file_menu.add_command(label="Save RectFile", command = self.menu_save_rectfile_clicked)
        self.file_menu.add_separator() # セパレーターを追加
        self.file_menu.add_command(label="Exit", command = self.menu_quit_clicked)

        self.menu_bar.bind_all("<Control-o>", self.menu_open_clicked) # ファイルを開くのショートカット(Ctrol-Oボタン)
        self.menu_bar.bind_all("<Control-r>", self.menu_reload_clicked) # ファイルを開くのショートカット(Ctrol-Rボタン)
        self.menu_bar.bind_all("<Control-s>", self.menu_save_clicked) # ファイルを開くのショートカット(Ctrol-Sボタン)
        self.master.config(menu=self.menu_bar) # メニューバーの配置
 
    def create_widget(self):
        '''ウィジェットの作成'''

        #####################################################
        # ステータスバー相当(親に追加)
        self.statusbar = tk.Frame(self.master)
        self.mouse_position = tk.Label(self.statusbar, relief = tk.SUNKEN, text="mouse position") # マウスの座標
        self.image_position = tk.Label(self.statusbar, relief = tk.SUNKEN, text="image position") # 画像の座標
        self.label_space = tk.Label(self.statusbar, relief = tk.SUNKEN)                           # 隙間を埋めるだけ
        self.image_info = tk.Label(self.statusbar, relief = tk.SUNKEN, text="image info")         # 画像情報
        self.mouse_position.pack(side=tk.LEFT)
        self.image_position.pack(side=tk.LEFT)
        self.label_space.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.image_info.pack(side=tk.RIGHT)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

        #----------------------------
        #notebook
        #----------------------------
        #メイン処理に関する項目を持ってくる
        style = ttk.Style()
        style.configure("TNotebook.Tab",width=10,font=("bold"))
        notebook = ttk.Notebook(self.master,style="TNotebook")
        notebook.configure(width=365)

        self.tab1 = ttk.Frame(notebook)
        notebook.add(self.tab1,text="メイン")

        self.tab2 = ttk.Frame(notebook)
        notebook.add(self.tab2,text="解析")

        self.tab3 = ttk.Frame(notebook)
        notebook.add(self.tab3,text="グラフ")

        #----------------------------
        #tab1 メインのwidget
        #----------------------------
        #機種名
        type_name_lbl = tk.Label(self.tab1, text = "機種名",font=("MSゴシック","15","bold"))
        self.TypeName = tk.StringVar() 
        self.TypeName = tk.StringVar()
        txt_type_name = tk.Entry(self.tab1,textvariable=self.TypeName)

        #ロットNo
        lot_name_lbl = tk.Label(self.tab1, text = "ロットNo",font=("MSゴシック","15","bold"))
        self.LotName = tk.StringVar()
        txt_lot_name = tk.Entry(self.tab1,textvariable=self.LotName)

        #処理type(ラジオボタン)
        process_type_lbl = tk.Label(self.tab1, text = "処理タイプ",font=("MSゴシック","15","bold"))
        self.type_radio_value = tk.IntVar(value=0)
        self.process_mode = 0

        self.type_radio0 = tk.Radiobutton(self.tab1,text="通常処理",font=("MSゴシック","15"),command=self.type_radio_click,variable=self.type_radio_value,value=0)
        self.type_radio1 = tk.Radiobutton(self.tab1,text="良品学習込みの全処理",font=("MSゴシック","15"),command=self.type_radio_click,variable=self.type_radio_value,value=1)
        self.type_radio2 = tk.Radiobutton(self.tab1,text="良品学習のみ",font=("MSゴシック","15"),command=self.type_radio_click,variable=self.type_radio_value,value=2)
        self.type_radio3 = tk.Radiobutton(self.tab1,text="PAT処理",font=("MSゴシック","15"),command=self.type_radio_click,variable=self.type_radio_value,value=3)

        #処理開始ボタン
        btn_Main = tk.Button(self.tab1, text = "処理開始", font=("MSゴシック","20","bold"),command = self.start_main_process)

        #widget設置
        type_name_lbl.place(x=10,y=15,height=30)
        txt_type_name.place(x=10,y=45,width=300,height=30)

        lot_name_lbl.place(x=10,y=95,height=30)
        txt_lot_name.place(x=10,y=125,width=300,height=30)

        process_type_lbl.place(x=10,y=175,height=30)

        self.type_radio0.place(x=15,y=205,height=20)
        self.type_radio1.place(x=15,y=225,height=20)
        self.type_radio2.place(x=15,y=245,height=20)
        self.type_radio3.place(x=15,y=265,height=20)

        btn_Main.place(x=10,y=500,height=30)

        #----------------------------
        #tab2 解析のwidget
        #----------------------------
        #表
        self.column = (0,1,2)
        self.tree=ttk.Treeview(self.tab2, columns=self.column,show="headings")
        self.tree.bind("<<TreeviewSelect>>",self.on_tree_select)

        self.tree.column(0,width=65,anchor="center")
        self.tree.column(1,width=180,anchor="center")
        self.tree.column(2,width=65,anchor="center")

        self.tree.heading(0,text="num")
        self.tree.heading(1,text="画像名")
        self.tree.heading(2,text="p/f")

        x_set = 10
        y_set = 10
        height = 450

        self.tree.place(x=x_set,y=y_set,height=height)

        vsb = ttk.Scrollbar(self.tab2,orient="vertical",command=self.tree.yview)
        vsb.place(x=x_set+325+3,y=y_set+3,height=height)
        self.tree["yscrollcommand"]=vsb.set

        self.id_list={}
        id_tmp=self.tree.insert("","end",values=(1,"test.jpg","p"))
        self.id_list[id_tmp]=[1,"test.jpg","p"]

        btn_view_fail = tk.Button(self.tab2, text = "Failのみ表示", width = 30, command = self.click_view_fail)
        btn_view_fail.place(x=x_set,y=y_set+height+10,height=30)

        btn_view_pass = tk.Button(self.tab2, text = "Passのみ表示", width = 30, command = self.click_view_pass)
        btn_view_pass.place(x=x_set,y=y_set+height+50,height=30)

        btn_view_all = tk.Button(self.tab2, text = "全て表示", width = 30, command = self.click_view_all)
        btn_view_all.place(x=x_set,y=y_set+height+90,height=30)

        #----------------------------
        #tab3 解析のwidget
        #----------------------------
        #表
        self.column = (0,1,2,3,4)
        self.graph_tree=ttk.Treeview(self.tab3, columns=self.column,show="headings")
        self.graph_tree.bind("<<TreeviewSelect>>",self.on_graph_tree_select)

        self.graph_tree.column(0,width=65,anchor="center")
        self.graph_tree.column(1,width=65,anchor="center")
        self.graph_tree.column(2,width=65,anchor="center")
        self.graph_tree.column(3,width=65,anchor="center")
        self.graph_tree.column(4,width=65,anchor="center")

        self.graph_tree.heading(0,text="分割番号")
        self.graph_tree.heading(1,text="規格")
        self.graph_tree.heading(2,text="med")
        self.graph_tree.heading(3,text="min")
        self.graph_tree.heading(4,text="max")

        x_set = 10
        y_set = 10
        height = 450

        self.graph_tree.place(x=x_set,y=y_set,height=height)

        vsb = ttk.Scrollbar(self.tab3,orient="vertical",command=self.graph_tree.yview)
        vsb.place(x=x_set+325+3,y=y_set+3,height=height)
        self.graph_tree["yscrollcommand"]=vsb.set

        #graph_treeの初期値
        self.graph_id_list={}
        id_tmp=self.graph_tree.insert("","end",values=(1,0.0,0.0,0.0,0.0))
        self.graph_id_list[id_tmp]=[1,0.0,0.0,0.0,0.0]

        btn_load_result = tk.Button(self.tab3, text = "resultファイルをロード", width = 30, command = self.click_load_result)
        btn_load_result.place(x=x_set,y=y_set+height+10,height=30)

        #notebookを配置
        notebook.pack(side = tk.RIGHT, fill = tk.Y)
        #####################################################
        # Canvas(画像の表示用)
        self.canvas = tk.Canvas(self.master, background= self.back_color)
        self.canvas.pack(expand=True,  fill=tk.BOTH)  # この両方でDock.Fillと同じ

        #####################################################

        #####################################################
        # マウスイベント
        self.canvas.bind("<Motion>", self.mouse_move)                       # MouseMove
        self.canvas.bind("<B1-Motion>", self.mouse_move_left)               # MouseMove（左ボタンを押しながら移動）
        self.canvas.bind("<Button-1>", self.mouse_down_left)                # MouseDown（左ボタン）
        self.canvas.bind("<Double-Button-1>", self.mouse_double_click_left) # MouseDoubleClick（左ボタン）
        self.canvas.bind("<MouseWheel>", self.mouse_wheel)                  # MouseWheel
        self.canvas.bind("<Button-3>", self.mouse_down_right)               # MouseDown (右ボタン)
        self.canvas.bind("<ButtonRelease-1>", self.left_click_release)      # 左クリックを離す
        self.rectangle=None

    def set_image(self, filename):
        ''' 画像ファイルを開く '''
        if not filename or filename is None:
            return

        # 画像ファイルの再読込用に保持
        self.filename = filename

        # PIL.Imageで開く
        self.pil_image = Image.open(filename)

        # PillowからNumPy(OpenCVの画像)へ変換
        self.cv_image = np.array(self.pil_image)
        
        #
        # カラー画像のときは、RGBからBGRへ変換する
        if self.cv_image.ndim == 3:
            self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_RGB2BGR)

        # 画像全体に表示するようにアフィン変換行列を設定
        self.zoom_fit(self.pil_image.width, self.pil_image.height)

        #canvas再作成
        self.remake_canvas()

        # 画像の表示
        self.draw_image(self.cv_image)

        # ウィンドウタイトルのファイル名を設定
        self.master.title(self.my_title + " - " + os.path.basename(filename))
        # ステータスバーに画像情報を表示する
        self.image_info["text"] = f"{self.pil_image.width} x {self.pil_image.height} {self.pil_image.mode}"
        # カレントディレクトリの設定
        os.chdir(os.path.dirname(filename))

        self.original_cv_image = self.cv_image.copy()
        self.original_cv_image = cv2.cvtColor(self.original_cv_image,cv2.COLOR_GRAY2RGB)

    #矩形描画用の座標テキストファイルをオープン
    def open_rectfile(self):

        filename = filedialog.askopenfilename(title="テキストファイルオープン",\
                filetypes=[("csv file",".csv"),("CSV",".csv")],\
                initialdir="./")

        rectFile = open(filename,"r")

        #現在の表の項目をすべて削除

        for key in self.id_list:
            self.tree.delete(key)
        
        self.id_list = dict()

        rect_line = rectFile.readline()
        n=0
        while rect_line:
            
            x = rect_line.split(",")[0]
            y = rect_line.split(",")[1]
            lx = rect_line.split(",")[2]
            ly = rect_line.split(",")[3]
            id_tmp=self.tree.insert("","end",values=(n,x,y,lx,ly))
            self.id_list[id_tmp]=[n,x,y,lx,ly]

            rect_line = rectFile.readline()
            n+=1
            
        rectFile.close()

    #矩形描画用の座標テキストファイルを保存
    def save_rectfile(self):

        filename = filedialog.asksaveasfilename(title="名前を付けて保存",\
                filetypes=[("CSV",".csv")],\
                initialdir="./",\
                defaultextension = "csv")

        output_rect_file = open(filename,"w")

        for key in self.id_list:
            output_line = "" 
            for i in self.id_list[key][1:]:
                output_line+=str(i)+","
            output_line+="\n"
            output_rect_file.write(output_line)

        output_rect_file.close()

        return 

    #画像保存
    def save(self):

        if self.cv_image is None:
            return

        filename = filedialog.asksaveasfilename(title="名前を付けて保存",\
                filetypes=[("JPEG",".jpg"),("PNG",".png")],\
                initialdir="./",\
                defaultextension = "jpg")

        cv2.imwrite(filename,self.cv_image)

    
    # -------------------------------------------------------------------------------
    # マウスイベント
    # -------------------------------------------------------------------------------

    def mouse_move(self, event):
        ''' マウスの移動時 '''
        # マウス座標
        self.mouse_position["text"] = f"mouse(x, y) = ({event.x: 4d}, {event.y: 4d})"
        
        if self.pil_image is None:
            return

        # 画像座標
        mouse_posi = np.array([event.x, event.y, 1]) # マウス座標(numpyのベクトル)
        mat_inv = np.linalg.inv(self.mat_affine)     # 逆行列（画像→Cancasの変換からCanvas→画像の変換へ）
        self.image_posi = np.dot(mat_inv, mouse_posi)     # 座標のアフィン変換
        x = int(np.floor(self.image_posi[0]))
        y = int(np.floor(self.image_posi[1]))
        if x >= 0 and x < self.pil_image.width and y >= 0 and y < self.pil_image.height:
            # 輝度値の取得
            value = self.pil_image.getpixel((x, y))
            self.image_position["text"] = f"image({x: 4d}, {y: 4d}) = {value}"
        else:
            self.image_position["text"] = "-------------------------"

    def mouse_move_left(self, event):
        ''' マウスの左ボタンをドラッグ '''
        if self.pil_image is None:
            return

        self.translate(event.x - self.__old_event.x, event.y - self.__old_event.y)
        self.redraw_image() # 再描画
        self.__old_event = event

    def mouse_down_left(self, event):
        ''' マウスの左ボタンを押した '''
        if event.state & 0x1:

            # 画像座標
            mouse_posi = np.array([event.x, event.y, 1]) # マウス座標(numpyのベクトル)
            mat_inv = np.linalg.inv(self.mat_affine)     # 逆行列（画像→Cancasの変換からCanvas→画像の変換へ）
            self.image_posi = np.dot(mat_inv, mouse_posi)     # 座標のアフィン変換
            x = int(np.floor(self.image_posi[0]))
            y = int(np.floor(self.image_posi[1]))
            if x >= 0 and x < self.pil_image.width and y >= 0 and y < self.pil_image.height:
                # 輝度値の取得
                #self.rx1 = x
                #self.ry1 = y
                self.rx1 = 10*(x//10)
                self.ry1 = 10*(y//10)
                self.rectangle = 1
        else:
            self.__old_event = event

    def mouse_double_click_left(self, event):
        ''' マウスの左ボタンをダブルクリック '''
        if self.pil_image is None:
            return
        self.zoom_fit(self.pil_image.width, self.pil_image.height)
        self.redraw_image() # 再描画

    def mouse_wheel(self, event):
        ''' マウスホイールを回した '''
        if self.pil_image is None:
            return

        if (event.delta < 0):
            # 上に回転の場合、縮小
            self.scale_at(0.8, event.x, event.y)
        else:
            # 下に回転の場合、拡大
            self.scale_at(1.25, event.x, event.y)
        
        self.redraw_image() # 再描画

    def left_click_release(self,event):
        ''' マウスの左クリックをリリース '''
        if self.pil_image is None:
            return
        if self.rectangle:
            self.rectangle=None

            # 画像座標
            #矩形描画
            if self.rectangle_mode == 0:
                cv2.rectangle(self.cv_image,(self.rx1,self.ry1),(self.rx1+20,self.ry1+80),(0,0,200),1)
                item = self.tree.insert("","end",values=(len(self.id_list),self.rx1,self.ry1,20,80))
                self.id_list[item]=[len(self.id_list),self.rx1,self.ry1,20,80]
                self.new_data.set("")
            elif self.rectangle_mode == 1:
                cv2.rectangle(self.cv_image,(self.rx1,self.ry1),(self.rx1+80,self.ry1+20),(0,0,200),1)
                item = self.tree.insert("","end",values=(len(self.id_list),self.rx1,self.ry1,80,20))
                self.id_list[item]=[len(self.id_list),self.rx1,self.ry1,80,20]
                self.new_data.set("")
            elif self.rectangle_mode == 2:
                cv2.rectangle(self.cv_image,(self.rx1,self.ry1),(self.rx1+40,self.ry1+40),(0,0,200),1)
                item = self.tree.insert("","end",values=(len(self.id_list),self.rx1,self.ry1,40,40))
                self.id_list[item]=[len(self.id_list),self.rx1,self.ry1,40,40]
                self.new_data.set("")
            elif self.rectangle_mode == 3:
                cv2.rectangle(self.cv_image,(self.rx1,self.ry1),(self.rx1+20,self.ry1+40),(0,0,200),1)
                item = self.tree.insert("","end",values=(len(self.id_list),self.rx1,self.ry1,20,40))
                self.id_list[item]=[len(self.id_list),self.rx1,self.ry1,20,40]
                self.new_data.set("")
            elif self.rectangle_mode == 4:
                cv2.rectangle(self.cv_image,(self.rx1,self.ry1),(self.rx1+40,self.ry1+20),(0,0,200),1)
                item = self.tree.insert("","end",values=(len(self.id_list),self.rx1,self.ry1,40,20))
                self.id_list[item]=[len(self.id_list),self.rx1,self.ry1,40,20]
                self.new_data.set("")

            self.redraw_image() # 再描画

    def mouse_down_right(self,event):
        ''' マウスの右ボタンをクリック '''
        return 0

    # -------------------------------------------------------------------------------
    # 画像表示用アフィン変換
    # -------------------------------------------------------------------------------

    def reset_transform(self):
        '''アフィン変換を初期化（スケール１、移動なし）に戻す'''
        self.mat_affine = np.eye(3) # 3x3の単位行列

    def translate(self, offset_x, offset_y):
        ''' 平行移動 '''
        mat = np.eye(3) # 3x3の単位行列
        mat[0, 2] = float(offset_x)
        mat[1, 2] = float(offset_y)

        self.mat_affine = np.dot(mat, self.mat_affine)

    def scale(self, scale:float):
        ''' 拡大縮小 '''
        mat = np.eye(3) # 単位行列
        mat[0, 0] = scale
        mat[1, 1] = scale

        self.mat_affine = np.dot(mat, self.mat_affine)

    def scale_at(self, scale:float, cx:float, cy:float):
        ''' 座標(cx, cy)を中心に拡大縮小 '''

        # 原点へ移動
        self.translate(-cx, -cy)
        # 拡大縮小
        self.scale(scale)
        # 元に戻す
        self.translate(cx, cy)

    def zoom_fit(self, image_width, image_height):
        '''画像をウィジェット全体に表示させる'''

        # キャンバスのサイズ
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if (image_width * image_height <= 0) or (canvas_width * canvas_height <= 0):
            return

        # アフィン変換の初期化
        self.reset_transform()

        scale = 1.0
        offsetx = 0.0
        offsety = 0.0

        if (canvas_width * image_height) > (image_width * canvas_height):
            # ウィジェットが横長（画像を縦に合わせる）
            scale = canvas_height / image_height
            # あまり部分の半分を中央に寄せる
            offsetx = (canvas_width - image_width * scale) / 2
        else:
            # ウィジェットが縦長（画像を横に合わせる）
            scale = canvas_width / image_width
            # あまり部分の半分を中央に寄せる
            offsety = (canvas_height - image_height * scale) / 2

        # 拡大縮小
        self.scale(scale)
        # あまり部分を中央に寄せる
        self.translate(offsetx, offsety)

    # -------------------------------------------------------------------------------
    # 描画
    # -------------------------------------------------------------------------------

    def remake_canvas(self):

        self.canvas.delete("all")

        ############
        self.canvas.destroy()
        ############
        self.canvas = tk.Canvas(self.master,background= self.back_color)
        self.canvas.pack(expand=True,  fill=tk.BOTH)  # この両方でDock.Fillと同じ
        #####################################################
        # マウスイベント
        self.canvas.bind("<Motion>", self.mouse_move)                       # MouseMove
        self.canvas.bind("<B1-Motion>", self.mouse_move_left)               # MouseMove（左ボタンを押しながら移動）
        self.canvas.bind("<Button-1>", self.mouse_down_left)                # MouseDown（左ボタン）
        self.canvas.bind("<Double-Button-1>", self.mouse_double_click_left) # MouseDoubleClick（左ボタン）
        self.canvas.bind("<MouseWheel>", self.mouse_wheel)                  # MouseWheel
        self.canvas.bind("<Button-3>", self.mouse_down_right)               # MouseDown (右ボタン)
        self.canvas.bind("<ButtonRelease-1>", self.left_click_release)      # 左クリックを離す
        #############
        
        #self.canvas.create_line(20,10,280,190,fill="Blue",width=5)

        return

    def draw_image(self, cv_image):
        
        if cv_image is None:
            return

        self.cv_image = cv_image

        # キャンバスのサイズ
        self.update()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # キャンバスから画像データへのアフィン変換行列を求める
        #（表示用アフィン変換行列の逆行列を求める）
        mat_inv = np.linalg.inv(self.mat_affine)

        # ndarray(OpenCV)からPillowへ変換
        # カラー画像のときは、BGRからRGBへ変換する
        if cv_image.ndim == 3:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        # NumPyからPillowへ変換
        self.pil_image = Image.fromarray(cv_image)

        # PILの画像データをアフィン変換する
        dst = self.pil_image.transform(
                    (canvas_width, canvas_height),  # 出力サイズ
                    Image.AFFINE,         # アフィン変換
                    tuple(mat_inv.flatten()),       # アフィン変換行列（出力→入力への変換行列）を一次元のタプルへ変換
                    Image.NEAREST,       # 補間方法、ニアレストネイバー 
                    fillcolor= self.back_color
                    )

        # 表示用画像を保持
        self.image = ImageTk.PhotoImage(image=dst)

        # 画像の描画
        self.canvas.create_image(
                0, 0,               # 画像表示位置(左上の座標)
                anchor='nw',        # アンカー、左上が原点
                image=self.image    # 表示画像データ
                )
       
    def redraw_image(self):
        ''' 画像の再描画 '''
        if self.cv_image is None:
            return
        self.remake_canvas()
        self.draw_image(self.cv_image)

    def type_radio_click(self):
        '''tab1のラジオボタンがクリックされたとき'''
        value = self.type_radio_value.get()
        self.process_mode = value

        return

    #---------------------------#
    #解析処理                   #
    #---------------------------#
    def result_analysis(self,result_file_address,lot_no):
        """resultファイルを読み込んで表を更新"""
        """フォルダ内の1枚目の画像を読み込む"""

        #現在のロットフォルダを記憶
        self.now_result_folder = "C:/workspace/chip_inspection/result/"+lot_no+"/"

        #現在の表の項目をすべて削除
        for key in self.id_list:
            self.tree.delete(key)

        self.id_list = dict()
        result_file = open(result_file_address,"r")

        result_line = result_file.readline()
        n = 0
        while result_line:

            if n >=3:
                result_data = result_line.split(",")

                img_name = result_data[0]
                pass_fail = result_data[1]
                id_tmp=self.tree.insert("","end",values=(n-2,img_name,pass_fail))
                self.id_list[id_tmp]=[n-2,img_name,pass_fail]

            result_line = result_file.readline()
            n+=1

        #元のリストを保持
        self.original_id_list = self.id_list
        self.now_id_list = self.id_list

        result_file.close()

    def on_graph_tree_select(self,event):
        """tab3の表wを選択したときの処理""" 
        """グラフを表示"""

        #canvasの初期化
        self.remake_canvas()

        # Canvas(画像の表示用)
        self.fig = Figure()

        for item in self.graph_tree.selection():
            item_text = self.graph_tree.item(item,"values")

        split_num = int(item_text[0])-1
        std = float(item_text[1])

        y = self.score_list[split_num]
        x = [ i+1 for i in range(len(y))]

        self.ax = self.fig.subplots()
 
        #####################################################
        # Canvas(画像の表示用)
        self.fig_canvas = FigureCanvasTkAgg(self.fig,self.canvas)
        self.toolbar = NavigationToolbar2Tk(self.fig_canvas)
        self.fig_canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True)  # この両方でDock.Fillと同じ

        self.ax.scatter(x,y)

        #規格線を引く
        self.ax.axhline(std,color="red",lw=2)

        self.fig_canvas.draw()
    
        return 
   
    def is_num(self,n):
        """文字列が小数かどうか判定する関数"""
        try:
            float(n)
            return True
        except ValueError:
            return False

    def click_load_result(self):
        """tab3のボタンを押したときの処理"""
        """resultファイルを読み込み"""

        self.graph_standards=[]

        filename = filedialog.askopenfilename(title="resultファイルオープン",\
                filetypes=[("csv file",".csv"),("CSV",".csv")],\
                initialdir="./")

        result_file = open(filename,"r")

        result_line = result_file.readline()

        flag = 0
        #resulファイルを読み込んで、規格とスコアをリストに収める
        while result_line:

            if "standard" in result_line:
                
                items = result_line.split(",")

                for item in items:
                    if self.is_num(item):
                        self.graph_standards.append(float(item))
                
                split_data_num = len(self.graph_standards)
                self.score_list=[[] for i in range(split_data_num)]
            
            if "no" in result_line:
                flag = 1
                result_line = result_file.readline()
        
            if flag==1:
                #reverseさせる
                items = result_line.split(",")[:-1]
                items = items[::-1]
                for i in range(split_data_num):
                    self.score_list[i].append(float(items[split_data_num-i-1]))

            result_line = result_file.readline()

        result_file.close()

        self.score_list = np.array(self.score_list)

        #表にデータを追加
        #現在の表の項目をすべて削除
        for key in self.graph_id_list:
            self.graph_tree.delete(key)

        self.graph_id_list = dict()

        for i in range(split_data_num):
            num = i+1
            std = self.graph_standards[i]
            med_value = np.median(self.score_list[i])
            max_value = np.amax(self.score_list[i])
            min_value = np.amin(self.score_list[i])
            
            id_tmp=self.graph_tree.insert("","end",values=(num,std,med_value,min_value,max_value))
            self.graph_id_list[id_tmp]=[num,std,med_value,min_value,max_value]

        return
        
    def on_tree_select(self,event):
        """tab2のグラフを選択したときの処理""" 
        for item in self.tree.selection():
            item_text = self.tree.item(item,"values")

        img_name = item_text[1] + ".jpg"

        if item_text[2] == "p":
            img_address = self.now_result_folder + "good_image/"+img_name
            self.set_image(img_address)
        elif item_text[2] == "f":
            img_address = self.now_result_folder + "fail_image/"+img_name
            self.set_image(img_address)

        return 
    
    def click_view_fail(self):
        """表に不良品のみ表示"""

        self.id_list = dict()

        #現在の表の項目をすべて削除
        for key in self.now_id_list:
            self.tree.delete(key)

        #不良品のみを取り出して表示:
        n=1
        for key in self.original_id_list.keys():

            if self.original_id_list[key][2] == "f":
                img_name = self.original_id_list[key][1]
                pass_fail = self.original_id_list[key][2]
                id_tmp=self.tree.insert("","end",values=(n,img_name,pass_fail))
                self.id_list[id_tmp]=[n,img_name,pass_fail]
                n=n+1

        self.now_id_list = self.id_list
        return 

    def click_view_pass(self):
        """表に良品のみ表示"""
        self.id_list = dict()

        #現在の表の項目をすべて削除
        for key in self.now_id_list:
            self.tree.delete(key)

        #不良品のみを取り出して表示:
        n=1
        for key in self.original_id_list.keys():

            if self.original_id_list[key][2] == "p":
                img_name = self.original_id_list[key][1]
                pass_fail = self.original_id_list[key][2]
                id_tmp=self.tree.insert("","end",values=(n,img_name,pass_fail))
                self.id_list[id_tmp]=[n,img_name,pass_fail]
                n=n+1

        self.now_id_list = self.id_list
        return 

    def click_view_all(self):
        """元に戻す"""
        self.id_list = dict()

        #現在の表の項目をすべて削除
        for key in self.now_id_list:
            self.tree.delete(key)

        #不良品のみを取り出して表示:
        n=1
        for key in self.original_id_list.keys():

            img_name = self.original_id_list[key][1]
            pass_fail = self.original_id_list[key][2]
            id_tmp=self.tree.insert("","end",values=(n,img_name,pass_fail))
            self.id_list[id_tmp]=[n,img_name,pass_fail]
            n=n+1

        self.now_id_list = self.id_list
        return 

    #---------------------------#
    #メイン処理                 #
    #---------------------------#
    def start_main_process(self):
        '''メイン処理'''
        lot_no = self.LotName.get()
        type_name = self.TypeName.get()

        #insp_moduleのinspクラス
        insp_test = insp()

        #各辺のポイントを取得するための矩形情報読み込み
        insp_test.read_setting_file(type_name,lot_no)

        #分割情報,規格情報を取得
        if insp_test.STD_TYPE == "FIX":
            split_data,standards = insp_test.get_split_data()
        else:
            split_data = insp_test.get_split_data()

        split_num = len(split_data)

        """
        良品学習実施
        """
        #初めにお手本画像を作成する:
        img_good_average = insp_test.get_parent_img(insp_test.PARENT_IMG_FILE)

        #お手本画像との輝度合わせ用にデータを取得
        mean_good = img_good_average.mean()
        std_good = img_good_average.std()

        brightness_data = [mean_good,std_good]

        #お手本にフィルターをかける
        img_good_average = insp_test.image_filter(img_good_average)

        #プロセスタイプによって分岐
        if self.process_mode ==1 or self.process_mode==2:
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
                    diff_image_list = insp_test.get_diff_image_list(img_good_average,image,split_data,brightness_data,i*insp_test.ONE_TEST_SIZE+j+1,1)

                    for s in range(split_num):
                        good_diff_image_list[s].append(diff_image_list[s])

               #メモリ開放
                del good_image_list

            #one-class-svmで良品学習
            print("良品学習中")
            models = insp_test.learn_good_feature(good_diff_image_list,split_num)

            #スコア判定規格設定
            print("\nスコア判定規格設定中")
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

        elif self.process_mode==0:
            #pickleファイルを読み込んで良品学習をスキップ
            print("モデルpickleファイルを読み込み")
            models = insp_test.load_model_pickle()
            print("スコアpickleファイルを読み込み")
            good_scores = insp_test.load_score_pickle()
        elif self.process_mode==3:
            #pickleファイルを読み込んで良品学習をスキップ
            print("モデルpickleファイルを読み込み")
            models = insp_test.load_model_pickle()

        if self.process_mode == 2:
            print("良品学習完了")
            print("正常終了")
            sys.exit()

        #規格設定
        if self.process_mode != 3 and insp_test.STD_TYPE != "FIX":
            print("規格設定中")
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

                diff_image_list = insp_test.get_diff_image_list(img_good_average,image,split_data,brightness_data,i*insp_test.ONE_TEST_SIZE+j+1,0)
                
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

        if self.process_mode == 3 and insp_test.STD_TYPE != "FIX":
            standards = insp_test.set_standards(predictions)

        #良品不良品判定
        judge_results = insp_test.judge_pass_fail(standards,predictions,split_data,test_num)

        #result出力用に転置する
        output_predictions = np.array(predictions).T

        #結果出力csvファイル
        output_file = open(insp_test.OUTPUT_RESULT_FILE,"w")
        #header出力
        output_line = "standards,,,,,,,"
        for i in range(split_num):
            output_line += ","
        for i in range(split_num):
            output_line += str(standards[i])+","
        output_line += "\n"
        output_file.write(output_line)
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
                output_line = str(j+1)+","+"f"+","+str(insp_test.theta_list[j][0])+","+str(insp_test.theta_list[j][1])+","+str(insp_test.theta_list[j][2])+","+str(insp_test.theta_list[j][3])+","+str(insp_test.theta_list[j][4])+","
                fail_num += 1
            else:
                output_line = str(j+1)+","+"p"+","+str(insp_test.theta_list[j][0])+","+str(insp_test.theta_list[j][1])+","+str(insp_test.theta_list[j][2])+","+str(insp_test.theta_list[j][3])+","+str(insp_test.theta_list[j][4])+","
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

        messagebox.showinfo("確認","処理が正常に終了しました")

        self.result_analysis(insp_test.OUTPUT_RESULT_FILE,lot_no)

        return

if __name__ == "__main__":
    root = tk.Tk()
    root.state("zoomed")
    app = Application(master=root)
    app.mainloop()
