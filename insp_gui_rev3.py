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
import subprocess
import gc
import datetime

"""VERSION 4.1"""
"""複数条件の処理に対応可能なように変更"""
"""VERSION 5.1"""
"""トレイデータ・外観結果ファイル読み込みモードに対応"""
"""VERSION 5.2"""
"""位置補正に失敗した場合に処理を続行できるように変更"""
VERSION_INFO = "5.2"
DATE_INFO = "2023/9/1"
MASSPRODUCTION_MODE = False
YIELD_STANDARD = 0.98

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack() 
 
        self.my_title = "Inspection By Oneclasssvm"  # タイトル
        self.back_color = "#FFFFFF"     # 背景色

        # ウィンドウの設定
        self.master.title(self.my_title)    # タイトル
        self.master.geometry("600x400")     # サイズ

        self.pil_image = None           # 表示する画像データ
        self.filename = None            # 最後に開いた画像ファイル名
        self.split_info = False         #split data情報があるかどうか
        self.split_info_list = []
 
        self.create_menu()   # メニューの作成
        self.create_widget() # ウィジェットの作成

        #iniファイルの読み込み
        try:
            ini_file = open("./setting.ini","r")
            ini_line = ini_file.readline()
            while ini_line:
                if "TEST_SAMPLE_ADDRESS" in ini_line:
                    self.test_sample_address = ini_line.split(",")[1].replace("\n","")
                elif "PARAMETER_ADDRESS" in ini_line:
                    self.parameter_address = ini_line.split(",")[1].replace("\n","")
                elif "SURF_RESULT_ADDRESS" in ini_line:
                    self.surf_resultfile_address = ini_line.split(",")[1].replace("\n","")
                elif "TRAYDATA_ADDRESS" in ini_line:
                    self.traydatafile_address = ini_line.split(",")[1].replace("\n","")
                elif "OUTPUT_RESULT_ADDRESS" in ini_line:
                    self.output_result_address = ini_line.split(",")[1].replace("\n","")
                
                ini_line = ini_file.readline()
            
            ini_file.close()

        except:
            print("setting.iniがありません")

    # -------------------------------------------------------------------------------
    # メニューイベント
    # -------------------------------------------------------------------------------
    def menu_quit_clicked(self):
        # ウィンドウを閉じる
        self.master.destroy() 

    def menu_save_clicked(self,event=None):
        # 画像を一つ戻す
        self.save() 

    def menu_open_resultfolder_clicked(self,event=None):
        # 検査結果を開く
        self.open_resultfolder()

    def menu_setting_clicked(self, event=None):
        # フォルダ設定を開く
        self.set_setting()

    def menu_version_clicked(self, event=None):
        # バージョン情報を確認
        self.open_version()

    def menu_open_makerecipe(self, event=None):
        # 別ソフトを起動
        self.open_makerecipe()

    def menu_version_coeffsetting(self, event=None):
        # 別ソフトを起動:
        self.open_coeffsetting()
    # -------------------------------------------------------------------------------

    # create_menuメソッドを定義
    def create_menu(self):
        self.menu_bar = tk.Menu(self) # Menuクラスからmenu_barインスタンスを生成
 
        self.file_menu = tk.Menu(self.menu_bar, tearoff = tk.OFF)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)

        self.file_menu.add_command(label="Save image", command = self.menu_save_clicked, accelerator="Ctrl+S")
        self.file_menu.add_separator() # セパレーターを追加
        self.file_menu.add_command(label="Open Resultfolder", command = self.menu_open_resultfolder_clicked)
        self.file_menu.add_separator() # セパレーターを追加
        self.file_menu.add_command(label="Exit", command = self.menu_quit_clicked)

        self.menu_bar.bind_all("<Control-s>", self.menu_save_clicked) # ファイルを開くのショートカット(Ctrol-Sボタン)

        self.setting_menu = tk.Menu(self.menu_bar, tearoff = tk.OFF)
        self.menu_bar.add_cascade(label="Setting", menu=self.setting_menu)
        self.setting_menu.add_command(label="アドレス設定", command = self.menu_setting_clicked)

        self.version_menu = tk.Menu(self.menu_bar, tearoff = tk.OFF)
        self.menu_bar.add_cascade(label="ソフト", menu=self.version_menu)
        self.version_menu.add_command(label="MakeRecipe", command = self.menu_open_makerecipe)
        self.version_menu.add_command(label="CoeffSeting", command = self.menu_open_coeffsetting)

        self.version_menu = tk.Menu(self.menu_bar, tearoff = tk.OFF)
        self.menu_bar.add_cascade(label="Version", menu=self.version_menu)
        self.version_menu.add_command(label="バージョン情報", command = self.menu_version_clicked)

        self.master.config(menu=self.menu_bar) # メニューバーの配置
 
    def create_widget(self):
        '''ウィジェットの作成'''

        #####################################################
        # ステータスバー相当(親に追加)
        self.statusbar = tk.Frame(self.master)
        self.mouse_position = tk.Label(self.statusbar, relief = tk.SUNKEN, text="mouse position") # マウスの座標
        self.image_position = tk.Label(self.statusbar, relief = tk.SUNKEN, text="image position") # 画像の座標
        self.split_position = tk.Label(self.statusbar, relief = tk.SUNKEN, text="split position") # 画像の座標
        self.label_space = tk.Label(self.statusbar, relief = tk.SUNKEN)                           # 隙間を埋めるだけ
        self.image_info = tk.Label(self.statusbar, relief = tk.SUNKEN, text="image info")         # 画像情報
        self.mouse_position.pack(side=tk.LEFT)
        self.image_position.pack(side=tk.LEFT)
        self.split_position.pack(sid=tk.LEFT)
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

        self.log_text = tk.Text(self.tab1)
        scrollbar = ttk.Scrollbar(self.tab1,orient="vertical",command=self.log_text.yview)
        self.log_text["yscrollcommand"] = scrollbar.set

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
        self.log_text.place(x=15,y=300,width=300,height=200)
        scrollbar.place(x=315,y=300,height=200)
        btn_Main.place(x=10,y=530,height=30)

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

        btn_clip_copy = tk.Button(self.tab2, text = "選択した項目をクリップボードにコピー", width = 30, command = self.click_clip_copy)
        btn_clip_copy.place(x=x_set,y=y_set+height+160,height=30)

        #----------------------------
        #tab3 解析のwidget
        #----------------------------
        #表
        self.graph_column = (0,1,2,3,4)
        self.graph_tree=ttk.Treeview(self.tab3, columns=self.graph_column,show="headings")
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

    def set_setting(self):
        """モーダルダイアログボックスを開く"""
        self.dlg_setting = tk.Toplevel(self)
        self.dlg_setting.title("アドレスを設定")
        self.dlg_setting.geometry("650x600")

        self.dlg_setting.grab_set()
        self.dlg_setting.focus_set()
        self.dlg_setting.transient(self.master)

        dlg_test_address_label = tk.Label(self.dlg_setting,text="検査画像フォルダ",font=("MSゴシック","15"))
        dlg_test_address_label.pack()
        dlg_test_address_label.place(x=10,y=10,height=30)
        self.test_address_name = tk.StringVar()
        self.test_address = tk.Entry(self.dlg_setting,textvariable=self.test_address_name)
        self.test_address.place(x=10,y=45,width=500,height=30)
        self.test_address.insert(0,self.test_sample_address)
        #参照・チェックボタンを設置
        btn_ref_1 = tk.Button(self.dlg_setting, text = "参照", font=("MSゴシック","15"),command = lambda: self.dlg_setting_ref(1))
        btn_ref_1.place(x=520,y=45,height=30)
        btn_chk_1 = tk.Button(self.dlg_setting, text = "確認", font=("MSゴシック","15"),command = lambda: self.dlg_setting_chk(1))
        btn_chk_1.place(x=580,y=45,height=30)

        dlg_param_address_label = tk.Label(self.dlg_setting,text="検査パラメーターフォルダ",font=("MSゴシック","15"))
        dlg_param_address_label.pack()
        dlg_param_address_label.place(x=10,y=80,height=30)
        self.param_address_name = tk.StringVar()
        self.param_address = tk.Entry(self.dlg_setting,textvariable=self.param_address_name)
        self.param_address.place(x=10,y=115,width=500,height=30)
        self.param_address.insert(0,self.parameter_address)
        #参照・チェックボタンを設置
        btn_ref_2 = tk.Button(self.dlg_setting, text = "参照", font=("MSゴシック","15"),command = lambda: self.dlg_setting_ref(2))
        btn_ref_2.place(x=520,y=115,height=30)
        btn_chk_2 = tk.Button(self.dlg_setting, text = "確認", font=("MSゴシック","15"),command = lambda: self.dlg_setting_chk(2))
        btn_chk_2.place(x=580,y=115,height=30)

        dlg_surf_result_address_label = tk.Label(self.dlg_setting,text="外観検査ResultFileフォルダ",font=("MSゴシック","15"))
        dlg_surf_result_address_label.pack()
        dlg_surf_result_address_label.place(x=10,y=150,height=30)
        self.surf_result_address_name = tk.StringVar()
        self.surf_result_address = tk.Entry(self.dlg_setting,textvariable=self.surf_result_address_name)
        self.surf_result_address.place(x=10,y=185,width=500,height=30)
        self.surf_result_address.insert(0,self.surf_resultfile_address)
        #参照・チェックボタンを設置
        btn_ref_3 = tk.Button(self.dlg_setting, text = "参照", font=("MSゴシック","15"),command = lambda: self.dlg_setting_ref(3))
        btn_ref_3.place(x=520,y=185,height=30)
        btn_chk_3 = tk.Button(self.dlg_setting, text = "確認", font=("MSゴシック","15"),command = lambda: self.dlg_setting_chk(3))
        btn_chk_3.place(x=580,y=185,height=30)

        dlg_traydata_address_label = tk.Label(self.dlg_setting,text="トレイデータフォルダ",font=("MSゴシック","15"))
        dlg_traydata_address_label.pack()
        dlg_traydata_address_label.place(x=10,y=230,height=30)
        self.traydata_address_name = tk.StringVar()
        self.traydata_address = tk.Entry(self.dlg_setting,textvariable=self.traydata_address_name)
        self.traydata_address.place(x=10,y=265,width=500,height=30)
        self.traydata_address.insert(0,self.traydatafile_address)
        #参照・チェックボタンを設置
        btn_ref_4 = tk.Button(self.dlg_setting, text = "参照", font=("MSゴシック","15"),command = lambda: self.dlg_setting_ref(4))
        btn_ref_4.place(x=520,y=265,height=30)
        btn_chk_4 = tk.Button(self.dlg_setting, text = "確認", font=("MSゴシック","15"),command = lambda: self.dlg_setting_chk(4))
        btn_chk_4.place(x=580,y=265,height=30)

        dlg_result_address_label = tk.Label(self.dlg_setting,text="検査結果出力フォルダ",font=("MSゴシック","15"))
        dlg_result_address_label.pack()
        dlg_result_address_label.place(x=10,y=310,height=30)
        self.result_address_name = tk.StringVar()
        self.result_address = tk.Entry(self.dlg_setting,textvariable=self.result_address_name)
        self.result_address.place(x=10,y=345,width=500,height=30)
        self.result_address.insert(0,self.output_result_address)
        #参照・チェックボタンを設置
        btn_ref_5 = tk.Button(self.dlg_setting, text = "参照", font=("MSゴシック","15"),command = lambda: self.dlg_setting_ref(5))
        btn_ref_5.place(x=520,y=345,height=30)
        btn_chk_5 = tk.Button(self.dlg_setting, text = "確認", font=("MSゴシック","15"),command = lambda: self.dlg_setting_chk(5))
        btn_chk_5.place(x=580,y=345,height=30)

        #注釈
        annotation_label_1 = tk.Label(self.dlg_setting,text="※アドレスのロットNo部分は %LOT% で指定",font=("MSゴシック","10"))
        annotation_label_2 = tk.Label(self.dlg_setting,text="※アドレスの品名部分は %PRODUCT% で指定",font=("MSゴシック","10"))
        annotation_label_1.pack()
        annotation_label_1.place(x=10,y=400,height=20)
        annotation_label_2.pack()
        annotation_label_2.place(x=10,y=420,height=20)

        #閉じるボタンを設置
        btn_close = tk.Button(self.dlg_setting, text = "閉じる", font=("MSゴシック","15"),command = self.dlg_setting_close)
        btn_close.place(x=300,y=470,height=30)

        app.wait_window(self.dlg_setting)

    #参照ボタンを押したときの処理
    def dlg_setting_ref(self,n):

        address = filedialog.askdirectory(title="結果フォルダオープン",initialdir="./")

        if n==1:
            self.test_address.delete(0,tk.END)
            self.test_address.insert(0,address)
        elif n==2:
            self.param_address.delete(0,tk.END)
            self.param_address.insert(0,address)
        elif n==3:
            self.surf_result_address.delete(0,tk.END)
            self.surf_result_address.insert(0,address)
        elif n==4:
            self.traydata_address.delete(0,tk.END)
            self.traydata_address.insert(0,address)
        elif n==5:
            self.result_address.delete(0,tk.END)
            self.result_address.insert(0,address)

        return

    #確認ボタンを押したときの処理
    def dlg_setting_chk(self,n):
        if n==1:
            address = self.test_address_name.get()
        elif n==2:
            address = self.param_address_name.get()
        elif n==3:
            address = self.surf_result_address_name.get()
        elif n==4:
            address = self.traydata_address_name.get()
        elif n==5:
            address = self.result_address_name.get()
        subprocess.Popen(["start",address],shell=True)
        return
    
    #settingウインドウを閉じたときの処理
    def dlg_setting_close(self):
        #更新するかどうかのポップアップを出す

        ret = messagebox.askyesno("確認","アドレスを更新しますか?")

        if ret==True:
            #iniファイルを更新する
            ini_file = open("./setting.ini","w")
            output_line = "TEST_SAMPLE_ADDRESS,"+self.test_address_name.get()+"\n"
            self.test_sample_address = self.test_address_name.get()
            ini_file.write(output_line)
            output_line = "PARAMETER_ADDRESS,"+self.param_address_name.get()+"\n"
            self.parameter_address = self.param_address_name.get()
            ini_file.write(output_line)
            output_line = "SURF_RESULT_ADDRESS,"+self.surf_result_address_name.get()+"\n"
            self.surf_resultfile_address = self.surf_result_address_name.get()
            ini_file.write(output_line)
            output_line = "TRAYDATA_ADDRESS,"+self.traydata_address_name.get()+"\n"
            self.traydatafile_address = self.traydata_address_name.get()
            ini_file.write(output_line)
            output_line = "OUTPUT_RESULT_ADDRESS,"+self.result_address_name.get()+"\n"
            self.output_result_address = self.result_address_name.get()
            ini_file.write(output_line)

            ini_file.close()

            print("iniファイルを更新しました")
            self.dlg_setting.destroy() 
        else:
            self.dlg_setting.destroy() 

        return

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
        if self.original_cv_image.ndim == 1:
            self.original_cv_image = cv2.cvtColor(self.original_cv_image,cv2.COLOR_GRAY2RGB)

    #レシピ設定用プログラムを開く
    def menu_open_makerecipe(self):
        """別プログラム起動"""
        command = ["py", "c:\\workspace\\git_local\\make_recipe_program\\MakeRecipe.py"]
        subprocess.Popen(command)
        return

    #Coeff設定用プログラムを開く
    def menu_open_coeffsetting(self):
        """別プログラム起動"""
        command = ["py", "c:\\workspace\\git_local\\make_recipe_program\\CoeffSetting.py"]
        subprocess.Popen(command)
        return

    #version情報を開く
    def open_version(self):
        self.dlg_version = tk.Toplevel(self.master)
        self.dlg_version.title("バージョン情報")
        self.dlg_version.geometry("300x150")

        self.dlg_version.grab_set()
        self.dlg_version.focus_set()
        self.dlg_version.transient(self.master)

        dlg_test_address_label = tk.Label(self.dlg_version,text="バージョン："+VERSION_INFO,font=("MSゴシック","15"))
        dlg_test_address_label.pack()
        dlg_test_address_label.place(x=60,y=30,height=20)

        dlg_test_address_label = tk.Label(self.dlg_version,text="作成日："+DATE_INFO,font=("MSゴシック","15"))
        dlg_test_address_label.pack()
        dlg_test_address_label.place(x=60,y=60,height=20)

        #閉じるボタンを設置
        btn_close = tk.Button(self.dlg_version, text = "閉じる", font=("MSゴシック","15"),command = self.dlg_version_close)
        btn_close.place(x=100,y=100,height=30)

        app.wait_window(self.dlg_version)
    
    #dlgを閉じる関数
    def dlg_version_close(self):

        self.dlg_version.destroy() 

    #検査結果がフォルダを開いてtab2とtab3のtreeに情報を表示
    def open_resultfolder(self):

        folder_address = filedialog.askdirectory(title="結果フォルダオープン",initialdir="./")

        """textboxへロットNo書き込み"""
        lot_no = folder_address.split("/")[-1]
        self.log_text.delete("1.0","end")
        self.log_text.insert(tk.END,"ロットNo：{}".format(lot_no))

        """tab2 画像用treeの更新"""
        self.now_result_folder = folder_address + "/"
        csv_files = glob.glob(folder_address+"/*.csv")
        result_file_address = ""
        for csv_file in csv_files:
            if "result_data" in csv_file:
                result_file_address = csv_file
                continue

        if result_file_address == "":
            messagebox.showinfo("確認","resultファイルが見つかりません")
            return

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

        """"""""""""""""""""""""""
        """tab3 データtreeの更新"""
        """"""""""""""""""""""""""
        self.graph_standards=[]
        self.filter_split=[]
        result_file = open(result_file_address,"r")
        result_line = result_file.readline()
        flag = 0
        #resultファイルを読み込んで、規格とスコアをリストに収める
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
                items = result_line.replace("\n","").split(",")
                for n,item in enumerate(items):
                    if n>6: #n==5がh
                        self.filter_split.append(item)

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
            num = self.filter_split[i]
            std = self.graph_standards[i]
            med_value = np.median(self.score_list[i])
            max_value = np.amax(self.score_list[i])
            min_value = np.amin(self.score_list[i])
            
            id_tmp=self.graph_tree.insert("","end",values=(num,std,med_value,min_value,max_value))
            self.graph_id_list[id_tmp]=[num,std,med_value,min_value,max_value]

        messagebox.showinfo("確認","読込完了!")

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

            if self.split_info == True:
                
                split_num_list = []
                for s1,sd_per_filter in enumerate(self.split_info_list):
                    for s2,sd in enumerate(sd_per_filter):
                        sx = int(sd[0])
                        sy = int(sd[1])
                        lsx = int(sd[2])
                        lsy = int(sd[3])

                        if (x>sx and x<=sx+lsx) and (y>sy and y<=sy+lsy):
                            split_num_list.append(str(s1+1)+"_"+str(s2+1))
                self.split_position["text"] = ",".join(split_num_list)
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
        #self.remake_canvas()
        self.draw_image(self.cv_image)

    def type_radio_click(self):
        '''tab1のラジオボタンがクリックされたとき'''
        value = self.type_radio_value.get()
        self.process_mode = value

        return

    #---------------------------#
    #解析処理                   #
    #---------------------------#
    def result_analysis(self,type_name,lot_no):
        """resultファイルを読み込んでtab2とtab3の表を更新"""
        """フォルダ内の1枚目の画像を読み込む"""

        #現在のロットフォルダを記憶
        self.now_result_folder = self.output_result_address.replace("%LOT%",lot_no).replace("%PRODUCT%",type_name) #結果ファイル置き場

        result_file_address = self.now_result_folder + "result_data_"+lot_no+".csv"
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

        """"""""""""""""""""""""""
        """tab3 データtreeの更新"""
        """"""""""""""""""""""""""
        self.graph_standards=[]
        self.filter_split=[]
        result_file = open(result_file_address,"r")
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
                items = result_line.replace("\n","").split(",")
                for n,item in enumerate(items):
                    if n>6: #n==5がh
                        self.filter_split.append(item)

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
            num = self.filter_split[i]
            std = self.graph_standards[i]
            med_value = np.median(self.score_list[i])
            max_value = np.amax(self.score_list[i])
            min_value = np.amin(self.score_list[i])
            
            id_tmp=self.graph_tree.insert("","end",values=(num,std,med_value,min_value,max_value))
            self.graph_id_list[id_tmp]=[num,std,med_value,min_value,max_value]

        return

    def on_graph_tree_select(self,event):
        """tab3の表を選択したときの処理""" 
        """グラフを表示"""

        #canvasの初期化
        self.remake_canvas()

        # Canvas(画像の表示用)
        self.fig = Figure()

        for item in self.graph_tree.selection():
            item_text = self.graph_tree.item(item,"values")

        '''item_textの何要素目かを出す必要がある 23/8/18'''
        split_num = self.filter_split.index(item_text[0])
        std = float(item_text[1])

        y = self.score_list[split_num]
        x = [ i+1 for i in range(len(y))]

        self.ax = self.fig.subplots()
 
        #####################################################
        # Canvas(画像の表示用)
        self.fig_canvas = FigureCanvasTkAgg(self.fig,self.canvas)
        self.toolbar = NavigationToolbar2Tk(self.fig_canvas,self.canvas)
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

        #再canvasの再表示
        #self.remake_canvas()
        for item in self.tree.selection():
            item_text = self.tree.item(item,"values")

        img_name = item_text[1] + ".jpg"

        if item_text[2] == "p":
            img_address = self.now_result_folder + "all_image/"+img_name
            self.set_image(img_address)
        elif item_text[2] == "f":
            img_address = self.now_result_folder + "fail_image/"+img_name
            self.set_image(img_address)
        elif item_text[2] == "n":
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
    
    def click_clip_copy(self):
        """クリップボードに選択した項目をコピー"""
        selected_items = self.tree.selection()
        #クリップボードにコピーする文字列を取得
        input_line = ""
        for item in selected_items:
            input_line += self.now_id_list[item][1]+"\n"

        #クリップボードにコピー
        subprocess.run("clip",input=input_line,text=True)

        return

    def create_popup(self,traydata_list,serial_list,image_list,pass_fail_list,lot_no):
        """低歩留まり時に結果書き換え用のウインドウをオープン"""
        """結果書き換え⇒データ出力"""

        self.lot_no = lot_no
        self.traydata = traydata_list
        self.serial_list = serial_list
        self.pass_fail_list = pass_fail_list
        #window作成
        self.check_window = tk.Toplevel(self.master)
        self.check_window.title("Pass/Fail書き換え")
        self.check_window.geometry("680x450")
        self.check_window.protocol("WM_DELETE_WINDOW",(lambda: "pass")())
        self.check_window.focus_set()
        self.check_window.transient(self.master)

        #ラベル表示
        check_label = tk.Label(self.check_window,text="Fail画像を確認して結果の書き換えを実施してください",font=("MSゴシック","9"))
        check_label.pack()
        check_label.place(x=10,y=10,height=20)

        #表を作成
        column = (0,1,2,3,4,5)
        self.result_tree=ttk.Treeview(self.check_window, columns=column,show="headings")
        self.result_tree.bind("<<TreeviewSelect>>",self.on_result_tree_select)
        self.result_tree.column(0,width=65,anchor="center")
        self.result_tree.column(1,width=65,anchor="center")
        self.result_tree.column(2,width=200,anchor="center")
        self.result_tree.column(3,width=150,anchor="center")
        self.result_tree.column(4,width=65,anchor="center")
        self.result_tree.column(5,width=65,anchor="center")
        self.result_tree.heading(0,text="P/F")
        self.result_tree.heading(1,text="シリアル")
        self.result_tree.heading(2,text="画像名")
        self.result_tree.heading(3,text="トレイ番号")
        self.result_tree.heading(4,text="トレイX")
        self.result_tree.heading(5,text="トレイY")

        x_set = 10
        y_set = 10
        height = 350
        self.result_tree.place(x=x_set,y=y_set,height=height)
        #スクロールバー
        vsb = ttk.Scrollbar(self.check_window,orient="vertical",command=self.result_tree.yview)
        vsb.place(x=x_set+610,y=y_set,height=height)
        self.result_tree["yscrollcommand"]=vsb.set
        #最後に確認ボタンを設置
        btn_complete = tk.Button(self.check_window, text = "目視確認完了",font=("bold"),foreground="blue",command = self.click_complete)
        btn_complete.place(x=250,y=y_set+height+20,height=30)

        #pass->fail, fail->passの変更履歴をログに残す
        self.change_log=[]
        #result_treeに記載する項目を作成
        #P/F,serial,image_name,trayname,trayx,trayy
        self.result_id_list = dict()
        for i in range(len(serial_list)):
            pf = pass_fail_list[i]
            serial = serial_list[i]
            img_address = image_list[i]

            #traydata_listにおける目的serialのインデックスを出す
            index_num = [x[0] for x in traydata_list].index(serial)
            tray_name = traydata_list[index_num][1]
            tray_x = traydata_list[index_num][2]
            tray_y = traydata_list[index_num][3]

            id_tmp = self.result_tree.insert("","end",values=(pf,serial,img_address,tray_name,tray_x,tray_y))
            self.result_id_list[id_tmp] = [pf,serial,img_address,tray_name,tray_x,tray_y]

    def on_result_tree_select(self,event):
        #pass/failの書き換え確認
        #複数項目が来た場合はリターンする
        if len(self.result_tree.selection())>1:
            messagebox.showinfo("確認","1つの項目だけ選択してください")
            return
        
        item_text = self.result_tree.item(self.result_tree.selection(),"values")
        #エラー回避用
        if item_text == "":
            return
        item_id = self.result_tree.focus()
        if item_text[0] == "Fail":
            tmp_list = self.result_id_list.copy()
            tmp_list[item_id][0] = "Pass"
            self.change_log.append([item_text[1],"Pass to Fail"])
            ret = messagebox.askyesno("確認","Fail->Passへの変更を実施しますか？")

            if ret:
                #pass_fail_listの書き換え
                serial = item_text[1]
                self.pass_fail_list[self.serial_list.index(serial)] = "Pass"

        elif item_text[0] == "Pass": 
            tmp_list = self.result_id_list.copy()
            tmp_list[item_id][0] = "Fail"
            self.change_log.append([item_text[1],"Fail to Pass"])
            ret = messagebox.askyesno("確認","Pass->Failへの変更を実施しますか？")
            if ret:
                #pass_fail_listの書き換え
                serial = item_text[1]
                self.pass_fail_list[self.serial_list.index(serial)] = "Fail"

        if ret==True:
            #treeの更新
            #まず全て削除
            for key in self.result_id_list:
                self.result_tree.delete(key)

            self.result_id_list = dict()

            for key in tmp_list:
                id_tmp = self.result_tree.insert("","end",values=tuple(tmp_list[key]))
                self.result_id_list[id_tmp] = tmp_list[key]
            
            print("変更完了!")

            return
        else:
            return

    def click_complete(self):
        #目視確認が確認したことを確認
        ret = messagebox.askyesno("確認","目視確認を完了しますか？")

        if ret==True:
            #パス・フェイル書き換えログ出力
            print("Logファイルを出力します")
            outputfilename=self.output_result_address.replace("%LOT%",self.lot_no)+"change_log.txt"
            outputfile = open(outputfilename,"w")

            if len(self.change_log)==0:
                outputfile.write("no change log")
            for log in self.change_log:
                for log in self.change_log:
                    outputline = ",".join(log)+"\n"
                    outputfile.write(outputline)
            outputfile.close()

            #トレイデータ出力
            print("トレイデータを出力します")
            self.output_traydata()
            print("出力完了")

            self.check_window.destroy()
            return
        else:
            return

    def output_traydata(self):
        #トレイデータ出力
        if not os.path.exists(self.output_result_address.replace("%LOT%",self.lot_no)+"output_traydata/"):
            os.makedirs(self.output_result_address.replace("%LOT%",self.lot_no)+"output_traydata/")
        
        traydata_list = traydata_list = glob.glob(self.traydatafile_address.replace("%LOT%",self.lot_no)+self.lot_no+"*.csv")

        for trayfilename in traydata_list:
            output_traydata_address = self.output_result_address.replace("%LOT%",self.lot_no)+"output_traydata/"+trayfilename.split("\\")[-1]

            basefile = open(trayfilename,"r")
            outputfile = open(output_traydata_address,"w")

            baseline = basefile.readline()
            n=0
            while baseline:
                if n==0:
                    outputfile.write(baseline)
                elif n>=1:
                    data=baseline.split(",")
                    serial = data[0]
                    pf = self.pass_fail_list[self.serial_list.index(serial)]

                    if pf == "Pass":
                        data[2] = "0"
                        outputline = ",".join(data)
                        outputfile.write(outputline)
                    elif pf == "Fail":
                        data[2] = "F"
                        outputline = ",".join(data)
                        outputfile.write(outputline)

                baseline = basefile.readline()
                n+=1

            basefile.close()
            outputfile.close()
        return
    #---------------------------#
    #メイン処理                 #
    #---------------------------#

    def start_main_process(self):
        """entryに複数ロット名を書き込んだ時に対応できるように変更 23/8/11"""
        lot_no = self.LotName.get()
        type_name = self.TypeName.get()

        lot_no_num = len(lot_no.split(","))
        type_name_num = len(type_name.split(","))

        #テキストボックス初期化
        self.log_text.delete("1.0","end")
        #スタート時の日付を記入
        self.log_text.insert(tk.END,"処理開始時刻：{}\n".format(datetime.datetime.now()))

        #異常対応
        if lot_no_num != type_name_num:
            self.log_text.insert(tk.END,"機種名とロットNoの数があっていません\n")
            return
        elif lot_no == "":
            self.log_text.insert(tk.END,"ロットNoが入力されていません\n")
            return
        elif type_name == "":
            self.log_text.insert(tk.END,"ロットNoが入力されていません\n")
            return
        
        if lot_no_num == 1:
            self.main_process(lot_no,type_name,1)
        elif lot_no_num > 1:
            lot_no_list = lot_no.split(",")
            type_name_list = type_name.split(",")
            for i,lot in enumerate(lot_no_list):
                self.main_process(lot,type_name_list[i],2)

        return

    def main_process(self,lot_no,type_name,test_mode):
        '''メイン処理'''
        #insp_moduleのinspクラス
        insp_test = insp()

        #setting.txtを読込
        n=insp_test.read_setting_file(type_name,lot_no,self.test_sample_address,self.parameter_address,self.output_result_address)

        #settingファイル読み込み時のエラー
        if n==1:
            self.log_text.insert(tk.END,"ロット結果フォルダが既に存在しています\n")
            return

        #分割情報,規格情報を取得
        if insp_test.STD_TYPE == "FIX":
            """FIXの場合split_data.txtに書き込まれている規格を読み込み"""
            standards = insp_test.get_split_data()
        else:
            insp_test.get_split_data()

        #検査後の解析時に使用するリストを残す
        self.split_info_list = insp_test.SPLIT_DATA.copy()
        self.split_info = True 

        #各分割の分割数を出しておく
        split_num=[]
        for i in range(len(insp_test.SPLIT_DATA)):
            split_num.append(len(insp_test.SPLIT_DATA[i]))

        """
        良品学習実施
        """
        #初めにお手本画像を作成する
        if insp_test.PROCESS_PARENT_IMAGE == 1:
            img_good_average = insp_test.get_parent_img(insp_test.PARENT_IMG_FILE)
            #加工後の画像を出力する
            cv2.imwrite(self.parameter_address.replace("%LOT%",lot_no).replace("%PRODUCT%",type_name)+"parent_img_after_processing.jpg",img_good_average)
        else:
            img_good_average = cv2.imread(insp_test.PARENT_IMG_FILE,cv2.IMREAD_GRAYSCALE)

        #お手本画像との輝度合わせ用にデータを取得
        mean_good = img_good_average.mean()
        std_good = img_good_average.std()
        brightness_data = [mean_good,std_good]

        #お手本にフィルターをかける(リストになる)
        img_good_average = insp_test.image_filter(img_good_average)

        #プロセスタイプによって分岐 1->良品条件出し込みの全処理 2->良品条件出しのみ
        if self.process_mode ==1 or self.process_mode==2:
            #良品画像読み込み
            print("良品画像読み込み中")
            good_image_list = []

            good_folder_list = os.listdir(insp_test.GOOD_SAMPLE_FOLDER)

            for folder in good_folder_list:
                """good_image_listに良品フォルダ内の全画像アドレスを入れる"""
                images = glob.glob(insp_test.GOOD_SAMPLE_FOLDER+folder+"/*AA.JPG")
                for image in images:
                    good_image_list.append(image)
            del good_folder_list
            gc.collect()

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

            good_num = len(good_image_list)
            if good_num%insp_test.ONE_TEST_SIZE != 0: 
                good_images.append(tmp)
            del tmp
            gc.collect() 
            print("良品画像数は{}枚です".format(good_num))

            #良品画像と平均画像の差分ベクトルを作成
            print("良品の差分ベクトル作成中")
            #good_diff_image_listを初期化
            good_diff_image_list = []
            for i in range(len(split_num)):
                good_diff_image_list.append([[] for s in range(split_num[i])])

            #画像読み込みから良品学習を開始
            for i,images in enumerate(good_images):
                good_image_list = []
                img_name_list = []
                #opencvで読み込んだ画像をgood_image_listに入れる(ONE_TEST_SIZE分)
                for j,image in enumerate(images):
                    img_good = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
                    good_image_list.append(img_good)
                    img_name_list.append(image)

                #差分ベクトルをgood_diff_image_listに入れていく(good_diff_image_listはsplit_numのおおきさ)
                for j,image in enumerate(good_image_list):
                    diff_image_list = insp_test.get_diff_image_list(img_good_average,image,brightness_data,i*insp_test.ONE_TEST_SIZE+j+1,1,img_name_list[j])
                    for k,s in enumerate(split_num):
                        for l in range(s):
                            good_diff_image_list[k][l].append(diff_image_list[k][l])

                    del diff_image_list
                    gc.collect()

               #メモリ開放
                del good_image_list
                gc.collect()

            #one-class-svmで良品学習
            print("\n良品学習中")
            models = insp_test.learn_good_feature(good_diff_image_list)

            #スコア判定規格設定
            print("\nスコア判定規格設定中")
            good_scores = insp_test.good_predict(models,good_diff_image_list)
            del good_diff_image_list
            gc.collect()

            #one-class-svmで良品学習
            print("良品学習中")
            """良品スコアのcsvへの書き出し"""
            output_file = open(insp_test.GOOD_RESULT_FILE,"w")
            #header
            output_line = "split,"
            for i in range(good_num):
                output_line += str(i+1)+","
            output_line += "\n"
            output_file.write(output_line)
            #本体結果出力
            for i,scores_per_filter in enumerate(good_scores):
                for j,scores in enumerate(scores_per_filter):
                    output_line = str(i+1)+"_"+str(j+1)+","
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
            """良品学習のみの場合"""
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
        print("テスト画像読み込み開始")

        if MASSPRODUCTION_MODE == True:
            """トレイデータ読み込み"""
            print("トレイデータ読み込み")
            traydata_info=[]
            traydata_list = glob.glob(self.traydatafile_address.replace("%LOT%",lot_no)+lot_no+"*.csv")
            for traydata_name in traydata_list:
                traydata = open(traydata_name,"r")
                traydata_line = traydata.readline()
                n=0
                while traydata_line:
                    if n>0:
                        data = traydata_line.split(",")
                        #PATでNGになっているものは見ない
                        if data[2] == "0":
                            #serial,tray番号,trayX,trayY
                            data_list = [data[0],data[1],data[3],data[4]]
                            traydata_info.append(data_list)
                    traydata_line = traydata.readline()
                    n+=1
                traydata.close()

            """resultファイル読み込み"""
            print("インライン外観検査result読み込み")
            test_image_files = []
            serial_list = []
            surf_result_name = self.surf_resultfile_address.replace("%LOT%",lot_no)+"result1_"+lot_no+"_Vision1.csv"
            surf_result_file = open(surf_result_name,"r")
            result_line = surf_result_file.readline()
            n=1
            while result_line:
                if n>=7:
                    data = result_line.split(",")
                    #Bin 0 良品のみ取り出し
                    if data[1] == "0":
                        serial = data[4]
                        #serialがPAT NG出ないことを確認
                        if serial in [x[0] for x in traydata_info]:
                            num = int(data[0])-1 #画像ファイル名はnum-1を5桁表示したものになっている
                            imgname = insp_test.TEST_SAMPLE_FOLDER + "%05d"%num+"AA.jpg"
                            if not os.path.exists(imgname):
                                print("{}が存在しません".format(imgname))
                                return
                            serial_list.append(serial)
                            test_image_files.append(imgname) 
                result_line = surf_result_file.readline()
                n=n+1
            surf_result_file.close() 
            test_num=len(test_image_files)
            print("テスト画像数は{}枚です".format(test_num))
        else:
            test_image_files = glob.glob(insp_test.TEST_SAMPLE_FOLDER+"*AA.JPG")
            test_num = len(test_image_files)
            print("テスト画像数は{}枚です".format(test_num))

        #ONE_TEST_SIZE区切りの2次元配列に変換
        test_images=[]
        test_image_name=[] #画像の名前を入れるリスト
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
        #結果を入れるリストを複数フィルターに対応するように初期化
        #predictionsを初期化
        predictions=[]
        for n in split_num:
            predictions.append([[] for i in range(n)])
        
        #画像出力用フォルダ作成
        if not os.path.exists(insp_test.OUTPUT_ALL_IMAGE):
            os.makedirs(insp_test.OUTPUT_ALL_IMAGE)
        if not os.path.exists(insp_test.OUTPUT_FAIL_IMAGE):
            os.makedirs(insp_test.OUTPUT_FAIL_IMAGE)

        for i,images in enumerate(test_images):
            #test_diff_image_listを初期化
            test_diff_image_list = []
            #位置情報取得に成功したかを入れるリストを初期化
            success_get_chip_image_onetestsize=[]
            for m in range(len(split_num)):
                test_diff_image_list.append([[] for s in range(split_num[m])])
            test_image_list=[] #opencvで開いた画像を入れる
            img_name_list = [] #画像の名前を入れる
            for j,image in enumerate(images): #imagesはONE_TEST_SIZE枚画像が入っている
                #画像をopencvで読み込み、リストに入れる
                img_test = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
                test_image_list.append(img_test)
                #test_image_name.append(image.split("/")[-1].split(".")[0]) #result_data.csvに表示用 #for linux
                test_image_name.append(image.split("\\")[-1].split(".")[0]) #result_data.csvに表示用 #for windows
                img_name_list.append(image) #terminalに表示用
            for j,image in enumerate(test_image_list):
                '''複数フィルターに対応できるように変更'''
                '''位置補正に失敗した場合に対応 23/9/2'''
                diff_image_list = insp_test.get_diff_image_list(img_good_average,image,brightness_data,i*insp_test.ONE_TEST_SIZE+j+1,0,img_name_list[j])
                if diff_image_list == 0:
                    success_get_chip_image_onetestsize.append(False)
                    '''位置補正失敗した場合はtest_diff_image_listに入れない'''
                    continue
                success_get_chip_image_onetestsize.append(True)
                for k,s in enumerate(split_num):
                    for l in range(s):
                        test_diff_image_list[k][l].append(diff_image_list[k][l]) #フィルターの数 X ONE_TEST_SIZE枚分の差分ベクトル

            #テストデータをつかってone-class-svmのスコア取得
            if len(test_diff_image_list[0]) == 0:
                '''全ての画像の位置情報取得に失敗した場合の分岐を追加 23/09/02'''
                #スコアが0のリストを結果とする
                result_list = [0 for x in range(len(success_get_chip_image_onetestsize))]

                #predictionsに総テスト結果を入れていく
                for k,s in enumerate(split_num): #フィルターの種類の数
                    for l in range(s): #分割の数
                        predictions[k][l] += result_list

                #リスト初期化、メモリ開放
                del test_image_list
                del test_diff_image_list
                continue

            results = insp_test.result_predict(models,test_diff_image_list,success_get_chip_image_onetestsize)

            #predictionsに総テスト結果を入れていく
            for k,s in enumerate(split_num): #フィルターの種類の数
                for l in range(s): #分割の数
                    predictions[k][l] += results[k][l]

            #リスト初期化、メモリ開放
            del test_image_list
            del test_diff_image_list

        print("\n画像評価完了")
        print("判定中")

        #全ての位置情報取得に失敗した場合
        if insp_test.success_get_chip_image.count(False)==test_num:
            print("全ての画像の位置情報取得に失敗しました")
            return

        #PATモードの場合
        if self.process_mode == 3 and insp_test.STD_TYPE != "FIX":
            standards = insp_test.set_standards(predictions)

        #良品不良品判定
        judge_results = insp_test.judge_pass_fail(standards,predictions,test_num,test_image_name)

        output_predictions = []
        for j in range(len(split_num)):
            output_predictions.append(np.array(predictions[j]).T)
        del predictions
        gc.collect()

        '''結果出力'''
        output_file = open(insp_test.OUTPUT_RESULT_FILE,"w")
        #header出力
        output_line = "standards,,,,,,,"
        for i,s in enumerate(split_num):
            for j in range(s):
                output_line+=str(standards[i][j])+","
        output_line += "\n"
        output_file.write(output_line)
        output_file.write("\n")
        output_line = "no,p/f,theta,x,y,w,h,"
        for i,s in enumerate(split_num):
            for j in range(s):
                output_line+=str(i+1)+"_"+str(j+1)+","
        output_line += "\n"
        output_file.write(output_line)

        #メイン結果出力
        fail_num = 0
        pass_num = 0
        unknown_num = 0
        pass_fail_list=[]
        for i in range(test_num): #テスト数
            flag=0
            if insp_test.success_get_chip_image[i]==False:
                flag=2
            elif insp_test.success_get_chip_image[i]==True:
                for j in range(len(split_num)): #分割数
                    if "1" in judge_results[j][i]:
                        flag=1
            if flag==1:    
                output_line = test_image_name[i]+","+"f"+","+str(insp_test.theta_list[i][0])+","+str(insp_test.theta_list[i][1])+","+str(insp_test.theta_list[i][2])+","+str(insp_test.theta_list[i][3])+","+str(insp_test.theta_list[i][4])+","
                fail_num += 1
                pass_fail_list.append("Fail")
            elif flag==2:
                #位置情報取得失敗の未確認画像用の出力
                output_line = test_image_name[i]+","+"n"+","+str(insp_test.theta_list[i][0])+","+str(insp_test.theta_list[i][1])+","+str(insp_test.theta_list[i][2])+","+str(insp_test.theta_list[i][3])+","+str(insp_test.theta_list[i][4])+","
                unknown_num += 1
                pass_fail_list.append("UNKNOWN")
            else:
                output_line = test_image_name[i]+","+"p"+","+str(insp_test.theta_list[i][0])+","+str(insp_test.theta_list[i][1])+","+str(insp_test.theta_list[i][2])+","+str(insp_test.theta_list[i][3])+","+str(insp_test.theta_list[i][4])+","
                pass_num += 1
                pass_fail_list.append("Pass")

            #各矩形のp/f(0/1)出力
            #for j in range(len(split_num)): #分割数
            #    for k in range(split_num[j]):
            #        output_line+=judge_results[j][i][k]+","

            #score出力
            for j in range(len(split_num)):
                for p in output_predictions[j][i]:
                    output_line+=str(p)+","
                
            output_line += "\n"
            output_file.write(output_line)

        output_file.close()
        print("良品数は{}、不良品数は{}、不明なものは{}、歩留まりは{}%".format(pass_num,fail_num,unknown_num,"%.1f"%(pass_num*100/(pass_num+fail_num+unknown_num))))

        if (MASSPRODUCTION_MODE==True and pass_num/(pass_num+fail_num) < YIELD_STANDARD) or (MASSPRODUCTION_MODE==True and unknown_num!=0):
            print("歩留まりが低い もしくは 検査できなかったチップがあったため不良チップの目視確認をお願いします")
            messagebox.showinfo("確認","目視確認を実施してください\n良品数は{}、不良品数は{}、未検査チップ数は{}".format(pass_num,fail_num,unknown_num))
            self.result_analysis(type_name,lot_no)
            print("目視確認用ポップアップを表示します")
            self.create_popup(traydata_info,serial_list,test_image_files,pass_fail_list,lot_no)
        elif MASSPRODUCTION_MODE==True and pass_num/(pass_num+fail_num)>=YIELD_STANDARD:
            if test_mode == 1:
                #処理ロットが1つであればポップアップ表示と解析用の処理をする
                messagebox.showinfo("確認","処理が正常に終了しました\n良品数は{}、不良品数は{}".format(pass_num,fail_num))
                self.result_analysis(type_name,lot_no)
            print("トレイデータ書き出し")
            self.lot_no = lot_no
            self.traydata = traydata_info
            self.serial_list = serial_list
            self.pass_fail_list = pass_fail_list
            self.output_traydata()
            print("正常終了")
        elif MASSPRODUCTION_MODE==False:        
            if test_mode == 1:
                #処理ロットが1つであればポップアップ表示と解析用の処理をする
                messagebox.showinfo("確認","処理が正常に終了しました\n良品数は{}、不良品数は{}".format(pass_num,fail_num))
                self.result_analysis(type_name,lot_no)
        return

if __name__ == "__main__":
    root = tk.Tk()
    root.state("zoomed")
    photo = tk.PhotoImage(file="./tk_icon.png")
    root.iconphoto(True,photo)
    app = Application(master=root)
    app.mainloop()
