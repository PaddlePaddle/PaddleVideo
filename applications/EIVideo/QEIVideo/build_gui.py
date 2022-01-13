# Author: Acer Zhang
# Datetime:2022/1/11 
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import json
import os

import numpy as np
from PIL import Image

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2

from EIVideo.api import json2frame, png2json, load_video
from EIVideo.main import main
# ToDo To AP-kai: 这是定义前端临时保存用于推理的json的地点之类的，因为是固定的，所以声明为全局常量是最好的
from EIVideo import TEMP_JSON_SAVE_PATH, TEMP_IMG_SAVE_PATH, TEMP_JSON_FINAL_PATH

from QEIVideo.gui.ui_main_window import Ui_MainWindow


class BuildGUI(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(BuildGUI, self).__init__()
        # ToDo To AP-kai: 这里定义当前选择的视频路径的占位符，相当于全局变量
        self.select_video_path = None
        # ToDo To AP-kai: 未来为用户提供个保存路径的入口哈，这里先随意定义了个路径
        self.save_path = "./result"
        os.makedirs(self.save_path, exist_ok=True)
        self.setupUi(self)

    def infer(self):
        self.label.setText("Start infer")
        self.progressBar.setProperty("value", 0)
        image = self.paintBoard.get_content_as_q_image()
        image.save(TEMP_IMG_SAVE_PATH)
        print(self.slider_frame_num)
        self.progressBar.setProperty("value", 25)
        # ToDo To AP-kai:相同的文件路径，直接定义一个常量就好
        png2json(TEMP_IMG_SAVE_PATH, self.slider_frame_num, TEMP_JSON_SAVE_PATH)
        self.progressBar.setProperty("value", 50)
        # ToDo To AP-kai:打印的信息，需要注意首字母大写
        # ToDo To AP-kai: 此处传入保存路径以及当前选择的视频路径，最后会在manet_stage1.py里通过cfg来传入
        out = main(video_path=self.select_video_path, save_path=self.save_path)
        print('Infer ok')
        self.progressBar.setProperty("value", 75)
        self.all_frames = json2frame(TEMP_JSON_FINAL_PATH)
        print("Success get submit_masks")
        self.open_frame()
        self.progressBar.setProperty("value", 100)
        self.label.setText("Infer succeed")

    def btn_func(self, btn):
        if btn == self.playbtn:
            self.label.setText("Play video")
            if self.progress_slider.value() == self.cap.get(7) - 1:
                self.slider_frame_num = 0
                self.progress_slider.setValue(self.slider_frame_num)
                self.time_label.setText('{}/{}'.format(self.slider_frame_num, self.cap.get(7)))
            self.timer_camera = QTimer()  # 定义定时器
            self.timer_camera.start(1000 / self.cap.get(cv2.CAP_PROP_FPS))
            self.slider_frame_num = self.progress_slider.value()
            self.timer_camera.timeout.connect(self.open_frame)

        elif btn == self.pushButton_2:
            self.label.setText("Stop video")
            self.slot_stop()

        elif btn == self.pushButton_4:
            self.label.setText("Choose video")
            self.select_video_path, _ = QFileDialog.getOpenFileName(self.frame, "Open", "", "*.mp4;;All Files(*)")
            print("-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")
            print("Select video file path:\t" + self.select_video_path)
            # ToDo To AP-kai:下断点来看一下，如果不选择的时候返回值是什么样的，然后再做判断，目前这个if没有生效
            if self.select_video_path != "":
                self.cap = cv2.VideoCapture(self.select_video_path)
                # 存所有frame
                self.save_temp_frame()
                print("save temp frame done")
                self.progress_slider.setRange(0, self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.slider_frame_num = 0
                self.open_frame()

            # ToDo To AP-kai: 未来这个地方增加提示框，告诉他没有选择文件

    def on_cbtn_eraser_clicked(self):
        self.label.setText("Eraser On")
        if self.cbtn_Eraser.isChecked():
            self.paintBoard.EraserMode = True  # 进入橡皮擦模式
        else:
            self.paintBoard.EraserMode = False  # 退出橡皮擦模式

    def fill_color_list(self, combo_box):
        index_black = 0
        index = 0
        for color in self.colorList:
            if color == "black":
                index_black = index
            index += 1
            pix = QPixmap(70, 20)
            pix.fill(QColor(color))
            combo_box.addItem(QIcon(pix), None)
            combo_box.setIconSize(QSize(70, 20))
            combo_box.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        combo_box.setCurrentIndex(index_black)

    def on_pen_color_change(self):
        self.label.setText("Change pen color")
        color_index = self.comboBox_penColor.currentIndex()
        color_str = self.colorList[color_index]

        self.paintBoard.change_pen_color(color_str)

    # 拖拽进度条
    def update_video_position_func(self):
        self.label.setText("Change slider position")
        self.slider_frame_num = self.progress_slider.value()
        self.slot_stop()
        self.open_frame()
        self.progress_slider.setValue(self.slider_frame_num)
        self.time_label.setText('{}/{}'.format(self.slider_frame_num, self.cap.get(7)))

    def save_temp_frame(self):
        _, self.all_frames = load_video(self.select_video_path, 480)

    def slot_stop(self):
        if self.cap != []:
            self.timer_camera.stop()  # 停止计时器
        else:
            # ToDo To AP-kai: QMessageBox.warning没有返回值，这里我把Warming = QMessageBox.warning的Warming删去了
            QMessageBox.warning(self, "Warming", "Push the left upper corner button to Quit.",
                                QMessageBox.Yes)

    def open_frame(self):
        self.progress_slider.setValue(self.slider_frame_num)
        self.slider_frame_num = self.progress_slider.value()
        self.frame = self.all_frames[self.slider_frame_num]
        frame = self.frame
        height, width, bytes_per_component = frame.shape
        bytes_per_line = bytes_per_component * width
        q_image = QImage(frame.data, width, height, bytes_per_line,
                         QImage.Format_RGB888).scaled(self.picturelabel.width(), self.picturelabel.height())
        self.picturelabel.setPixmap(QPixmap.fromImage(q_image))
        self.slider_frame_num = self.slider_frame_num + 1
        self.time_label.setText('{}/{}'.format(self.slider_frame_num, self.cap.get(7)))
        if self.progress_slider.value() == self.cap.get(7) - 1:
            self.slot_stop()
