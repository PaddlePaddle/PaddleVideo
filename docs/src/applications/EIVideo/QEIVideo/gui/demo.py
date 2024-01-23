# Author: Acer Zhang
# Datetime: 2022/1/6 
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QFrame, QWidget
from PyQt5.QtGui import QPainter, QPixmap, QPen, QColor, QPainterPath
from PyQt5.QtCore import Qt, QPoint
from PyQt5 import QtCore, QtGui, QtWidgets

from QEIVideo.ui.demo import Ui_MainWindow as DemoUIRoot


class DrawFrame(QWidget):
    def __init__(self, painter, *args, **kwargs):
        super(DrawFrame, self).__init__(*args, **kwargs)
        self.painter = painter

    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen(QColor("orange"))
        pen.setWidth(5)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawPath(self.painter)

    def mousePressEvent(self, event):
        self.painter.moveTo(event.pos())
        self.update()

    def mouseMoveEvent(self, event):
        self.painter.lineTo(event.pos())

        self.update()


class DemoUI(QMainWindow, DemoUIRoot):
    def __init__(self):
        super(DemoUI, self).__init__()
        self.setupUi(self)

        self.painter = QPainterPath()
        self.draw_frame = DrawFrame(self.painter, self.video_frame)
        self.draw_frame.setGeometry(QtCore.QRect(0, 10, 751, 301))
        self.draw_frame.setObjectName("draw_frame")
        self.draw_frame.raise_()
        self.draw_frame.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        self.start_btn.clicked.connect(self.export)

    def export(self):
        a = self.painter.toFillPolygon()
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui_class = DemoUI()
    gui_class.show()
    sys.exit(app.exec_())
