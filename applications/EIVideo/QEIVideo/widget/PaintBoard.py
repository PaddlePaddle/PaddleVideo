from PyQt5.QtWidgets import QWidget
from PyQt5.Qt import QPixmap, QPainter, QPoint, QPaintEvent, QMouseEvent, QPen, \
    QColor, QSize
from PyQt5.QtCore import Qt


class PaintBoard(QWidget):

    def __init__(self, parent=None):
        '''
        Constructor
        '''
        super().__init__(parent)

        self.__init_data()  # 先初始化数据，再初始化界面
        self.__init_view()

    def __init_data(self):

        self.__size = QSize(810, 458)

        # 新建QPixmap作为画板，尺寸为__size
        self.__board = QPixmap(self.__size)
        self.__board.fill(Qt.transparent)  # 用透明填充画板

        self.__IsEmpty = True  # 默认为空画板
        self.EraserMode = False  # 默认为禁用橡皮擦模式

        self.__lastPos = QPoint(0, 0)  # 上一次鼠标位置
        self.__currentPos = QPoint(0, 0)  # 当前的鼠标位置

        self.__painter = QPainter()  # 新建绘图工具

        self.__thickness = 15  # 默认画笔粗细为10px
        self.__penColor = QColor("black")  # 设置默认画笔颜色为黑色
        self.__colorList = QColor.colorNames()  # 获取颜色列表

    def __init_view(self):
        # 设置界面的尺寸为__size
        self.setFixedSize(self.__size)

    def clear(self):
        # 清空画板
        # self.__board.fill(Qt.white)
        self.__board = QPixmap(self.__size)
        self.__board.fill(Qt.transparent)  # 用透明填充画板

        self.update()
        self.__IsEmpty = True

    def change_pen_color(self, color="black"):
        # 改变画笔颜色
        # rgbaColor = QColor(255, 255, 0, 100)
        self.__penColor = QColor(color)

    def change_pen_thickness(self, thickness=10):
        # 改变画笔粗细
        self.__thickness = thickness

    def is_empty(self):
        # 返回画板是否为空
        return self.__IsEmpty

    def get_content_as_q_image(self):
        # 获取画板内容（返回QImage）
        image = self.__board.toImage()
        return image

    def paintEvent(self, paint_event):
        # 绘图事件
        # 绘图时必须使用QPainter的实例，此处为__painter
        # 绘图在begin()函数与end()函数间进行
        # begin(param)的参数要指定绘图设备，即把图画在哪里
        # drawPixmap用于绘制QPixmap类型的对象
        self.__painter.begin(self)
        # 0,0为绘图的左上角起点的坐标，__board即要绘制的图
        self.__painter.drawPixmap(0, 0, self.__board)
        self.__painter.end()

    def mousePressEvent(self, mouse_event):
        # 鼠标按下时，获取鼠标的当前位置保存为上一次位置
        self.__currentPos = mouse_event.pos()
        self.__lastPos = self.__currentPos

    def mouseMoveEvent(self, mouse_event):
        # 鼠标移动时，更新当前位置，并在上一个位置和当前位置间画线
        self.__currentPos = mouse_event.pos()
        self.__painter.begin(self.__board)

        if self.EraserMode == False:
            # 非橡皮擦模式
            self.__painter.setPen(QPen(self.__penColor, self.__thickness))  # 设置画笔颜色，粗细
        else:
            # 橡皮擦模式下画笔为纯白色，粗细为10
            self.__painter.setPen(QPen(Qt.transparent, 10))

        # 画线
        # print(self.__lastPos + self.__currentPos)
        self.__painter.drawLine(self.__lastPos, self.__currentPos)
        self.__painter.end()
        self.__lastPos = self.__currentPos

        self.update()  # 更新显示

    def mouseReleaseEvent(self, mouseEvent):
        self.__IsEmpty = False  # 画板不再为空
