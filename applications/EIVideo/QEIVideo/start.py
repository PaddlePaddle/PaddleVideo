# Author: AP-Kai
# Datetime: 2022/1/7
# Copyright belongs to the author.
# Please indicate the source for reprinting.


import sys
from QEIVideo.build_gui import BuildGUI
from PyQt5.QtWidgets import QApplication


def run():
    app = QApplication(sys.argv)
    demo = BuildGUI()
    demo.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    run()
