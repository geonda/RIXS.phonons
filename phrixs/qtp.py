from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import json
import sys

class Window(QMainWindow):
    def __init__(self, *args, **kwargs):
        super(Window, self).__init__(*args, **kwargs)
        self.setWindowTitle("Input")
        self.layout = QVBoxLayout()
        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)


class qbox(object):
    """spinBox widgets"""
    def __init__(self,layout,dict,labels):
        super(qbox, self).__init__()
        self.dict=dict
        self.labels=labels
        self.layout=layout
        self.box=QDoubleSpinBox()
        self.label=QLabel()
        self.label.setText(labels)
        self.box.setRange(1, 1000)
        self.box.setValue(self.dict[labels])
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.box)
        self.box.valueChanged.connect(lambda: self.change_dict())

    def change_dict(self):
        self.dict[self.labels]=self.box.value()
        print('new value is ',self.dict[self.labels])
class app(object):
    """docstring for app."""
    def __init__(self, dict):
        super(app, self).__init__()
        _app = QApplication(sys.argv)
        self.window=Window()
        for labels in dict:
            qbox(self.window.layout,dict,labels)


        self.app=_app

        # window.show()
        # _app.exec_()
        # sys.exit(_app.exec_())
