from lib import *
from tqdm import tqdm
from qtp import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


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
        self.box.setRange(0, 10)
        self.box.setValue(self.dict[labels])
        # self.box.setText("%.2f", % (calculation * 30))
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.box)
        self.box.valueChanged.connect(lambda: self.change_dict())

    def change_dict(self):
        self.dict[self.labels]=self.box.value()
        print('new value is ',self.dict[self.labels])


class app(QMainWindow):

    def __init__(self):
        super().__init__()
        self.ws=workspace()
        self.ws.timer_start()
        self.ws.initp()
        self.ws.timer_round('init done [s]: ')
        self.ws.inputp('skip')
        self.layout = QVBoxLayout()
        self.tytab = QTabWidget(self)
        self.tab =QWidget()
        self.tytab.addTab(self.tab, 'name')
        self.tytab.addTab(self.tab, 'name2')
        self.ttlayout = QGridLayout()
        # self.widget = QWidget()
        self.tytab.setLayout(self.ttlayout)
        # self.setCentralWidget(self.widget)
        self.initUI()

    def initUI(self):


        for labels in self.ws.dict_total['input']:
            qbox(self.ttlayout,self.ws.dict_total['input'],labels)
        # button = QPushButton('run', self)
        # # button.setToolTip('This is an example button')
        # # button.move(100,70)
        # button.clicked.connect(self.on_click)

        self.show()

    @pyqtSlot()
    def on_click(self):
        print('running')
        self.ws.runp()
        self.ws.plotp()

if __name__ == '__main__':
    app_ = QApplication(sys.argv)
    ex = app()
    # app_.exec_()
    sys.exit(app_.exec_())
