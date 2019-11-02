import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import qdarkgraystyle
import pyqtgraph as pg
from lib import *
from tqdm import tqdm
from qtp import *



class qbox(object):
    """docstring for qbox."""
    def __init__(self,grid,dict,labels,row=0,column=0 ):
        super(qbox, self).__init__()
        self.dict=dict
        try:
            if labels=='g0' or labels=='g1':
                self.box=QLineEdit()
                self.box.setText('')
            else:
                self.box=QDoubleSpinBox()
                self.box.setRange(0, 10)
                self.box.setSingleStep(0.1)
                self.box.setValue(float(self.dict[labels]))
                self.box.valueChanged.connect(lambda: self.change_dict(self.box,self.label))
        except:
            self.box=QLineEdit()
            self.box.setText(str(self.dict[labels]))
        self.label=QLabel()
        self.label.setText(labels)

        self.grid=grid
        self.grid.addWidget(self.label,row,column)
        self.grid.addWidget(self.box,row,column+1)
    def change_dict(self,box,label):
        self.dict[label.text()]=float(box.value())
        print(label.text()+' new value is ',self.dict[label.text()],box.value())

class ExampleWidget(QGroupBox):
    def __init__(self, numAddWidget,plot,layout):
        QGroupBox.__init__(self)
        self.layout =QVBoxLayout(self)
        self.numAddWidget = numAddWidget
        self.numAddItem   = 1
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        # self.tabs.resize(500,500)

        # self.tabs.resize(500,500)
        # self.tabs.setStyleSheet("QTabBar::tab { height: 500px; width: 500px}")
        # Add tabs

        self.tabs.addTab(self.tab1,"input".format(self.numAddWidget))
        self.tabs.addTab(self.tab2,"problem")
        
        self.plot=plot
        self.ws=workspace()
        self.ws.timer_start()
        self.ws.initp()
        self.ws.inputp('skip')
        self.dict_total=self.ws.dict_total

        self.tab1.layout = QGridLayout(self)
        self.tab1.setLayout(self.tab1.layout)
        self.tab2.layout = QGridLayout(self)
        self.tab2.setLayout(self.tab2.layout)
        self.initSubject_problem()
        self.initSubject_input()

        self.layout.addWidget(self.tabs)
        self.addbtn = QPushButton("run", self)
        self.addbtn.clicked.connect(self.run_calc)
        self.layout.addWidget(self.addbtn)
        self.setLayout(self.layout)
        # self.setGeometry(0, 0, 300, 500)
        # self.organize()
    def initSubject_problem(self):
        column=0;row=0
        for labels in (self.dict_total['problem']):
            if row > 3:
                column+=2
                row=0
            qbox(self.tab2.layout,self.dict_total['problem'],labels,row=row,column=column)
            row+=1

    def initSubject_input(self):

        # self.lblName = QLabel("input {}".format(self.numAddWidget), self)
        # self.lblSelectItem = QLabel(self)
        column=0;row=0
        for labels in (self.dict_total['input']):
            if row > 3:
                column+=2
                row=0
            qbox(self.tab1.layout,self.dict_total['input'],labels,row=row,column=column)
            # column=1 if (int(i) >len(self.dict_total['input'])/2) else column=2
            row+=1

    def run_calc(self):
        print(self.ws.dict_total)
        with open('input.json', 'w') as fp:
            json.dump(self.ws.dict_total['input'],fp)
        self.ws.runp()
        self.ws.plotp_app(self.plot)

class MyApp(QWidget):
    def __init__(self):
        super().__init__()

        loc=QLocale.system()
        loc.setNumberOptions(QLocale().c().numberOptions())
        QLocale.setDefault(loc)
        # print(QLocale.decimalPoint())
        self.numAddWidget = 1
        self.plot = pg.PlotWidget()
        # self.control()
        self.initUi()


    def initUi(self):
        mygroupbox = QtGui.QGroupBox('control')
        self.myform = QtGui.QFormLayout()
        widget=ExampleWidget(self.numAddWidget,self.plot,self.myform)
        self.myform.addWidget(widget)
        mygroupbox.setLayout(self.myform)
        scroll = QtGui.QScrollArea()
        scroll.setWidget(mygroupbox)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(550)
        scroll.setFixedWidth(600)
        self.layout = QtGui.QGridLayout(self)
        self.layout.addWidget(scroll,0,0)
        self.add_button = QPushButton("+")

        self.rem_button = QPushButton("-")
        self.layout.addWidget(self.add_button,1,0)
        self.layout.addWidget(self.rem_button,2,0)
        # layoutH=QHBoxLayout(self)
        self.layout.addWidget(self.plot,0,1,1,1)
        # layout.addLayout(layoutH)
        self.add_button.clicked.connect(self.addWidget)
        self.rem_button.clicked.connect(self.remWidget)
        # pg.mkQApp()
        # self.wg=pg.GraphicsWindow()
        # self.wg.resize(800,400)
        # self.plot=self.wg.addPlot()
        # layout = QtGui.QGridLayout()
        # w.setLayout(layout)
        # self.layoutH = QHBoxLayout(self)
        # self.initWidget([self.plot,self.plot,self.plot,self.plot])
        # # self.layoutH.addWidget(self.plot)
        # self.layoutH.addWidget(self.area)
        self.setGeometry(0, 0, 1300, 400)
        self.show()

    def addWidget(self):
        self.numAddWidget += 1
        self.widget = ExampleWidget(self.numAddWidget,self.plot,self.myform)
        self.myform.addWidget(self.widget)

    def remWidget(self):
        self.numAddWidget -= 1
        print(self.numAddWidget)
        self.layout.removeWidget(self.widget)
        self.widget.deleteLater()
        self.widget= None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # app.setStyleSheet(qdarkgraystyle.load_stylesheet())
    w = MyApp()



    sys.exit(app.exec_())
