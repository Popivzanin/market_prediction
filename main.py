import sys
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QLabel, QPushButton, QLineEdit, QMessageBox, QCheckBox, QVBoxLayout, QHBoxLayout, QGridLayout
import pandas as pd
import yfinance as yf
import pickle as pkl
from datetime import datetime, date, timedelta
from preprocessing import preprocess
import sklearn


app = QApplication(sys.argv)


class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		
		layout = QVBoxLayout()
		layoutForest = QHBoxLayout()
		layoutIchimoku = QHBoxLayout()
		pagelayout = QVBoxLayout()
		lower_layout = QHBoxLayout()
		self.setWindowTitle("Gold trend prediction")
		self.setFixedSize(600,600)
		
		pagelayout.addLayout(layout)
		pagelayout.addLayout(lower_layout)

		self.button = QPushButton("Predict")
		self.button.clicked.connect(self.predict)

		self.labelRandomForest = QLabel("Random forest prediction:", self)
		self.labelRandomForest.setStyleSheet("color: black;font-weight: bold; font-size: 18pt;")
		self.labelRandomForestPrediction = QLabel("No prediction yet", self)
		self.labelRandomForestPrediction.setStyleSheet("color: blue;font-weight: bold; font-size: 18pt;")	

		self.labelIchimoku = QLabel("Ichimoku prediction:", self)	
		self.labelIchimoku.setStyleSheet("color: black;font-weight: bold; font-size: 18pt;")	
		self.labelIchimokuPrediction = QLabel("No prediction yet", self)
		self.labelIchimokuPrediction.setStyleSheet("color: blue;font-weight: bold; font-size: 18pt;")	
		#self.labelCsv = QLabel("Use CSV", self)
		#self.inputCsv = QLineEdit()
		#self.inputCsv.setPlaceholderText("Enter data path")
		#self.inputCsv.hide()
		
		#self.checkBoxCsv = QCheckBox(self)
		#self.checkBoxCsv.stateChanged.connect(self.check)
		
		layoutForest.addWidget(self.labelRandomForest)
		layoutForest.addWidget(self.labelRandomForestPrediction)
		layoutIchimoku.addWidget(self.labelIchimoku)
		layoutIchimoku.addWidget(self.labelIchimokuPrediction)
		layout.addLayout(layoutForest)
		layout.addLayout(layoutIchimoku)
		lower_layout.addWidget(self.button)
		#lower_layout.addWidget(self.labelCsv)
		#lower_layout.addWidget(self.checkBoxCsv)
		#lower_layout.addWidget(self.inputCsv)

		widget = QWidget()
		widget.setLayout(pagelayout)
		self.setCentralWidget(widget)
		#grid.addWidget(self.button)
		#grid.addWidget(self.labelRandomForest)
		#grid.addWidget(self.labelRandomForestPrediction)
		#grid.addWidget(self.labelIchimoku)
		#grid.addWidget(self.labelIchimokuPrediction)
		#grid.addWidget(self.labelIchimokuPrediction)
		

	def predict(self):		
		df = None
		#if(self.checkBoxCsv.isChecked()):	
		#	csv = self.inputCsv.text()
		#	try:
		#		df = pd.read_csv(csv)
		#	except:
		#		msg = QMessageBox()
		#		msg.setWindowTitle("Error")
		#		msg.setText("Enter correct path")
		#		msg.exec_()
		#		return
		#else:	
		df = yf.download("GC=F", start=(datetime.today() - timedelta(days=200)).strftime("%Y-%m-%d"), end = pd.Timestamp.today().strftime("%Y-%m-%d"))
		df.to_csv('current_data.csv')	

		self.ichimoku(df)

		self.randomforest(df)		
		
	#def check(self):
	#	if(self.checkBoxCsv.isChecked()):
	#		self.inputCsv.show()
	#	else:
	#		self.inputCsv.hide()


	def ichimoku(self, df):
		
		return	

	def randomforest(self, df):
		#try:
		df = preprocess(df)
		#except:	
		#	msg = QMessageBox()
		#	msg.setWindowTitle("Error")
		#	msg.setText("Data is incorrect")
		#	msg.exec_()
		#	return
		model = None

		try:
			model = pkl.load(open('./models/model.pkl', 'rb'))
		except:	
			msg = QMessageBox()
			msg.setWindowTitle("Error")
			msg.setText("Model is not loaded")
			msg.exec_()
			return
		if model.predict(df.tail(1)) == 1:
			self.labelRandomForestPrediction.setText("Growing")
			self.labelRandomForestPrediction.setStyleSheet("color: green;font-weight: bold; font-size: 18pt")
		else:
			self.labelRandomForestPrediction.setText("Falling")
			self.labelRandomForestPrediction.setStyleSheet("color: red;font-weight: bold; font-size: 18pt;")


window = MainWindow()
window.show()

app.exec()
