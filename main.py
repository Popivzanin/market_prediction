import sys
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QLabel, QPushButton, QLineEdit, QMessageBox, QCheckBox, QVBoxLayout, QHBoxLayout, QGridLayout, QComboBox, QBoxLayout
import pandas as pd
import yfinance as yf
import pickle as pkl
from datetime import datetime, date, timedelta
from preprocessing import preprocess
import sklearn
from datetime import timedelta
import matplotlib
from matplotlib import pyplot as plt


app = QApplication(sys.argv)


class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		self.current_metal = 'Gold'
		self.df_ichimoku = None

		layout = QVBoxLayout()
		layoutForest = QHBoxLayout()
		layoutIchimoku1 = QHBoxLayout()
		layoutIchimoku2 = QHBoxLayout()
		pagelayout = QVBoxLayout()
		lower_layout = QHBoxLayout()
		self.setWindowTitle("Metal trend prediction")
		self.setFixedSize(600,600)
		
		pagelayout.addLayout(layout)
		pagelayout.addLayout(lower_layout)

		self.dropdown = QComboBox()
		self.dropdown.addItems(['Gold', 'Silver', 'Platinum'])
		self.dropdown.currentTextChanged.connect(self.combobox_changed)

		self.button = QPushButton("Predict")
		self.button.clicked.connect(self.predict)

		self.labelRandomForest = QLabel("Random forest prediction:", self)
		self.labelRandomForest.setStyleSheet("color: black;font-weight: bold; font-size: 18pt;")
		self.labelRandomForestPrediction = QLabel("No prediction yet", self)
		self.labelRandomForestPrediction.setStyleSheet("color: blue;font-weight: bold; font-size: 18pt;")	

		self.labelIchimoku1 = QLabel("Ichimoku prediction:", self)	
		self.labelIchimoku1.setStyleSheet("color: black;font-weight: bold; font-size: 18pt;")	
		self.labelIchimokuPrediction1 = QLabel("No prediction yet", self)
		self.labelIchimokuPrediction1.setStyleSheet("color: blue;font-weight: bold; font-size: 18pt;")

		#self.labelCsv = QLabel("Use CSV", self)
		#self.inputCsv = QLineEdit()
		#self.inputCsv.setPlaceholderText("Enter data path")
		#self.inputCsv.hide()


		self.labelIchimoku2 = QLabel("Ichimoku prediction:", self)	
		self.labelIchimoku2.setStyleSheet("color: black;font-weight: bold; font-size: 18pt;")	
		self.labelIchimokuPrediction2 = QLabel("No prediction yet", self)
		self.labelIchimokuPrediction2.setStyleSheet("color: blue;font-weight: bold; font-size: 18pt;")	
		

		self.buttonShow_Ichimoku = QPushButton("Show ichimoku")
		self.buttonShow_Ichimoku.clicked.connect(self.show_ichimoku)

		#self.checkBoxCsv = QCheckBox(self)
		#self.checkBoxCsv.stateChanged.connect(self.check)
		layout.addWidget(self.dropdown)
		layoutForest.addWidget(self.labelRandomForest)
		layoutForest.addWidget(self.labelRandomForestPrediction)
		layoutIchimoku1.addWidget(self.labelIchimoku1)
		layoutIchimoku1.addWidget(self.labelIchimokuPrediction1)
		layoutIchimoku2.addWidget(self.labelIchimoku2)
		layoutIchimoku2.addWidget(self.labelIchimokuPrediction2)
		layout.addLayout(layoutForest)
		layout.addLayout(layoutIchimoku1)
		layout.addLayout(layoutIchimoku2)
		lower_layout.addWidget(self.button)
		layout.addWidget(self.buttonShow_Ichimoku)
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
		
		
	def combobox_changed(self):
		self.current_metal = self.dropdown.currentText()

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
		match self.current_metal:
			case 'Gold':
				df = yf.download("GC=F", start=(datetime.today() - timedelta(days=200)).strftime("%Y-%m-%d"), end = pd.Timestamp.today().strftime("%Y-%m-%d"))
				df.to_csv('./data/gold.csv')	
			case 'Silver':
				df = yf.download("SI=F", start=(datetime.today() - timedelta(days=200)).strftime("%Y-%m-%d"), end = pd.Timestamp.today().strftime("%Y-%m-%d"))
				df.to_csv('./data/silver.csv')
			case 'Platinum':
				df = yf.download("PL=F", start=(datetime.today() - timedelta(days=200)).strftime("%Y-%m-%d"), end = pd.Timestamp.today().strftime("%Y-%m-%d"))
				df.to_csv('./data/platinum.csv')
		self.ichimoku_preprocess(df.copy())
		self.ichimoku_1()
		self.ichimoku_2()

		self.randomforest(df)		
		
	#def check(self):
	#	if(self.checkBoxCsv.isChecked()):
	#		self.inputCsv.show()
	#	else:
	#		self.inputCsv.hide()


	def ichimoku_preprocess(self, df):
		high_9 = df['High'].rolling(window= 9).max()
		low_9 = df['Low'].rolling(window= 9).min()
		df['tenkan_sen'] = (high_9 + low_9) /2

		high_26 = df['High'].rolling(window= 26).max()
		low_26 = df['Low'].rolling(window= 26).min()
		df['kijun_sen'] = (high_26 + low_26) /2

		# this is to extend the 'df' in future for 26 days
		# the 'df' here is numerical indexed df
		# last_index = df.iloc[-1:].index[0]
		# last_date = df['Date'].iloc[-1].date()
		# for i in range(26):
		# 	df.loc[last_index+1 +i, 'Date'] = last_date + timedelta(days=i)

		df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

		high_52 = df['High'].rolling(window= 52).max()
		low_52 = df['Low'].rolling(window= 52).min()
		df['senkou_span_b'] = ((high_52 + low_52) /2).shift(26)

		# most charting softwares dont plot this line
		df['chikou_span'] = df['Close'].shift(-22) #sometimes -26 

		tmp = df[['Close', 'senkou_span_a', 'senkou_span_b', 'kijun_sen', 'tenkan_sen', 'chikou_span']]
		self.df_ichimoku = tmp

	def ichimoku_1(self):
		df = self.df_ichimoku
		close = df['Close'].iloc[-1].item()
		tenkan = df['tenkan_sen'].iloc[-1].item()
		kijun = df['kijun_sen'].iloc[-1].item()
		chikou = df['chikou_span'].iloc[-1].item()
		if(tenkan > kijun and close > kijun and close > tenkan):
			self.labelIchimokuPrediction1.setText('Growing')
			self.labelIchimokuPrediction1.setStyleSheet('color: green;font-weight: bold; font-size: 18pt')
		else:
			self.labelIchimokuPrediction1.setText('Falling')
			self.labelIchimokuPrediction1.setStyleSheet('color: red;font-weight: bold; font-size: 18pt')
		return	
	
	def ichimoku_2(self):
		df = self.df_ichimoku
		close = df['Close'].iloc[-1].item()
		senkou_a = df['senkou_span_a'].iloc[-1].item()
		senkou_b = df['senkou_span_b'].iloc[-1].item()
		if(close > senkou_a and close > senkou_b):
			self.labelIchimokuPrediction2.setText('Growing')
			self.labelIchimokuPrediction2.setStyleSheet('color: green;font-weight: bold; font-size: 18pt')
		elif(close > senkou_a and close < senkou_b or close < senkou_a and close > senkou_b):
			self.labelIchimokuPrediction2.setText('Unclear')
			self.labelIchimokuPrediction2.setStyleSheet('color: blue;font-weight: bold; font-size: 18pt')
		else:
			self.labelIchimokuPrediction2.setText('Falling')
			self.labelIchimokuPrediction2.setStyleSheet('color: red;font-weight: bold; font-size: 18pt')

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
			self.labelRandomForestPrediction.setStyleSheet('color: green;font-weight: bold; font-size: 18pt;')
		else:
			self.labelRandomForestPrediction.setText('Falling')
			self.labelRandomForestPrediction.setStyleSheet('color: red;font-weight: bold; font-size: 18pt;')

	def show_ichimoku(self):
		if(self.df_ichimoku is None):
			msg = QMessageBox()
			msg.setWindowTitle("Error")
			msg.setText("Data is not loaded")
			msg.exec_()
			return
		df = self.df_ichimoku
		a1 = df.plot(figsize=(5,3))
		a1.fill_between(df.index, df.senkou_span_a, df.senkou_span_b)
		plt.title("Ichimoku")
		plt.show()
		return


window = MainWindow()
window.show()

app.exec()
