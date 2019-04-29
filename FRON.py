import sys
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QMainWindow, QCheckBox, QApplication, QWidget, QPushButton, QLabel
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot, QSize
import cv2
from imutils import face_utils
import numpy as np

import dlib

class App(QWidget):
	def __init__(self):
		super().__init__()
		self.title = 'Face Recognition'
		self.left = 200
		self.top = 200
		self.width = 400
		self.height = 200
		self.setWindowIcon(QtGui.QIcon('icon.png'))
		
		self.initUI()
		# self.webcam()
		# self.Face()
		
		# using the landmarks method
		self.landmarks()
		
	def webcam(self):
		cap = cv2.VideoCapture(0)

		while(True):
			# Capture frame-by-frame
			ret, frame = cap.read()

			# Our operations on the frame come here
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			# Display the resulting frame
			cv2.imshow('hey', gray)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		# When everything done, release the capture
		cap.release()
		cv2.destroyAllWindows()
	
	def Face(self):
		cascPath = sys.argv[1]
		faceCascade = cv2.CascadeClassifier(cascPath)
		video_capture = cv2.VideoCapture(0)
		while True:
			# Capture frame-by-frame
			ret, frame = video_capture.read()

			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			faces = faceCascade.detectMultiScale(
				gray,
				scaleFactor = 1.1,
				minNeighbors = 5,
				minSize = (30, 30),
				# flags = cv2.cv.CV_HAAR_SCALE_IMAGE
				flags = cv2.CASCADE_SCALE_IMAGE
			)

			# Draw a rectangle around the faces
			for (x, y, w, h) in faces:
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

			# Display the resulting frame
			cv2.imshow('Video', frame)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

		# When everything is done, release the capture
		video_capture.release()
		cv2.destroyAllWindows()

	def landmarks(self):
		# Vamos inicializar um detector de faces (HOG) para então
		# let's go code an faces detector(HOG) and after detect the 
		# landmarks on this detected face

		# p = our pre-treined model directory, on my case, it's on the same script's diretory.
		p = "shape_predictor_68_face_landmarks.dat"
		
		detector = dlib.get_frontal_face_detector()
		predictor = dlib.shape_predictor(p)

		cap = cv2.VideoCapture(0)
		 
		while True:
			# Getting out image by webcam 
			_, frame = cap.read()
			# Converting the image to gray scale
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				
			# Get faces into webcam's image
			rects = detector(gray, 0)
			
			# For each detected face, find the landmark.
			for (i, rect) in enumerate(rects):
				# Make the prediction and transfom it to numpy array
				shape = predictor(gray, rect)
				shape = face_utils.shape_to_np(shape)
			
				# Draw on our image, all the finded cordinate points (x,y)
				count = 1
				for (x, y) in shape:
					cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
					#print(count)
					if count > 36 and count < 49:
						cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
					elif count > 27 and count < 37:
						cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)
					elif count > 48:
						cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
					count = count + 1
					
			# Show the image
			cv2.imshow("Output", frame)
			
			k = cv2.waitKey(5) & 0xFF
			if k == 27:
				break

		cv2.destroyAllWindows()
		cap.release()

	def initUI(self):
		self.setWindowTitle(self.title)
		self.setGeometry(self.left, self.top, self.width, self.height)
		QApplication.setStyle(QtWidgets.QStyleFactory.create('Windows'))
		
		self.setStyleSheet("QPushButton { background-color: gray; }\n"
              "QPushButton:enabled { background-color: green; }\n");
		# Buttons 
		button = QPushButton('Initialize', self)
		button.setToolTip('This is an example button')
		button.move(100,140)
		button.clicked.connect(self.on_click)
	 
		# Labels
		self.openMouthlbl = QLabel("Open Mouth", self)
		self.eyebrowslbl = QLabel("Eyebrows", self)
		self.smilelbl = QLabel("Smile", self)
		self.openMouthlbl.move(65, 50)
		self.eyebrowslbl.move(175, 50)
		self.smilelbl.move(290, 50)
		
		# Checkboxes
		# Open Mouth
		self.b = QCheckBox("",self)
		self.b.stateChanged.connect(self.clickBox)
		self.b.move(93, 69)
		self.b.resize(320,40)
		# Eyebrows
		self.g = QCheckBox("",self)
		self.g.stateChanged.connect(self.clickBox)
		self.g.move(193, 69)
		self.g.resize(320,40)
		# Smile
		self.g = QCheckBox("",self)
		self.g.stateChanged.connect(self.clickBox)
		self.g.move(293, 69)
		self.g.resize(320,40)
		
		self.styleChoice = QLabel("Windows Vista", self)
		#Combobox
		comboBox = QtWidgets.QComboBox(self)
		comboBox.addItem("motif")
		comboBox.addItem("Windows")
		comboBox.addItem("cde")
		comboBox.addItem("Plastique")
		comboBox.addItem("Cleanlooks")
		comboBox.addItem("windowsvista")
		comboBox.move(50, 100)
		
		comboBox2 = QtWidgets.QComboBox(self)
		comboBox2.addItem("motif")
		comboBox2.addItem("Windows")
		comboBox2.addItem("cde")
		comboBox2.addItem("Plastique")
		comboBox2.addItem("Cleanlooks")
		comboBox2.addItem("windowsvista")
		comboBox2.move(150, 100)

		comboBox3 = QtWidgets.QComboBox(self)
		comboBox3.addItem("motif")
		comboBox3.addItem("Windows")
		comboBox3.addItem("cde")
		comboBox3.addItem("Plastique")
		comboBox3.addItem("Cleanlooks")
		comboBox3.addItem("windowsvista")
		comboBox3.move(250, 100)
		
		self.styleChoice.move(50,180)
		comboBox.activated[str].connect(self.style_choice)

		self.show()
	def style_choice(self, text):
		self.styleChoice.setText(text)

	def clickBox(self, state):
		if state == QtCore.Qt.Checked:
			print('Checked')
		else:
			print('Unchecked')
		
	@pyqtSlot()
	def on_click(self):
		print('PyQt5 button click')

		
if __name__ == '__main__':
	app = QApplication(sys.argv)
	ex = App()
	sys.exit(app.exec_())