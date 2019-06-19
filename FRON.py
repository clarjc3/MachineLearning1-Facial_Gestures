# Author: Joshua Clarke
# NOTE: This program is not intended to be used as the final product, it was used for code testing and development of features for the final product.
# 		Code was copied from here to the final product repo and the latest version can be found at: https://github.com/manulea/MITGUI

import sys
# Libraries for the PyQt5 GUI
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QMainWindow, QCheckBox, QApplication, QWidget, QPushButton, QLabel
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot, QSize

# Libraries for computer vision and image processing
import cv2
import dlib
from imutils import face_utils

# Libraries for calculations
import numpy as np
from collections import deque
import time

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
		# let's go code an faces detector(HOG) and after detect the 
		# landmarks on this detected face

		# p = our pre-treined model directory, on my case, it's on the same script's diretory.
		p = "shape_predictor_68_face_landmarks.dat"
		
		detector = dlib.get_frontal_face_detector()
		predictor = dlib.shape_predictor(p)

		cap = cv2.VideoCapture(0)
			
		gesture_arr = deque(maxlen=20) # This is a list-like container that is used to hold the last 20 gesture recognitions
		gesture_arr.extend([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]) # Setting up the array with all non-gesture numbers
		
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
				# The following code is used to change the colour of the facial landmark points shown on the screen.
				# You can set different parts of the face to different colours if you like.
				count = 1
				for (x, y) in shape:
					#All points (Outline)
					cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
					# Eyebrows
					# if count > 17 and count < 28:
						# cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
					# Eyes
					# elif count > 36 and count < 49:
						# cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
					#Nose
					# elif count > 27 and count < 37:
						# cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
					#Mouth
					# elif count > 48:
						# cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
					
					count = count + 1
				
				# Gesture recognition code is show below
				# Baseline
				base_line = ((shape[16][0]) - (shape[0][0])) # This fixes the issue of fake gesture recognitions when the user gets closer to, or moves away from, the camera. It uses the width of the users head as a proportional reference.
				
				# Open mouth
				mouth_top = ((shape[61][1]) + (shape[62][1]) + (shape[63][1]))/3 # Take an average measurement of the top of the mouth.
				mouth_bottom = ((shape[65][1]) + (shape[66][1]) + (shape[67][1]))/3 # Take an average measurement of the bottom of the mouth.
				mouth_height = mouth_bottom - mouth_top # Calculate the distance between the top and bottom lips, i.e. if the mouth is open or shut.
				if(mouth_height/base_line > 0.18):
					gesture_arr.append(0) # If the calculated distance exceeds the threshold, add an open mouth gesture to the array (0)
				
				# Raise Eyebrow
				eye_top = ((shape[18][1]) + (shape[19][1]) + (shape[20][1]) + (shape[23][1]) + (shape[24][1]) + (shape[25][1]))/6 # Take an average measurement of the eyebrow y value.
				eye_bottom = ((shape[27][1]) + (shape[28][1]))/2 # Take an average measurement of the eye y value.
				eye_height = eye_bottom - eye_top # Calculate the distance between the eyes and the eyebrows.
				if(eye_height/base_line > 0.22):
					gesture_arr.append(1) # If the calculated distance exceeds the threshold, add an eyebrow raise gesture to the array (1).
				
				# Eye shut
				eyelid_top = ((shape[37][1]) + (shape[38][1]) + (shape[43][1]) + (shape[44][1]))/4 # Take an average measurement of the upper eyelid y value.
				eyelid_bottom = ((shape[40][1]) + (shape[41][1]) + (shape[46][1]) + (shape[47][1]))/4 # Take an average measurement of the lower eyelid y value.
				eyelid_height = eyelid_bottom - eyelid_top # Calculate the distance between the eyelids.
				if(eyelid_height/base_line < 0.022):
					gesture_arr.append(2) # If the calculated distance is smaller than the threshold, add an eye close gesture to the array (2).
				
				# Smile
				mouth_left = ((shape[48][0]) + (shape[49][0]) + (shape[59][0]) + (shape[60][0]))/4 # Take an average measurement of the left side of the mouth.
				mouth_right = ((shape[53][0]) + (shape[54][0]) + (shape[55][0]) + (shape[64][0]))/4 # Take an average measurement of the right side of the mouth.
				mouth_width = mouth_right - mouth_left # Calculate the width of the mouth.
				if(mouth_width/base_line > 0.34):
					gesture_arr.append(3) # If the calculated distance exceeds the threshold, add a smile gesture to the array (3).
				
				# Anger
				nose_top = ((shape[21][1]) + (shape[22][1]))/2 # Take an average measurement of the innermost part of the eyebrows (i.e. the part that moves downwards when someone frowns).
				nose_bottom = ((shape[31][1]) + (shape[35][1]))/2 # Take an average measurement of the bottom of the nose. In the future it may be more accurate to use this in combination with an upper lip measurement, as the upper lip usually raises upwards when snarling.
				nose_height = nose_bottom - nose_top # Calculate the distance between the inner eyebrows and the bottom of the nose.
				if(nose_height/base_line < 0.36):
					gesture_arr.append(4) # If the calculated distance is smaller than the threshold, add a snarl gesture to the array (4).
				
				gesture_output = max(set(gesture_arr), key=gesture_arr.count) # Calculate the mode of the gesture array. Used to eliminate noise from the detections.
				
				if(gesture_output == 0):
					print("Mouth opened! - ",(mouth_height/base_line)) # If the most common number in the array is a 0, output a mouth opened gesture recognition
				elif(gesture_output == 1):
					print("Eyebrows raised! - ",(eye_height/base_line)) # If the most common number in the array is a 1, output an eyebrows raised gesture recognition
				elif(gesture_output == 2):
					print("Eye close detected! - ",(eyelid_height/base_line)) # If the most common number in the array is a 2, output an eyes closed gesture recognition
				elif(gesture_output == 3):
					print("Smile detected! - ",(mouth_width/base_line)) # If the most common number in the array is a 3, output a smile gesture recognition
				elif(gesture_output == 4):
					print("Anger detected! - ",(nose_height/base_line)) # If the most common number in the array is a 4, output a snarl gesture recognition
				
				if(gesture_output == 0 or gesture_output == 1 or gesture_output == 2 or gesture_output == 3 or gesture_output == 4): # If any gesture is calculated, reset the gesture array so it can be used to calculate the next gesture
					gesture_arr = deque(maxlen=20) # Recreate the array
					gesture_arr.extend([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]) # Setting all numbers in the array to the non-gesture numbers

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
		QApplication.setStyle(QtWidgets.QStyleFactory.create('Fusion'))
		
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
		
		self.styleChoice = QLabel("Gesture Recognition", self)
		#Combobox
		comboBox = QtWidgets.QComboBox(self)
		comboBox.addItem("a")
		comboBox.addItem("s")
		comboBox.addItem("d")
		comboBox.addItem("f")
		comboBox.move(50, 100)
		
		comboBox2 = QtWidgets.QComboBox(self)
		comboBox2.addItem("a")
		comboBox2.addItem("s")
		comboBox2.addItem("d")
		comboBox2.addItem("f")
		comboBox2.move(150, 100)

		comboBox3 = QtWidgets.QComboBox(self)
		comboBox3.addItem("a")
		comboBox3.addItem("s")
		comboBox3.addItem("d")
		comboBox3.addItem("f")
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