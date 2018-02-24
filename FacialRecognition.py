#OpenCV module
import cv2
#os module for reading training data directories and paths
import os
#numpy to convert python lists to numpy arrays as it is needed by OpenCV face recognizers
import numpy as np
from imutils.video import VideoStream
from imutils import face_utils
from datetime import datetime, date, time
import argparse
import imutils
import time
import cv2
import RPi.GPIO as GPIO
from time import sleep
import pygame, sys
from pygame.locals import *
import pygame.camera
import time

locked = False

f = open('Log.txt', 'a')
f.write("\n\n\n\n")
f.write("\n" + "Lock Turning on at ")
f.write(str(datetime.now()))
f.write("\n")
f.close()

#there is no label 0 in our training data so subject name for index/label 0 is empty
subjects = [ ''  , "Ben" , "Samuel"]

def detect_face(img):
	#convert the test image to gray scale as opencv face detector expects gray images
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#load OpenCV face detector, I am using LBP which is fast
	#there is also a more accurate but slow: Haar classifier
	face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
	#print(face_cascade == empty())
	#let's detect multiscale images(some images may be closer to camera than others)
	#result is a list of faces
	#faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
	#if no faces are detected then return original img
	if (len(faces) == 0):
		return None, None
	#under the assumption that there will be only one face,
	#extract the face area
	(x, y, w, h) = faces[0]
	#return only the face part of the image
	return gray[y:y+w, x:x+h], faces[0]


#this function will read all persons' training images, detect face from each image
#and will return two lists of exactly same size, one list 
#of faces and another list of labels for each face
def prepare_training_data(data_folder_path):
	f = open('Log.txt', 'a')
	f.write("Preparing Training Data")
	f.close()
	x = 1 
	#------STEP-1--------
	#get the directories (one directory for each subject) in data folder
	dirs = os.listdir(data_folder_path)

	#list to hold all subject faces
	faces = []
	#list to hold labels for all subjects
	labels = []

	#let's go through each directory and read images within it
	for dir_name in dirs:

		#our subject directories start with letter 's' so
		#ignore any non-relevant directories if any
		if not dir_name.startswith("s"):
			continue;
 
		#------STEP-2--------
		#extract label number of subject from dir_name
		#format of dir name = slabel
		#, so removing letter 's' from dir_name will give us label
		label = int(dir_name.replace("s", ""))
 
		#build path of directory containing images for current subject subject
		#sample subject_dir_path = "training-data/s1"
		subject_dir_path = data_folder_path + "/" + dir_name
 
 		#get the images names that are inside the given subject directory
		subject_images_names = os.listdir(subject_dir_path)
 
		#------STEP-3--------
		#go through each image name, read image, 
		#detect face and add face to list of faces
		for image_name in subject_images_names:
 
			#ignore system files like .DS_Store
			if image_name.startswith("."):
				continue;
 
			#build image path
			#sample image path = training-data/s1/1.pgm
			image_path = subject_dir_path + "/" + image_name
 			print(image_name)
			#read image
			image = cv2.imread(image_path)
 
			#display an image window to show the image 
			#cv2.imshow("Training on image...", image)
			#cv2.waitKey(100)
 
			#detect face
			face, rect = detect_face(image)
 
			#------STEP-4--------
			#for the purpose of this tutorial
			#we will ignore faces that are not detected
			if face is not None:
				#add face to list of faces
				faces.append(face)
				#add label for this face
				labels.append(label)
				print(label)
 
			cv2.destroyAllWindows()
			cv2.waitKey(1)
			cv2.destroyAllWindows()
			print(x)
			x = x+1
			sleep(3)
	sleep(10)
	return faces, labels




#let's first prepare our training data
#data will be in two lists of same size
#one list will contain all the faces
#and the other list will contain respective labels for each face
print("Preparing data...")
faces, labels = prepare_training_data("training_data")

print("Data prepared")
f = open('Log.txt', 'a')
f.write("Data Prepared")
f.close()
#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))


#create our LBPH face recognizer 
#face_recognizer = cv2.face.createLBPHFaceRecognizer()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))

#function to draw rectangle on image 
#according to given (x, y) coordinates and 
#given width and heigh
def draw_rectangle(img, rect, vect):
	(x, y, w, h) = rect
	cv2.rectangle(img, (x, y), (x+w, y+h), vect, 2)
 
#function to draw text on give image starting from
#passed (x, y) coordinates. 
def draw_text(img, text, x, y):
	cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)




#this function recognizes the person in image passed
#and draws a rectangle around detected face with name of the 
#subject
label_pair = (0,0)
def predict(test_img):
	global label_pair
	global face_recognizer
	#make a copy of the image as we don't want to change original image
	#img = test_img.copy()
	img = test_img
	#detect face from the image
	face, rect = detect_face(img)
	if face is not None:
		if len(face) > 0:
			#predict the image using our face recognizer 
			label_pair = face_recognizer.predict(face)
			label = label_pair[0]
			#get name of respective label returned by face recognizer
			if label_pair[1] <45 and label > 0:
				label_text = ""
				draw_rectangle(img, rect, (0,255,0))
			else:
				label_text = ""
				draw_rectangle(img, rect, (0,0,255))
			#draw a rectangle around face detected
			#draw_rectangle(img, rect)
			#draw name of predicted person
			draw_text(img, label_text, rect[0], rect[1]-5)
 
	return img

width = 640
height = 480

video = cv2.VideoCapture(0)
count = 0
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
Motor1A = 16
Motor1B = 20
Motor1E = 21
GPIO.setup(Motor1A,GPIO.OUT)
GPIO.setup(Motor1B,GPIO.OUT)
GPIO.setup(Motor1E,GPIO.OUT)
GPIO.setup(18, GPIO.IN, pull_up_down=GPIO.PUD_UP)
#locked = True

GPIO.output(Motor1A, GPIO.HIGH)
GPIO.output(Motor1B, GPIO.LOW)
GPIO.output(Motor1E, GPIO.LOW)

while True:
	global locked
	frame = video.read()[1]
	#array_frame = pygame.surfarray.array3d(frame)
	predicted_frame = predict(frame)
	#new_frame = imutils.resize(predicted_frame, width=400)

	cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        imS = cv2.resize(predicted_frame, (960, 540)) 
        cv2.imshow("output", imS)
	key = cv2.waitKey(3) & 0xFF

	if label_pair[1] < 45 and label_pair[0] > 0 and locked is True:
		f = open('Log.txt', 'a')
		f.write("\n At ")
		f.write(str(datetime.now()))
		f.write(", ")
		f.write(subjects[label_pair[0]])
		f.write(" is entering the room with a variance of " + str(label_pair[1]))
		f.close()
		GPIO.output(Motor1E,GPIO.HIGH)
		waitTime = time.clock()+10
		while time.clock() < waitTime:
			pass
		GPIO.output(Motor1E, GPIO.LOW)
		locked = False

	input_state = GPIO.input(18)
	if input_state == False:
		f = open('Log.txt', 'a')
		if locked is True:
			GPIO.output(Motor1E,GPIO.HIGH)
                	waitTime = time.clock()+10
			while time.clock() < waitTime:
				pass
                	GPIO.output(Motor1E, GPIO.LOW)
			locked = False
			f.write("\n" + "The button was used to unlock the door at " + str(datetime.now()))
		else:
			sleep(6)
			GPIO.output(Motor1A, GPIO.LOW)
			GPIO.output(Motor1B, GPIO.HIGH)
			GPIO.output(Motor1E,GPIO.HIGH)
	                waitTime = time.clock()+10
			while time.clock() < waitTime:
				pass
        	        GPIO.output(Motor1E, GPIO.LOW)
			GPIO.output(Motor1B, GPIO.LOW)
			GPIO.output(Motor1A, GPIO.HIGH)
			locked = True
			f.write("\n" + "The button was used to lock the door at " + str(datetime.now()))
		f.close()

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	if count == 1000:
		cv2.destroyAllWindows()
		count = 0
	label_pair = (0,0)

video.stop()
cv2.destroyAllWindows()
