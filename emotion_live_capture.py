######## Imports ########
import os
import cv2
from PIL import ImageGrab

import matplotlib
matplotlib.use("Qt5agg") 
import matplotlib.pyplot as plt
from matplotlib import gridspec

import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

# ######## LOAD MODEL #########
model = model_from_json(open("fer2.json", "r").read()) 
model.load_weights('fer2.h5') #load weights

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #load cascade

######## Global Variables #########
plt.style.use('ggplot')
x_axis = np.arange(100)  # will be used in graphing. 
line1 = [] # will initaite the loop for live plotting.
plot_array = np.full((7,100), 0) #create an array that will be updated with live values.
counter= 1 #used to close every 10th window
max_faces = 4 #sets a max number of faces to be recognized to protect performance.
results_array = np.full((1,8), 0.0, dtype='float64')

######## Will need softmax to properly graph the emotions ########
def softmax(x):
	"""Compute softmax values for each sets of scores in x."""
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()

### This will subplot each emotion in the array. ###
### Called by live_plotter ###
def make_subplots(x, y, colors, labels):
	gs = gridspec.GridSpec(len(y), 1) 
	for i, y_i in enumerate(y):
	# log scale for axis Y of the first subplot
		if i == 0:
			ax0 = plt.subplot(gs[0])
			ax0.set_ylim(0,1)
			line0, = ax0.plot(x, y_i, color=colors[i])
			ax0.legend((line0,), (labels[i],), loc='lower left')
			ax0.axes.yaxis.set_visible(False)

		else:
			ax1 = plt.subplot(gs[i], sharex = ax0)
			ax1.set_ylim(0,1)
			line1, = ax1.plot(x, y_i, color=colors[i], linestyle='--')
			plt.setp(ax0.get_xticklabels(), visible=False)
			yticks = ax1.yaxis.get_major_ticks()
			yticks[-1].label1.set_visible(False)
			ax1.legend((line1,), (labels[i],), loc='lower left')
			ax1.axes.yaxis.set_visible(False)

	plt.subplots_adjust(hspace=.0)
	#plt.show(block=False)
	#plt.get_current_fig_manager().window.wm_geometry("+400+900") 
	plt.show()
	return line0

### This will plot the incoming data live. ###
def live_plotter(x_vec, y1_data, line1, labels, close, identifier='', pause_time=0.5):
	if line1==[]:
		# this is the call to matplotlib that allows dynamic plotting
		if close == True:
			plt.close('all')
		plt.ion()
		fig = plt.figure(figsize=(12,3))
		fig.canvas.manager.window.move(1400, 725)
		colors = ['r', 'b', 'g', 'm', 'k', 'c', 'y']

		line1 = make_subplots(x_vec, y1_data, colors, labels)

	# adjust limits if new data goes beyond bounds
	if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
		plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])
	# this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
	plt.pause(pause_time)

#######  Image function ########
def process_img(image):
	original_image = image
	processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300)
	return processed_img
	close = False

try:
	while True:
		screen = np.array(ImageGrab.grab(bbox=(0,40,1200,1200)))
		RGB_img = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
		new_screen = process_img(screen)
		faces_detected = face_haar_cascade.detectMultiScale(new_screen, 1.2, 5)
		emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
		preds_probs = np.full((1,7), 0, dtype = 'float64')

		#Limit to 4 faces for now for performance. 
		if len(faces_detected) > max_faces:
			faces_detected = faces_detected[0:max_faces]

		for (x,y,w,h) in faces_detected:
			cv2.rectangle(RGB_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
			roi_gray=new_screen[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
			roi_gray=cv2.resize(roi_gray,(48,48))
			#roi_gray = roi_gray.resize(48,48,1)
			img_pixels = image.img_to_array(roi_gray)
			img_pixels = np.expand_dims(img_pixels, axis = 0)
			img_pixels /= 255

			predictions = model.predict(img_pixels) # get predictions
			preds_probs += predictions

			#find max indexed array
			max_index = np.argmax(predictions[0])
			predicted_emotion = emotions[max_index]

			cv2.putText(RGB_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

		resized_img = cv2.resize(RGB_img, (1000, 675))
		cv2.imshow('Facial emotion analysis ',resized_img)

		#Process and Display Predictions.
		softmax_preds = softmax(preds_probs) #softmax predictions
		plot_array[:,-1] = softmax_preds  #make them the last value in the array
		#need to close the windows every 10th time. 
		close = True if counter % 10 == 0 else False
		line2 = live_plotter(x_axis, plot_array, line1, emotions, close) 
		counter += 1
		#need to update the array so it takes one step forward.
		plot_array = np.append(plot_array[:, 1:], np.full((7,1), 0.0), 1)
		#store results 
		results = np.append(softmax_preds, len(faces_detected))
		results_array = np.vstack((results_array, results))


except KeyboardInterrupt:
	cv2.destroyAllWindows()
	
np.savetxt('emotion_results', results_array, delimiter=",")

