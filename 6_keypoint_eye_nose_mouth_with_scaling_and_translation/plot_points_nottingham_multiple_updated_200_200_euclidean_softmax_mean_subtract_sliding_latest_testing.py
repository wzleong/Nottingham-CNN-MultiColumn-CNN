''' 
Author: Leong Wei Zhen
Description: Python interface to deploy the trained network (Sliding Approach) 
Further description: The trained model <.caffemodel> is deployed (either using only CPU or with GPU). The predicted coordinates of the eyes are plotted with the image.
			
The algorithm flow is described as follows:

1) apply train model to patches of the input image
2) compile the estimated results
3) using scipy python package to get pairs of euclidean distance 
4) find the index of the compiled prediction result array with maximum pair euclidean distance less than 5 pixels (<5)
5) the output of the result then obtained (by separating the process described above for both right and left eye)
	       	
A new approach demonstrating immunity to spatial transformation (namely translation)	
'''

import collections
import matplotlib.pyplot as pl
import matplotlib.image as mpimg
import sys
import os
import time
#import matplotlib.mlab as mlab
import scipy as sc
from scipy import stats
iteration = sys.argv[1]
file_name = sys.argv[2]
stride_no = int(sys.argv[3])
save = int(sys.argv[4])

import numpy as np 
#import Image

# use grayscale output
#plt.rcParams['image.cmap'] = 'gray'

import sys
#caffe_root = '/home/wzleong/caffe'
#sys.path.insert(0, caffe_root + 'python')
import caffe

import os


def get_coordinate(coordinates, x_max, y_max):
	for i, value in enumerate(coordinates):
		if i%2 == 0: 
			coordinates[i] = float(coordinates[i])*(x_max)
		else: 
			coordinates[i] = float(coordinates[i])*(y_max)
	return coordinates


# the estimated results are only from patches of the image, so the coordinates must be shifted back with respect of the original image
def renormalize_coordinates(coordinates, shift_x, shift_y):
	for i, value in enumerate(coordinates):
		if i%2 == 0: 
			coordinates[i] = float(coordinates[i]) + shift_x
		else: 
			coordinates[i] = float(coordinates[i]) + shift_y
	return coordinates

# function to get filename of input image 
# the filename will be used to save the result for better referencing
def get_filename(file_path):
	path, file_name = os.path.split(file_path)
	file_name = file_name.rsplit('.', 1)[0]
	return file_name	


# after applying the mask to remove outliers --> the finalize output will be the average of all the estimated coordinates
def estimate_average(data):
	count = 0 
	accumulate = 0 
	for i, value in enumerate(data):
		if value != 0:
			count += 1
			accumulate += value
			
	return accumulate/count


# set Caffe to run in GPU mode 
#caffe.set_device(0)
#caffe.set_mode_gpu()

# uncomment to set Caffe to run in CPU mode 
caffe.set_mode_cpu()

print "ok"

model_def_prototxt = '/home/wzleong/caffe/examples/nottingham_complete_eye_nose_mouth/bioid_deploy_only_one.prototxt'
model_weights = '/home/wzleong/caffe/examples/nottingham_complete_eye_nose_mouth/_iter_'+iteration+'.caffemodel'


#defines the CNN network 
start = time.time()
net = caffe.Net(model_def_prototxt, model_weights, caffe.TEST)		
#use test mode (not in train mode -> no dropout)
#model_weights, 	#containes the trained weights  

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# caffe reads the image into array of (28,28,1) 
# after transpose will convert the array to (1,28,28)
transformer.set_transpose('data', (2,0,1))
# normalize the valus of the image based on 0-255 range 
#transformer.set_raw_scale('data', 255)

net.blobs['data'].reshape(1,1,200,200)

#test_image = caffe.io.load_image(os.path.join("/home/wzleong/caffe/data/BIO_ID_face/BioID-FaceDatabase-V1.2/BioID_"+file_name+".pgm"), color=False)
test_image = caffe.io.load_image(os.path.join(file_name), color=False)

test_image = test_image.astype(float)
test_image = test_image - np.mean(test_image)

# cafe load image as H x W x N
height, width, number = test_image.shape
print test_image.shape

count_x = count_y = 0


img = mpimg.imread(file_name)
# first plot of input image
#pl.subplot(211)
#pl.imshow(img, cmap='gray' )
# second plot of images with estimated locations of the eye applying the 'sliding' approach
#pl.subplot(212)
#pl.imshow(img, cmap='gray' )


for i in range(stride_no):
	for j in range(stride_no):

		#print 'enter'
		if i == stride_no: 
			start_x = width - 200
		else: 
			start_x = count_x + i*((width - 200)/stride_no)
		if j == stride_no: 
			start_y = height - 200
		else:
			start_y = count_y + j*((height - 200)/stride_no)
		input_image = test_image[ start_y : start_y + 200, start_x : start_x + 200]
		pl.imshow(img[ start_y : start_y + 200, start_x : start_x + 200], cmap='gray' )

		#input_image = test_image[ count_y + j*((height - 200)/stride_no) : count_y +j*((height - 200)/stride_no) + 200, count_x + i*((width - 200)/stride_no) : count_x + i*((width - 200)/stride_no) + 200]
		#print input_image.shape	
		net.blobs['data'].data[0] = transformer.preprocess('data', input_image)
		output = net.forward()

		# the output probability for the first image (the tested image) 
		left_eye_confidence = output['fc2_left_eye_confidence']
		right_eye_confidence = output['fc2_right_eye_confidence']
		nose_confidence = output['fc2_nose_confidence']
		left_mouth_confidence = output['fc2_left_mouth_confidence']
		center_mouth_confidence = output['fc2_center_mouth_confidence']
		right_mouth_confidence = output['fc2_right_mouth_confidence']

		left_eye_coordinate = output['fc2_left_eye_coordinate'][0]
		right_eye_coordinate = output['fc2_right_eye_coordinate'][0]
		nose_coordinate = output['fc2_nose_coordinate'][0]
		left_mouth_coordinate = output['fc2_left_mouth_coordinate'][0]
		center_mouth_coordinate = output['fc2_center_mouth_coordinate'][0]
		right_mouth_coordinate = output['fc2_right_mouth_coordinate'][0]

		left_eye_coordinate = get_coordinate(left_eye_coordinate,200,200)
		right_eye_coordinate = get_coordinate(right_eye_coordinate,200,200)
		nose_coordinate = get_coordinate(nose_coordinate,200,200)
		left_mouth_coordinate = get_coordinate(left_mouth_coordinate,200,200)
		center_mouth_coordinate = get_coordinate(center_mouth_coordinate,200,200)
		right_mouth_coordinate = get_coordinate(right_mouth_coordinate,200,200)
		
		#print left_confidence, 'left', np.argmax(left_confidence)
		#print right_confidence, 'right', np.argmax(right_confidence)

		pl.scatter(left_eye_coordinate[0], left_eye_coordinate[1], marker='x', s=30)
		pl.scatter(right_eye_coordinate[0], right_eye_coordinate[1], marker='x', s=30)
		pl.scatter(nose_coordinate[0], nose_coordinate[1], marker='x', s=30)
		pl.scatter(left_mouth_coordinate[0], left_mouth_coordinate[1], marker='x', s=30)
		pl.scatter(center_mouth_coordinate[0], center_mouth_coordinate[1], marker='x', s=30)
		pl.scatter(right_mouth_coordinate[0], right_mouth_coordinate[1], marker='x', s=30)
		#pl.show()

		if left_eye_confidence > 0.95: 
			print 'left_eye_confidence', left_eye_confidence, 'pass'
			pl.scatter(left_eye_coordinate[0], left_eye_coordinate[1], marker='x', s=50, color = 'blue')
		else: 
			print 'left_eye_confidence', left_eye_confidence, 'fail'
			pl.scatter(left_eye_coordinate[0], left_eye_coordinate[1], marker='x', s=50, color = 'red')

		if right_eye_confidence > 0.95: 
			print 'right_eye_confidence', right_eye_confidence, 'pass'
			pl.scatter(right_eye_coordinate[0], right_eye_coordinate[1], marker='x', s=50, color = 'blue')			
		else:
			print 'right_eye_confidence', right_eye_confidence, 'fail'
			pl.scatter(right_eye_coordinate[0], right_eye_coordinate[1], marker='x', s=50, color = 'red')

		if nose_confidence > 0.95: 
			print 'nose_confidence', nose_confidence, 'pass'
			pl.scatter(nose_coordinate[0], nose_coordinate[1], marker='x', s=50, color = 'blue')
		else: 
			print 'nose_confidence', nose_confidence, 'fail'
			pl.scatter(nose_coordinate[0], nose_coordinate[1], marker='x', s=50, color = 'red')

		if left_mouth_confidence > 0.95: 
			print 'left_mouth_confidence', left_mouth_confidence, 'pass'
			pl.scatter(left_mouth_coordinate[0], left_mouth_coordinate[1], marker='x', s=50, color = 'blue')			
		else:
			print 'left_mouth_confidence', left_mouth_confidence, 'fail'
			pl.scatter(left_mouth_coordinate[0], left_mouth_coordinate[1], marker='x', s=50, color = 'red')

		if center_mouth_confidence > 0.95: 
			print 'center_mouth_confidence', center_mouth_confidence, 'pass'
			pl.scatter(center_mouth_coordinate[0], center_mouth_coordinate[1], marker='x', s=50, color = 'blue')
		else: 
			print 'center_mouth_confidence', center_mouth_confidence, 'fail'
			pl.scatter(center_mouth_coordinate[0], center_mouth_coordinate[1], marker='x', s=50, color = 'red')

		if right_mouth_confidence > 0.95: 
			print 'right_mouth_confidence', right_mouth_confidence, 'pass'
			pl.scatter(right_mouth_coordinate[0], right_mouth_coordinate[1], marker='x', s=50, color = 'blue')			
		else:
			print 'right_mouth_confidence', right_mouth_confidence, 'fail'
			pl.scatter(right_mouth_coordinate[0], right_mouth_coordinate[1], marker='x', s=50, color = 'red')

		print "************************************************************************************"
		pl.pause(2)	
		pl.close()

pl.ion()





