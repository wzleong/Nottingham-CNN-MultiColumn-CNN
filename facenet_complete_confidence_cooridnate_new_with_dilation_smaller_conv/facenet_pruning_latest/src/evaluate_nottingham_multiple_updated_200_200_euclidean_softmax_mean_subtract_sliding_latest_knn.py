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

#import multiprocessing
#import collections
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
save = int(sys.argv[3])

import numpy as np 
#import Image

# use grayscale output
#plt.rcParams['image.cmap'] = 'gray'

import sys
#caffe_root = '/home/wzleong/caffe'
#sys.path.insert(0, caffe_root + 'python')
import caffe

from sklearn.cluster import KMeans

pl.rcParams['legend.fontsize'] = '9'


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

def read_annotations(eye_path):
	with open(eye_path,'r') as f: 
		for i, line in enumerate(f):
			#print line
			if i == 1:
				annotations = line.split()
				eye_coordinates = [float(i) for i in annotations[0:4]]
				nose_coordinates = [float(i) for i in annotations[4:6]]
				mouth_coordinates = [float(i) for i in annotations[6:12]]

	return eye_coordinates, nose_coordinates, mouth_coordinates


# set Caffe to run in GPU mode 
#caffe.set_device(0)
#caffe.set_mode_gpu()

# uncomment to set Caffe to run in CPU mode 
caffe.set_mode_cpu()

print "ok"

model_def_prototxt = '/home/wzleong/caffe/examples/nottingham_pruning_latest/bioid_deploy_only_one_prune.prototxt'
model_weights = '/home/wzleong/caffe/examples/nottingham_pruning_latest/_iter_'+iteration+'.caffemodel'


def run_network(model_def_prototxt, model_weights, file_name, stride_no, accuracy_test):

	left_eye_coordinate_count = 0
	right_eye_coordinate_count = 0
	nose_coordinate_count =0
	left_mouth_coordinate_count = 0
	center_mouth_coordinate_count = 0
	right_mouth_coordinate_count=0


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

	height_img, width_img, channel = test_image.shape
	# first plot of input image

	# second plot of images with estimated locations of the eye applying the 'sliding' approach

	count_coordinate_knn = 0
	cluster_left_eye = cluster_right_eye = cluster_nose = cluster_left_mouth = cluster_center_mouth = cluster_right_mouth = 0
	# cluster_left_eye,cluster_right_eye,cluster_nose,cluster_left_mouth,cluster_center_mouth,cluster_right_mouth = 0

	for i in range(stride_no):
		for j in range(stride_no):
			if i == stride_no-1: 
				start_x = width - 200
			else: 
				start_x = i*((width - 200)/(stride_no-1))
				#start_x = i*((width - 200)/(stride_no-1))
			if j == stride_no-1: 
				start_y = height - 200
			else:
				start_y = j*((height - 200)/(stride_no-1))
			#print start_x, start_y
			input_image = test_image[ start_y : start_y + 200, start_x : start_x + 200]
			#print input_image.shape
	
			net.blobs['data'].data[0] = transformer.preprocess('data', input_image)
			output = net.forward()

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

			left_eye_coordinate = renormalize_coordinates(left_eye_coordinate, i*((width - 200)/(stride_no-1)), j*((height - 200)/(stride_no-1)))
			right_eye_coordinate = renormalize_coordinates(right_eye_coordinate, i*((width - 200)/(stride_no-1)), j*((height - 200)/(stride_no-1)))
			nose_coordinate = renormalize_coordinates(nose_coordinate, i*((width - 200)/(stride_no-1)), j*((height - 200)/(stride_no-1)))
			left_mouth_coordinate = renormalize_coordinates(left_mouth_coordinate, i*((width - 200)/(stride_no-1)), j*((height - 200)/(stride_no-1)))
			center_mouth_coordinate = renormalize_coordinates(center_mouth_coordinate, i*((width - 200)/(stride_no-1)), j*((height - 200)/(stride_no-1)))
			right_mouth_coordinate = renormalize_coordinates(right_mouth_coordinate, i*((width - 200)/(stride_no-1)), j*((height - 200)/(stride_no-1)))

			if left_eye_confidence > 0.95:
				cluster_left_eye = 1
				if count_coordinate_knn == 0:
					count_coordinate_knn = 1
					coordinate_knn = np.array(left_eye_coordinate)
				else: 
					coordinate_knn = np.concatenate((coordinate_knn,left_eye_coordinate), axis=0)
				if accuracy_test and left_eye_coordinate_count == 0 : 
					left_eye_coordinate_compiled = np.array(left_eye_coordinate)
					left_eye_coordinate_count += 1
				if accuracy_test and left_eye_coordinate_count > 0 :  
					left_eye_coordinate_compiled = np.concatenate((left_eye_coordinate_compiled,left_eye_coordinate), axis=0)	
				count_coordinate_knn += 1

			if right_eye_confidence > 0.95:
				cluster_right_eye = 1
				if count_coordinate_knn == 0:
					count_coordinate_knn = 1
					coordinate_knn = np.array(right_eye_coordinate)
				else:
					coordinate_knn = np.concatenate((coordinate_knn,right_eye_coordinate), axis=0)
				if accuracy_test and right_eye_coordinate_count == 0 : 
					right_eye_coordinate_compiled = np.array(right_eye_coordinate)
					right_eye_coordinate_count += 1
				if accuracy_test and right_eye_coordinate_count > 0 :  
					right_eye_coordinate_compiled = np.concatenate((right_eye_coordinate_compiled,right_eye_coordinate), axis=0)
				count_coordinate_knn += 1

			if nose_confidence > 0.95:
				cluster_nose = 1
				if count_coordinate_knn == 0:
					count_coordinate_knn = 1
					coordinate_knn = np.array(c)
				else:
					coordinate_knn = np.concatenate((coordinate_knn,nose_coordinate), axis=0)
				if accuracy_test and nose_coordinate_count == 0 : 
					nose_coordinate_compiled = np.array(nose_coordinate)
					nose_coordinate_count += 1
				if accuracy_test and nose_coordinate_count > 0 :  
					nose_coordinate_compiled = np.concatenate((nose_coordinate_compiled,nose_coordinate), axis=0)
				count_coordinate_knn += 1

			if left_mouth_confidence > 0.95:
				cluster_left_mouth = 1
				if count_coordinate_knn == 0:
					count_coordinate_knn = 1
					coordinate_knn = np.array(left_mouth_coordinate)
				else:
					coordinate_knn = np.concatenate((coordinate_knn,left_mouth_coordinate), axis=0)
				if accuracy_test and left_mouth_coordinate_count == 0 : 
					left_mouth_coordinate_compiled = np.array(left_mouth_coordinate)
					left_mouth_coordinate_count += 1
				if accuracy_test and left_mouth_coordinate_count > 0 :  
					left_mouth_coordinate_compiled = np.concatenate((left_mouth_coordinate_compiled,left_mouth_coordinate), axis=0)
				count_coordinate_knn += 1

			if center_mouth_confidence > 0.95:
				cluster_center_mouth = 1
				if count_coordinate_knn == 0:
					count_coordinate_knn = 1
					coordinate_knn = np.array(center_mouth_coordinate)
				else:
					coordinate_knn = np.concatenate((coordinate_knn,center_mouth_coordinate), axis=0)
				if accuracy_test and center_mouth_coordinate_count == 0 : 
					center_mouth_coordinate_compiled = np.array(center_mouth_coordinate)
					center_mouth_coordinate_count += 1
				if accuracy_test and center_mouth_coordinate_count > 0 :  
					center_mouth_coordinate_compiled = np.concatenate((center_mouth_coordinate_compiled,center_mouth_coordinate), axis=0)
				count_coordinate_knn += 1

			if right_mouth_confidence > 0.95:
				cluster_right_mouth = 1
				if count_coordinate_knn == 0:
					coordinate_knn = np.array(right_mouth_coordinate)
				else:
					coordinate_knn = np.concatenate((coordinate_knn,right_mouth_coordinate), axis=0)
				if accuracy_test and right_mouth_coordinate_count == 0 : 
					right_mouth_coordinate_compiled = np.array(right_mouth_coordinate)
					right_mouth_coordinate_count += 1
				if accuracy_test and right_mouth_coordinate_count > 0 : 
					right_mouth_coordinate_compiled = np.concatenate((right_mouth_coordinate_compiled,right_mouth_coordinate), axis=0)
				count_coordinate_knn += 1

		
	cluster = cluster_left_eye + cluster_right_eye + cluster_nose + cluster_left_mouth + cluster_center_mouth + cluster_right_mouth 
	#print coordinate_knn
	print count_coordinate_knn
	if count_coordinate_knn > 2:
		coordinate_knn = coordinate_knn.reshape(count_coordinate_knn-1,2)


	#pl.scatter(coordinate_knn[:,0],coordinate_knn[:,1])

	KMeans1 = KMeans(n_clusters=cluster)
	KMeans1.fit(coordinate_knn)

	centroids = KMeans1.cluster_centers_
	labels = KMeans1.labels_

	end = time.time()
	time_elapsed = (end - start) * 1000
	#print time_elapsed
	if accuracy_test: 
		print len(left_eye_coordinate_compiled)
		left_eye_coordinate_compiled = left_eye_coordinate_compiled.reshape(len(left_eye_coordinate_compiled)/2,2)
		right_eye_coordinate_compiled = right_eye_coordinate_compiled.reshape(len(right_eye_coordinate_compiled)/2,2)
		nose_coordinate_compiled = nose_coordinate_compiled.reshape(len(nose_coordinate_compiled)/2,2)
		left_mouth_coordinate_compiled = left_mouth_coordinate_compiled.reshape(len(left_mouth_coordinate_compiled)/2,2)
		center_mouth_coordinate_compiled = center_mouth_coordinate_compiled.reshape(len(center_mouth_coordinate_compiled)/2,2)
		right_mouth_coordinate_compiled = right_mouth_coordinate_compiled.reshape(len(right_mouth_coordinate_compiled)/2,2)
		
		labels_string = []
		for centroid_temp in centroids: 
			label_temp = get_label(centroid_temp, left_eye_coordinate_compiled, right_eye_coordinate_compiled, nose_coordinate_compiled, left_mouth_coordinate_compiled, center_mouth_coordinate_compiled, right_mouth_coordinate_compiled)
			labels_string.append(label_temp)
			
		return time_elapsed,coordinate_knn, centroids, labels_string
	else:
		return time_elapsed,coordinate_knn, centroids

#time_elapsed, coordinate_knn, centroids = run_network(model_def_prototxt, model_weights, file_name, 4, 1)

def get_label(centroid_temp, left_eye_coordinate_compiled, right_eye_coordinate_compiled, nose_coordinate_compiled, left_mouth_coordinate_compiled, center_mouth_coordinate_compiled, right_mouth_coordinate_compiled): 

	labels_string_temp = []

# np.linalg.norm --> euclidean distance between arrays
	sum_left_eye = 0 
	if len(left_eye_coordinate_compiled)>0:
		labels_string_temp.append("left_eye")
		for target in left_eye_coordinate_compiled:
			sum_left_eye +=np.linalg.norm(centroid_temp-target)
	sum_right_eye = 0
	if len(right_eye_coordinate_compiled)>0:
		labels_string_temp.append("right_eye")
		for target in right_eye_coordinate_compiled:
			sum_right_eye += np.linalg.norm(centroid_temp-target)
	sum_nose = 0
	if len(nose_coordinate_compiled)>0:
		labels_string_temp.append("nose")
		for target in nose_coordinate_compiled:
			sum_nose += np.linalg.norm(centroid_temp-target)
	sum_left_mouth = 0
	if len(left_mouth_coordinate_compiled)>0:
		labels_string_temp.append("left_mouth")
		for target in left_mouth_coordinate_compiled:
			sum_left_mouth += np.linalg.norm(centroid_temp-target)
	sum_centre_mouth = 0
	if len(center_mouth_coordinate_compiled)>0:
		labels_string_temp.append("centre_mouth")
		for target in center_mouth_coordinate_compiled:
			sum_centre_mouth += np.linalg.norm(centroid_temp-target)
	sum_right_mouth = 0
	if len(right_mouth_coordinate_compiled)>0:
		labels_string_temp.append("right_mouth")
		for target in right_mouth_coordinate_compiled:
			sum_right_mouth += np.linalg.norm(centroid_temp-target)

	return labels_string_temp[np.argmin([sum_left_eye,sum_right_eye, sum_nose, sum_left_mouth, sum_centre_mouth, sum_right_mouth])]

def distance_error_per_detection(centroids, labels, eye_coordinates, nose_coordinates, mouth_coordinates):

# np.linalg.norm --> euclidean distance between arrays
	distance_error = []
	mean_error_count = 0
	mean_error = 0
	if "left_eye" in labels:
		left_eye_distance = np.linalg.norm(centroids[labels.index("left_eye")]-eye_coordinates[0:2])
		print left_eye_distance
		distance_error.append(left_eye_distance)
		mean_error += left_eye_distance
		mean_error_count += 1
	else:
		distance_error.append(None)
	if "right_eye" in labels:
		right_eye_distance = np.linalg.norm(centroids[labels.index("right_eye")]-eye_coordinates[2:4])
		distance_error.append(right_eye_distance)
		mean_error += right_eye_distance
		mean_error_count += 1
	else:
		distance_error.append(None)
	if "nose" in labels:
		nose_distance = np.linalg.norm(centroids[labels.index("nose")]-nose_coordinates)
		distance_error.append(nose_distance)
		mean_error += nose_distance
		mean_error_count += 1
	else:
		distance_error.append(None)
	if "left_mouth" in labels:
		left_mouth_distance = np.linalg.norm(centroids[labels.index("left_mouth")]-mouth_coordinates[0:2])
		distance_error.append(left_mouth_distance)
		mean_error += left_mouth_distance
		mean_error_count += 1
	else:
		distance_error.append(None)
	if "centre_mouth" in labels:
		centre_mouth_distance = np.linalg.norm(centroids[labels.index("centre_mouth")]-mouth_coordinates[2:4])
		distance_error.append(centre_mouth_distance)
		mean_error += centre_mouth_distance
		mean_error_count += 1
	else:
		distance_error.append(None)
	if "right_mouth" in labels:
		right_mouth_distance = np.linalg.norm(centroids[labels.index("right_mouth")]-mouth_coordinates[4:6])
		distance_error.append(right_mouth_distance)
		mean_error += right_mouth_distance
		mean_error_count += 1
	else:
		distance_error.append(None)

	mean_error = mean_error/mean_error_count

	return distance_error, mean_error

	

time_elapsed_compiled = []
distance_error_compiled = []
mean_error_compiled = []


for i in range(4,9):

	time_elapsed,coordinate_knn, centroids, labels_string = run_network(model_def_prototxt, model_weights, file_name, i, 1)
	time_elapsed_compiled.append(time_elapsed)
	print labels_string

	eye_coordinates, nose_coordinates, mouth_coordinates = read_annotations(file_name.replace(".jpg",".eye"))
	distance_error, mean_error = distance_error_per_detection(centroids, labels_string, eye_coordinates, nose_coordinates, mouth_coordinates)
	mean_error_compiled.append(mean_error)
	distance_error_compiled = np.concatenate((distance_error_compiled,distance_error), axis=0)
size = len(distance_error_compiled)
distance_error_compiled = distance_error_compiled.reshape(size/6,6)

mean_error_compiled = [i*100 for i in mean_error_compiled]
print mean_error_compiled


#print time_elapsed_compiled

#print multiprocessing.cpu_count()


stride_no_compiled = range(4,9)

z = np.polyfit(stride_no_compiled, time_elapsed_compiled, 2)
print z
poly_eqn = np.poly1d(z)

time_elapsed_plot, = pl.plot(stride_no_compiled, time_elapsed_compiled, label = "Plot of inference time required (ms)")
time_elapsed_plot_approx, = pl.plot(stride_no_compiled, poly_eqn(stride_no_compiled), label = "Second order approximation of prediction time: \n" + str(round(z[2],1))+" x^2 + " + str(round(z[1],1))+" x + " + str(round(z[0],1)), marker='o', linewidth=0.5)
distance_left_eye_plot, = pl.plot(stride_no_compiled, distance_error_compiled[:,0]*100, label = "Plot of euclidean distance error of left eye prediction * 100", linestyle='--', linewidth=1)
distance_right_eye_plot, = pl.plot(stride_no_compiled, distance_error_compiled[:,1]*100, label = "Plot of euclidean distance error of right eye prediction * 100", linestyle='--', linewidth=1)
distance_nose_plot, = pl.plot(stride_no_compiled, distance_error_compiled[:,2]*100, label = "Plot of euclidean distance error of nose prediction * 100", linestyle='--', linewidth=0.5)
distance_left_mouth_plot, = pl.plot(stride_no_compiled, distance_error_compiled[:,3]*100, label = "Plot of euclidean distance error of left mouth prediction * 100", linestyle='--', linewidth=0.5)
distance_centre_mouth_plot, = pl.plot(stride_no_compiled, distance_error_compiled[:,4]*100, label = "Plot of euclidean distance error of centre mouth prediction * 100", linestyle='--', linewidth=0.5)
distance_right_mouth_plot, = pl.plot(stride_no_compiled, distance_error_compiled[:,5]*100, label = "Plot of euclidean distance error of right mouth prediction * 100", linestyle='--', linewidth=0.5)
distance_mean_error, = pl.plot(stride_no_compiled, mean_error_compiled, label = "Plot of mean euclidean distance error of all six facial landmarks * 100", color = 'r')
pl.xlabel('Number of strides', size = 16)
#pl.title('Plot to deduce the optimum stride number')
first_legend = pl.legend(handles=[time_elapsed_plot, time_elapsed_plot_approx, distance_left_eye_plot, distance_right_eye_plot, distance_nose_plot, distance_left_mouth_plot, distance_centre_mouth_plot, distance_right_mouth_plot, distance_mean_error], loc = 'upper left')
pl.tight_layout()
file_name_string = get_filename(file_name)
pl.title('Evaluation of Sliding Window Facial Landmark Localisation \n At Iteration '+iteration+' File: '+file_name_string)
#print save
if save:
	pl.savefig('time_elapsed_'+iteration+'_'+file_name_string+'.jpg',dpi=1000)
else:
	pl.show()





