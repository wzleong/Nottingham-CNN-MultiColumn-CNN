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


###
###	Command to execute: python plot_points_nottingham_multiple_updated_200_200_euclidean_softmax_mean_subtract_sliding_latest_knn.py 1000 ~/Downloads/nottingham_with_scaling/test/f005.jpg 6 1
###
'''

from mpl_toolkits.axes_grid1 import make_axes_locatable
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

from sklearn.cluster import KMeans

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


pl.rc('xtick', labelsize=7)
pl.rc('ytick', labelsize=7)

# set Caffe to run in GPU mode 
#caffe.set_device(0)
#caffe.set_mode_gpu()

# uncomment to set Caffe to run in CPU mode 
caffe.set_mode_cpu()

print "ok"

model_def_prototxt = '/home/wzleong/caffe/examples/nottingham_pruning_latest/bioid_deploy_only_one_prune.prototxt'
model_weights = '/home/wzleong/caffe/examples/nottingham_pruning_latest/_iter_'+iteration+'.caffemodel'


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
height_img, width_img = img.shape
# first plot of input image
ax1=pl.subplot(141)
pl.imshow(img, cmap='gray' )
ylims1 = [0, height_img]
ax1.set_yticks(ylims1)

xlims1 = [0, width_img] 
ax1.set_xticks(xlims1)
# second plot of images with estimated locations of the eye applying the 'sliding' approach
ax2=pl.subplot(142)
pl.imshow(img, cmap='gray' )

count_coordinate_knn = 0
cluster_left_eye = cluster_right_eye = cluster_nose = cluster_left_mouth = cluster_center_mouth = cluster_right_mouth = 0
# cluster_left_eye,cluster_right_eye,cluster_nose,cluster_left_mouth,cluster_center_mouth,cluster_right_mouth = 0

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
		#pl.imshow(img[ start_y : start_y + 200, start_x : start_x + 200], cmap='gray' )

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

		left_eye_coordinate = renormalize_coordinates(left_eye_coordinate, i*((width - 200)/stride_no), j*((height - 200)/stride_no))
		right_eye_coordinate = renormalize_coordinates(right_eye_coordinate, i*((width - 200)/stride_no), j*((height - 200)/stride_no))
		nose_coordinate = renormalize_coordinates(nose_coordinate, i*((width - 200)/stride_no), j*((height - 200)/stride_no))
		left_mouth_coordinate = renormalize_coordinates(left_mouth_coordinate, i*((width - 200)/stride_no), j*((height - 200)/stride_no))
		center_mouth_coordinate = renormalize_coordinates(center_mouth_coordinate, i*((width - 200)/stride_no), j*((height - 200)/stride_no))
		right_mouth_coordinate = renormalize_coordinates(right_mouth_coordinate, i*((width - 200)/stride_no), j*((height - 200)/stride_no))
		
		#print left_confidence, 'left', np.argmax(left_confidence)
		#print right_confidence, 'right', np.argmax(right_confidence)

		pl.scatter(left_eye_coordinate[0], left_eye_coordinate[1], marker='x', s=30)
		pl.scatter(right_eye_coordinate[0], right_eye_coordinate[1], marker='x', s=30)
		pl.scatter(nose_coordinate[0], nose_coordinate[1], marker='x', s=30)
		pl.scatter(left_mouth_coordinate[0], left_mouth_coordinate[1], marker='x', s=30)
		pl.scatter(center_mouth_coordinate[0], center_mouth_coordinate[1], marker='x', s=30)
		pl.scatter(right_mouth_coordinate[0], right_mouth_coordinate[1], marker='x', s=30)
		#pl.show()

		# for testing purpose
		# pl.scatter(left_eye[0], left_eye[1], marker='x', s=30)
		# pl.scatter(right_eye[0], right_eye[1], marker='x', s=30)

		#print 'left', np.argmax(left_confidence), left_confidence
		#print 'right', np.argmax(right_confidence), right_confidence
		#print output_result
		# plotting all the estimated locations 
		# (might include wrong estimation because the trained model will slide through multiple locations of the image - including patches without eye)

		# print 'y:', count_y + j*((height - 200)/stride_no) 
		# print 'x:', count_x + j*((width - 200)/stride_no)
		
		if i == 0 and j == 0:
			left_eye_coordinate_compiled = np.array(left_eye_coordinate)
			right_eye_coordinate_compiled = np.array(right_eye_coordinate)
			nose_coordinate_compiled = np.array(nose_coordinate)
			left_mouth_coordinate_compiled = np.array(left_mouth_coordinate)
			center_mouth_coordinate_compiled = np.array(center_mouth_coordinate)
			right_mouth_coordinate_compiled = np.array(right_mouth_coordinate)

			left_eye_confidence_compiled = np.array(left_eye_confidence)
			right_eye_confidence_compiled = np.array(right_eye_confidence)
			nose_confidence_compiled = np.array(nose_confidence)
			left_mouth_confidence_compiled = np.array(left_mouth_confidence)
			center_mouth_confidence_compiled = np.array(center_mouth_confidence)
			right_mouth_confidence_compiled = np.array(right_mouth_confidence)

			#output_result_compiled = output_result (WRONG will store the 2nd prediction twice)
		else: 
			left_eye_coordinate_compiled = np.concatenate((left_eye_coordinate_compiled,left_eye_coordinate), axis = 0)
			right_eye_coordinate_compiled = np.concatenate((right_eye_coordinate_compiled,right_eye_coordinate), axis = 0)
			nose_coordinate_compiled = np.concatenate((nose_coordinate_compiled,nose_coordinate), axis = 0)
			left_mouth_coordinate_compiled = np.concatenate((left_mouth_coordinate_compiled,left_mouth_coordinate), axis = 0)
			center_mouth_coordinate_compiled = np.concatenate((center_mouth_coordinate_compiled,center_mouth_coordinate), axis = 0)
			right_mouth_coordinate_compiled = np.concatenate((right_mouth_coordinate_compiled,right_mouth_coordinate), axis = 0)

			left_eye_confidence_compiled = np.concatenate((left_eye_confidence_compiled,left_eye_confidence), axis = 0)
			right_eye_confidence_compiled = np.concatenate((right_eye_confidence_compiled,right_eye_confidence), axis = 0)
			nose_confidence_compiled = np.concatenate((nose_confidence_compiled,nose_confidence), axis = 0)
			left_mouth_confidence_compiled = np.concatenate((left_mouth_confidence_compiled,left_mouth_confidence), axis = 0)
			center_mouth_confidence_compiled = np.concatenate((center_mouth_confidence_compiled,center_mouth_confidence), axis = 0)
			right_mouth_confidence_compiled = np.concatenate((right_mouth_confidence_compiled,right_mouth_confidence), axis = 0)

		if left_eye_confidence > 0.95:
			cluster_left_eye = 1
			if count_coordinate_knn == 0:
				count_coordinate_knn = 1
				coordinate_knn = np.array(left_eye_coordinate)
			else: 
				coordinate_knn = np.concatenate((coordinate_knn,left_eye_coordinate), axis=0)
			count_coordinate_knn += 1

		if right_eye_confidence > 0.95:
			cluster_right_eye = 1
			if count_coordinate_knn == 0:
				count_coordinate_knn = 1
				coordinate_knn = np.array(right_eye_coordinate)
			else:
				coordinate_knn = np.concatenate((coordinate_knn,right_eye_coordinate), axis=0)
			count_coordinate_knn += 1

		if nose_confidence > 0.95:
			cluster_nose = 1
			if count_coordinate_knn == 0:
				count_coordinate_knn = 1
				coordinate_knn = np.array(nose_coordinate)
			else:
				coordinate_knn = np.concatenate((coordinate_knn,nose_coordinate), axis=0)
			count_coordinate_knn += 1

		if left_mouth_confidence > 0.95:
			cluster_left_mouth = 1
			if count_coordinate_knn == 0:
				count_coordinate_knn = 1
				coordinate_knn = np.array(left_mouth_coordinate)
			else:
				coordinate_knn = np.concatenate((coordinate_knn,left_mouth_coordinate), axis=0)
			count_coordinate_knn += 1

		if center_mouth_confidence > 0.95:
			cluster_center_mouth = 1
			if count_coordinate_knn == 0:
				count_coordinate_knn = 1
				coordinate_knn = np.array(center_mouth_coordinate)
			else:
				coordinate_knn = np.concatenate((coordinate_knn,center_mouth_coordinate), axis=0)
			count_coordinate_knn += 1

		if right_mouth_confidence > 0.95:
			cluster_right_mouth = 1
			if count_coordinate_knn == 0:
				coordinate_knn = np.array(right_mouth_coordinate)
			else:
				coordinate_knn = np.concatenate((coordinate_knn,right_mouth_coordinate), axis=0)
			count_coordinate_knn += 1

		
cluster = cluster_left_eye + cluster_right_eye + cluster_nose + cluster_left_mouth + cluster_center_mouth + cluster_right_mouth 
coordinate_knn = coordinate_knn.reshape(count_coordinate_knn-1,2)

left_eye_coordinate_compiled = left_eye_coordinate_compiled.reshape(stride_no*stride_no,2)
right_eye_coordinate_compiled = right_eye_coordinate_compiled.reshape(stride_no*stride_no,2)
nose_coordinate_compiled = nose_coordinate_compiled.reshape(stride_no*stride_no,2)
left_mouth_coordinate_compiled = left_mouth_coordinate_compiled.reshape(stride_no*stride_no,2)
center_mouth_coordinate_compiled = center_mouth_coordinate_compiled.reshape(stride_no*stride_no,2)
right_mouth_coordinate_compiled = right_mouth_coordinate_compiled.reshape(stride_no*stride_no,2)


left_eye_confidence_compiled = left_eye_confidence_compiled.reshape(stride_no*stride_no,1)
right_eye_confidence_compiled = right_eye_confidence_compiled.reshape(stride_no*stride_no,1)
nose_confidence_compiled = nose_confidence_compiled.reshape(stride_no*stride_no,1)
left_mouth_confidence_compiled = left_mouth_confidence_compiled.reshape(stride_no*stride_no,1)
center_mouth_confidence_compiled = center_mouth_confidence_compiled.reshape(stride_no*stride_no,1)
right_mouth_confidence_compiled = right_mouth_confidence_compiled.reshape(stride_no*stride_no,1)

ylims2 = [0, height_img]
ax2.set_yticks(ylims2)

xlims2 = [0, width_img] 
ax2.set_xticks(xlims2)

end = time.time()
time_elapsed = (end - start) * 1000
print time_elapsed



'''
########################################################################
## Scipy example to test the operation of pdist
########################################################################
testing = np.array([[1,2],[3,4],[5,6]])
Y = sc.spatial.distance.pdist(testing, 'euclidean')
print Y
Y = sc.spatial.distance.squareform(Y)
print Y
'''


#print left_confidence_compiled.shape
ax3=pl.subplot(143)
pl.imshow(img, cmap='gray' )

count_left = 0
count_right = 0


keydict_left_eye = dict(zip(left_eye_confidence_compiled[:,0], left_eye_coordinate_compiled))
keydict_left_eye = {key: value for key, value in keydict_left_eye.items() if key >0.95}
if len(keydict_left_eye)>0:
	keydict_left_eye = collections.OrderedDict(sorted(keydict_left_eye.items()))

keydict_right_eye = dict(zip(right_eye_confidence_compiled[:,0], right_eye_coordinate_compiled))
keydict_right_eye = {key: value for key, value in keydict_right_eye.items() if key >0.95}
if len(keydict_right_eye)>0:
	keydict_right_eye = collections.OrderedDict(sorted(keydict_right_eye.items()))

keydict_nose = dict(zip(nose_confidence_compiled[:,0], nose_coordinate_compiled))
keydict_nose = {key: value for key, value in keydict_nose.items() if key >0.95}
if len(keydict_nose)>0:
	keydict_nose = collections.OrderedDict(sorted(keydict_nose.items()))

keydict_left_mouth = dict(zip(left_mouth_confidence_compiled[:,0], left_mouth_coordinate_compiled))
keydict_left_mouth = {key: value for key, value in keydict_left_mouth.items() if key >0.95}
if len(keydict_left_mouth)>0:
	keydict_left_mouth = collections.OrderedDict(sorted(keydict_left_mouth.items()))

keydict_center_mouth = dict(zip(center_mouth_confidence_compiled[:,0], center_mouth_coordinate_compiled))
keydict_center_mouth = {key: value for key, value in keydict_center_mouth.items() if key >0.95}
if len(keydict_center_mouth)>0:
	keydict_center_mouth = collections.OrderedDict(sorted(keydict_center_mouth.items()))

keydict_right_mouth = dict(zip(right_mouth_confidence_compiled[:,0], right_mouth_coordinate_compiled))
keydict_right_mouth = {key: value for key, value in keydict_right_mouth.items() if key >0.95}
if len(keydict_right_mouth)>0:
	keydict_right_mouth = collections.OrderedDict(sorted(keydict_right_mouth.items()))


if len(keydict_left_eye)>0:
	min_color = keydict_left_eye.keys()[0] 
else:
	min_color = 9999
if len(keydict_right_eye)>0:
	min_color = min(min_color, keydict_right_eye.keys()[0])
if len(keydict_nose)>0:
	min_color = min(min_color, keydict_nose.keys()[0])
if len(keydict_left_mouth)>0:
	min_color = min(min_color, keydict_left_mouth.keys()[0])
if len(keydict_center_mouth)>0:
	min_color = min(min_color, keydict_center_mouth.keys()[0])
if len(keydict_right_mouth)>0:
	min_color = min(min_color, keydict_right_mouth.keys()[0])

if len(keydict_left_eye)>0:
	max_color = keydict_left_eye.keys()[len(keydict_left_eye)-1] 
else: 
	max_color = 0
if len(keydict_right_eye)>0:
	max_color = max(max_color, keydict_right_eye.keys()[len(keydict_right_eye)-1])
if len(keydict_nose)>0:
	max_color = max(max_color, keydict_nose.keys()[len(keydict_nose)-1])
if len(keydict_left_mouth)>0:
	max_color = max(max_color, keydict_left_mouth.keys()[len(keydict_left_mouth)-1])
if len(keydict_center_mouth)>0:
	max_color = max(min_color, keydict_center_mouth.keys()[len(keydict_center_mouth)-1])
if len(keydict_right_mouth)>0:
	max_color = max(max_color, keydict_right_mouth.keys()[len(keydict_right_mouth)-1])


'''
min_color = min(keydict_left_eye.keys()[0], keydict_right_eye.keys()[0], keydict_nose.keys()[0], keydict_left_mouth.keys()[0], keydict_center_mouth.keys()[0], keydict_right_mouth.keys()[0])
max_color = max(keydict_left_eye.keys()[len(keydict_left_eye)-1], keydict_right_eye.keys()[len(keydict_right_eye)-1],keydict_nose.keys()[len(keydict_nose)-1], keydict_left_mouth.keys()[len(keydict_left_mouth)-1],keydict_center_mouth.keys()[len(keydict_center_mouth)-1], keydict_right_mouth.keys()[len(keydict_right_mouth)-1])

'''

if len(keydict_left_eye)>0:
	for i in range(0,len(keydict_left_eye)):
		pl.scatter(keydict_left_eye.values()[i][0], keydict_left_eye.values()[i][1], marker='x', s=5, c=keydict_left_eye.keys()[i], vmin=min_color, vmax=max_color)
if len(keydict_right_eye)>0:
	for i in range(0,len(keydict_right_eye)):
		pl.scatter(keydict_right_eye.values()[i][0], keydict_right_eye.values()[i][1], marker='x', s=5, c=keydict_right_eye.keys()[i], vmin=min_color, vmax=max_color)
if len(keydict_nose)>0:
	for i in range(0,len(keydict_nose)):
		pl.scatter(keydict_nose.values()[i][0], keydict_nose.values()[i][1], marker='x', s=5, c=keydict_nose.keys()[i], vmin=min_color, vmax=max_color)
if len(keydict_left_mouth)>0:
	for i in range(0,len(keydict_left_mouth)):
		pl.scatter(keydict_left_mouth.values()[i][0], keydict_left_mouth.values()[i][1], marker='x', s=5, c=keydict_left_mouth.keys()[i], vmin=min_color, vmax=max_color)
if len(keydict_center_mouth)>0:
	for i in range(0,len(keydict_center_mouth)):
		pl.scatter(keydict_center_mouth.values()[i][0], keydict_center_mouth.values()[i][1], marker='x', s=5, c=keydict_center_mouth.keys()[i], vmin=min_color, vmax=max_color)
if len(keydict_right_mouth)>0:
	for i in range(0,len(keydict_right_mouth)):
		pl.scatter(keydict_right_mouth.values()[i][0], keydict_right_mouth.values()[i][1], marker='x', s=5, c=keydict_right_mouth.keys()[i], vmin=min_color, vmax=max_color)


divider = make_axes_locatable(ax3)
cax = divider.append_axes("right", size="5%", pad=0.05)
pl.colorbar(cax=cax)

ylims3 = [0, height_img]
ax3.set_yticks(ylims3)

xlims3 = [0, width_img] 
ax3.set_xticks(xlims3)




KMeans = KMeans(n_clusters=cluster)
KMeans.fit(coordinate_knn)

centroids = KMeans.cluster_centers_
labels = KMeans.labels_

#print centroids
#print labels


ax6 = pl.subplot(144)
pl.imshow(img, cmap='gray' )
pl.scatter(centroids[:,0],centroids[:,1], marker="x", s=10)

ylims6 = [0, height_img]
ax6.set_yticks(ylims6)

xlims6 = [0, width_img] 
ax6.set_xticks(xlims6)

file_name = get_filename(file_name)
pl.suptitle('Sliding Window Facial Landmark Localisation \n At Iteration '+iteration+' File: '+file_name)
pl.tight_layout()
if save:
	pl.savefig('sliding_window_with_confidence_level_'+iteration+'_'+file_name+'.jpg',dpi=500)
else:
	pl.show()






