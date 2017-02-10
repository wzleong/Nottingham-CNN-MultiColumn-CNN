##################################
#
#	Command to run: python rename_fc.py 66000  ==> 	python rename_fc.py <caffe model iteration number>
#
####################################

import numpy as np 
import matplotlib.pyplot as pl
import matplotlib.image as mpimg
#import Image
import math
import cv2

# use grayscale output
#plt.rcParams['image.cmap'] = 'gray'

import sys
#caffe_root = '/home/wzleong/caffe'
#sys.path.insert(0, caffe_root + 'python')
import caffe
from shutil import copyfile
import os

iteration = sys.argv[1]

pl.rc('xtick', labelsize=10) 
pl.rc('ytick', labelsize=10) 

if os.path.isfile('/home/wzleong/caffe/examples/nottingham/filter1_9_filter2_7_filter3_7_dropout1_xavier_average/_iter_10000.caffemodel'):
	print 'BIOID found'
else: 
	print 'Error'


def get_coordinate(coordinates, x_max, y_max):
	for i, value in enumerate(coordinates):
		if i%2 == 0: 
			coordinates[i] = float(coordinates[i])*(x_max)
		else: 
			coordinates[i] = float(coordinates[i])*(y_max)
	return coordinates

def get_filename(file_path):
	path, file_name = os.path.split(file_path)
	file_name = file_name.rsplit('.', 1)[0]
	return file_name	

# set Caffe to run in GPU mode 
#caffe.set_device(0)
#caffe.set_mode_gpu()

# uncomment to set Caffe to run in CPU mode 
caffe.set_mode_cpu()

print "ok"

model_def_prototxt = '/home/wzleong/caffe/examples/nottingham_pruning_latest/bioid_deploy_only_one_prune_previous.prototxt'
model_weights = '/home/wzleong/caffe/examples/nottingham_pruning_latest/_iter_'+iteration+'.caffemodel'


#defines the CNN network 
net = caffe.Net(model_def_prototxt, model_weights, caffe.TEST)		

#copy file	==> copyfile(source,destination)
copyfile('/home/wzleong/caffe/examples/nottingham_pruning_latest/bioid_deploy_only_one_prune_previous.prototxt', '/home/wzleong/caffe/examples/nottingham_pruning_latest/bioid_deploy_copy_for_pruning.prototxt')

with open('/home/wzleong/caffe/examples/nottingham_pruning_latest/bioid_deploy_copy_for_pruning.prototxt','w') as new_file:
        with open('/home/wzleong/caffe/examples/nottingham_pruning_latest/bioid_deploy_only_one_prune_previous.prototxt') as old_file:
            for line in old_file:
		if 'fc' in line:
			line = line.rsplit('\"', 1)
			line = "_temp\"".join(line)
               		new_file.write(line)
		else:
			new_file.write(line)
old_file.close()
new_file.close()

# refer to template file without fully-connected layers (to prune fully-connected layers)

model_def_prototxt = '/home/wzleong/caffe/examples/nottingham_pruning_latest/bioid_deploy_copy_for_pruning.prototxt'
model_weights = '/home/wzleong/caffe/examples/nottingham_pruning_latest/_iter_'+iteration+'.caffemodel'

newnet = caffe.Net(model_def_prototxt,model_weights, caffe.TEST)	

layer_names_whole_network = dict.keys(newnet.params)

layer_names_only_fc = []
for s in layer_names_whole_network:
	if not 'conv' in s:
            layer_names_only_fc.append(s)

# sort to work on all FC1 then FC2
layer_names_only_fc.sort()
print layer_names_only_fc

for layer_name_new in layer_names_only_fc:
	layer_name_old = layer_name_new.replace("_temp","")
	#print layer_name_new
	#print layer_name_old
	newnet.params[layer_name_new][0].data[...] = net.params[layer_name_old][0].data[...]	# copy weights
	newnet.params[layer_name_new][1].data[...] = net.params[layer_name_old][1].data[...]	# copy bias

newnet.save('_iter_rename_caffemodel.caffemodel')

# Debug (good working)
#print newnet.params['fc1_right_mouth_coordinate_temp'][1].data











