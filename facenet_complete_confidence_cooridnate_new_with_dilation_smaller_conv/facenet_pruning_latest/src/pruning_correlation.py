################################################################
#
#	Command to run: python pruning_correlation.py rename_caffemodel
#
################################################################

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

#print "ok"

model_def_prototxt = '/home/wzleong/caffe/examples/nottingham_pruning_latest/bioid_deploy_copy_for_pruning.prototxt'
model_weights = '/home/wzleong/caffe/examples/nottingham_pruning_latest/_iter_'+iteration+'.caffemodel'


#defines the CNN network (previous pretrained model)
net = caffe.Net(model_def_prototxt, model_weights, caffe.TEST)		

layer_names_whole_network = dict.keys(net.params)

layer_names_only_fc = []
for s in layer_names_whole_network:
	if not 'conv' in s:
            layer_names_only_fc.append(s)

# sort to work on all FC1 then FC2
layer_names_only_fc.sort()
#print layer_names_only_fc

# use to generate only CONV layer caffemodel (excluding all FC layers)
#newnet.save('_iter_conv_only.caffemodel')


'''
############################################### 
#  Example of only 1 specific FC layer
############################################### 

fc_ori_weights_fc1 = net.params['fc1_right_mouth_coordinate'][0].data
fc_ori_bias_fc1 = net.params['fc1_right_mouth_coordinate'][1].data
fc_weights_fc1 = np.array(fc_ori_weights_fc1)
fc_bias_fc1 = np.array(fc_ori_bias_fc1)
fc_ori_weights_fc2 = net.params['fc2_right_mouth_coordinate'][0].data
fc_ori_bias_fc2 = net.params['fc2_right_mouth_coordinate'][1].data
fc_weights_fc2 = np.array(fc_ori_weights_fc2)
fc_bias_fc2 = np.array(fc_ori_bias_fc2)


Debugging: 
print 'fc_weights_fc1.shape',fc_weights_fc1.shape
# fc_weights_fc2.shape (2, 256)
print 'fc_weights_fc2.shape',fc_weights_fc2.shape
# fc_weights_fc1.shape (256, 2048)


fc_correlation = np.corrcoef(fc_weights_fc1)
mask =  np.tri(fc_correlation.shape[0], k=-1)
mask=np.logical_not(mask)
print mask

fc_correlation_diagonal = np.ma.array(fc_correlation, mask=mask) # mask out the upper triangle by setting to 0

fc_correlation_diagonal = np.ma.filled(fc_correlation_diagonal, 0)

height, width = fc_correlation_diagonal.shape



###################################################
#Debug / Testing the functionality of the code:
###################################################

print fc_correlation_diagonal

[[ 0.          0.          0.         ...,  0.          0.          0.        ]
 [-0.28271931  0.          0.         ...,  0.          0.          0.        ]
 [-0.3913509   0.53121182  0.         ...,  0.          0.          0.        ]
 ..., 
 [-0.08739955 -0.41785979 -0.23674112 ...,  0.          0.          0.        ]
 [ 0.31336691 -0.56456489 -0.52363988 ...,  0.40666417  0.          0.        ]
 [-0.1766842   0.57887009  0.46464565 ..., -0.60143934 -0.57731528  0.        ]]


print fc_correlation_diagonal.shape

# Pearson correlation matrix is a square matrix (height = width = size of hidden layer)
height, width = fc_correlation_diagonal.shape

# check from every column for the most correlated pair of weights

print fc_correlation_diagonal[0][0]
# 0

print fc_correlation_diagonal[1][0]
# -0.282719307482

print fc_correlation_diagonal[2][0]
# -0.391350902066

# element-wise absolute the values of the vector
print np.fabs(fc_correlation_diagonal[:,0])

[ 0.          0.28271931  0.3913509   0.27992662  0.42085241  0.45100223
  0.02584475  0.14401323  0.21651611  0.3466558   0.21778038  0.3481507
  0.16316998  0.04849061  0.23218536  0.17573857  0.3408282   0.08998728
  0.25041452  0.29484937  0.15851812  0.42206376  0.14608588  0.33952622
  0.38081869  0.13043047  0.34328436  0.25196837  0.16583371  0.13040166
  0.41085745  0.17130115  0.27624433  0.40935468  0.42601367  0.17694953
  0.31363273  0.3159126   0.17969091  0.12712658  0.28046727  0.28971986
  0.45119545  0.2603075   0.28036595  0.20358413  0.29609182  0.3571928
  0.03667988  0.36924331  0.22659894  0.30164737  0.14279861  0.39937949
  0.13932951  0.27635233  0.33186745  0.37289789  0.23514493  0.10993609
  0.42023367  0.20363302  0.25460399  0.27158178  0.31296284  0.25160763
  0.29143865  0.40679715  0.37309447  0.30824317  0.11127624  0.09511652
  0.32458584  0.23563948  0.24755515  0.30660512  0.39866685  0.14571627
  0.23507093  0.4055349   0.11780714  0.17972653  0.27092139  0.32289291
  0.02826687  0.4068423   0.2329065   0.21380028  0.42188744  0.40749342
  0.28621346  0.32051092  0.37090916  0.22578515  0.28890096  0.27684251
  0.09546397  0.26474093  0.31320029  0.40346921  0.15332311  0.40331443
  0.04087747  0.27188423  0.35996512  0.26092832  0.23001099  0.22942116
  0.19844652  0.19452804  0.24943708  0.32088208  0.07541697  0.33460913
  0.31294554  0.44657048  0.35423698  0.41988895  0.37600057  0.44317592
  0.18264349  0.32071078  0.30569357  0.28721294  0.08236405  0.07749614
  0.42174932  0.36461224  0.12061415  0.16151433  0.38070922  0.02600557
  0.3388205   0.23714809  0.11697584  0.13290089  0.35713086  0.31015651
  0.26591392  0.34151188  0.29983553  0.37911166  0.40672994  0.07647773
  0.42539332  0.09101005  0.27423029  0.29776863  0.27463808  0.38883569
  0.0165639   0.15718197  0.28910452  0.24272154  0.21171643  0.33093164
  0.38341432  0.35024635  0.35038656  0.32199861  0.38591495  0.42916059
  0.32090592  0.32592003  0.40399682  0.13298299  0.41750765  0.33799996
  0.31242304  0.30851748  0.15188729  0.15861722  0.20208203  0.16881607
  0.35602909  0.42527034  0.42772172  0.29206548  0.16501289  0.35197731
  0.404237    0.42260457  0.36919732  0.28284652  0.21501402  0.35293457
  0.44096253  0.22896962  0.24604773  0.31132211  0.00646019  0.11640376
  0.34333897  0.19339971  0.11291098  0.38441277  0.20216447  0.37544436
  0.0193846   0.32070717  0.31945963  0.31814362  0.33056922  0.17301123
  0.18684513  0.29586793  0.17456263  0.25287669  0.276634    0.38824802
  0.45641813  0.41815595  0.32764906  0.1320647   0.4081471   0.32470609
  0.29785354  0.27196705  0.41479265  0.29918928  0.03098934  0.40287626
  0.40502885  0.32983978  0.31778834  0.0940776   0.12155315  0.01722746
  0.38191032  0.0520385   0.28680857  0.13364743  0.43074676  0.26777088
  0.24615661  0.38732785  0.43022752  0.08319303  0.41879964  0.1774836
  0.36506028  0.02906498  0.25595625  0.34169132  0.07156441  0.24957885
  0.3573012   0.39929389  0.18726509  0.01423148  0.20991092  0.4068472
  0.38150645  0.08739955  0.31336691  0.1766842 ]


print fc_correlation_diagonal[:,0].shape
# (256,)

# get the index of the maximum value in the vector
index = np.argmax(np.fabs(fc_correlation_diagonal[:,0]))
# 0.456418125618 --> maximum value

# remove column (axis = 1) 
fc_correlation_diagonal = np.delete(fc_correlation_diagonal, 0, axis = 1)
print fc_correlation_diagonal
print fc_correlation_diagonal.shape

# remove row (axis = 0)
fc_correlation_diagonal = np.delete(fc_correlation_diagonal, index, axis = 0)
print fc_correlation_diagonal.shape


[[ 0.          0.          0.         ...,  0.          0.          0.        ]
 [ 0.          0.          0.         ...,  0.          0.          0.        ]
 [ 0.53121182  0.          0.         ...,  0.          0.          0.        ]
 ..., 
 [-0.41785979 -0.23674112 -0.58142867 ...,  0.          0.          0.        ]
 [-0.56456489 -0.52363988 -0.18058135 ...,  0.40666417  0.          0.        ]
 [ 0.57887009  0.46464565  0.36782007 ..., -0.60143934 -0.57731528  0.        ]]




threshold = 0.7
removed_index_list = []

for column in range(0,height):
	if (np.amax(np.fabs(fc_correlation_diagonal[:,0])) >= threshold):
		index = np.argmax(np.fabs(fc_correlation_diagonal[:,0]))
		removed_index_list.append(index)
		fc_correlation_diagonal = np.delete(fc_correlation_diagonal, index, axis = 0)

		# update weights on current FC1 (remove row)
		fc_weights_fc1 = np.delete(fc_weights_fc1, index, axis = 0)
		fc_bias_fc1 = np.delete(fc_bias_fc1, index, axis = 0)

		# update weights on next FC2 (remove column)
		fc_weights_fc2 = np.delete(fc_weights_fc2, index, axis = 1)

	fc_correlation_diagonal = np.delete(fc_correlation_diagonal, 0, axis = 1)

# Debugging 

print removed_index_list
# [241, 112, 225, 54, 83, 69, 194, 93, 220, 120, 184, 187, 237, 232]
print len(removed_index_list)
# 14
print fc_correlation_diagonal.shape
# (242, 0)
print fc_weights_fc1.shape
# (242, 2048)
print fc_bias_fc1.shape
# (242,)
print fc_weights_fc2.shape
# (2, 242)
print fc_bias_fc2.shape
# (2,)

'''

# read file and store in list (line by line)
with open('/home/wzleong/caffe/examples/nottingham_pruning_latest/bioid_deploy_copy_for_pruning.prototxt') as old_file:
	    content = old_file.readlines()
old_file.close()

# For debugging: 
#print content
#print content[1]

for current_fc in layer_names_only_fc:
	
	# get the fc layer weights from the original pretrained model
	fc_ori_weights_fc1 = net.params[current_fc][0].data
	fc_ori_bias_fc1 = net.params[current_fc][1].data
	fc_weights_fc1 = np.array(fc_ori_weights_fc1)
	fc_bias_fc1 = np.array(fc_ori_bias_fc1)

	#print fc_weights_fc1.shape

	# if there is a subsequent FC layer ( i.e., the current FC layer is fc1) then need to get the weights too
	
	if 'fc1' in current_fc:
		# replace the FC layer name from "fc1_...." to "fc2_...."
		next_fc = current_fc.replace("fc1","fc2",1) 

		fc_ori_weights_fc2 = net.params[next_fc][0].data
		fc_ori_bias_fc2 = net.params[next_fc][1].data
		fc_weights_fc2 = np.array(fc_ori_weights_fc2)
		fc_bias_fc2 = np.array(fc_ori_bias_fc2)


		# get Pearson correlation matrix 
		fc_correlation = np.corrcoef(fc_weights_fc1)
		mask =  np.tri(fc_correlation.shape[0], k=-1)
		mask=np.logical_not(mask)

		fc_correlation_diagonal = np.ma.array(fc_correlation, mask=mask) # mask out the upper triangle by setting to 0
		fc_correlation_diagonal = np.ma.filled(fc_correlation_diagonal, 0)

		height, width = fc_correlation_diagonal.shape

		threshold = 0.2

		for column in range(0,height):
			if (np.amax(np.fabs(fc_correlation_diagonal[:,0])) >= threshold):
				index = np.argmax(np.fabs(fc_correlation_diagonal[:,0]))
				fc_correlation_diagonal = np.delete(fc_correlation_diagonal, index, axis = 0)

				# update weights on current FC1 (remove row)
				fc_weights_fc1 = np.delete(fc_weights_fc1, index, axis = 0)
				fc_bias_fc1 = np.delete(fc_bias_fc1, index, axis = 0)

				#print fc_weights_fc1.shape[0]

				# update weights on next FC2 (remove column) if current fc is FC1
				if 'fc1' in current_fc:
					fc_weights_fc2 = np.delete(fc_weights_fc2, index, axis = 1)

				#print fc_weights_fc2.shape[1]

			fc_correlation_diagonal = np.delete(fc_correlation_diagonal, 0, axis = 1)

		index = -1
		for line_string in content: 
			index += 1

			# once detected the pattern (to find for the startpoint of the fc layer)
			if current_fc in content[index] and 'name' in content[index]:			
				# update the number of outputs in the current fc (FC1)
				# 1. continue skipping to the next line until the current line defines the number of output of the layer
				while 'num_output' not in content[index]: 
					index += 1
					continue
				# 2. update the number of output of the current layer 
				if 'num_output' in content[index]: 	# the if statement is actually not required but just for sanity check and for code -readability 
					#print content[index]
					#print str([int(s) for s in content[index].split() if s.isdigit()][0])
					content[index] = content[index].replace( str([int(s) for s in content[index].split() if s.isdigit()][0]), str(fc_weights_fc1.shape[0]))
					#print content[index]
				break

# remove all the substring "temp"
index = -1
for line_string in content: 
	index += 1
	content[index] = content[index].replace( "_temp", "")


new_file = open('/home/wzleong/caffe/examples/nottingham_pruning_latest/bioid_deploy_only_one_prune.prototxt', 'w')

for line_string in content: 
	new_file.write("%s" % line_string)

new_file.close()
	

# refer to template file without fully-connected layers (to prune fully-connected layers)

model_def_prototxt = '/home/wzleong/caffe/examples/nottingham_pruning_latest/bioid_deploy_only_one_prune.prototxt'
model_weights = '/home/wzleong/caffe/examples/nottingham_pruning_latest/_iter_'+iteration+'.caffemodel'

newnet = caffe.Net(model_def_prototxt,model_weights, caffe.TEST)	

for current_fc in layer_names_only_fc:
	
	# get the fc layer weights from the original pretrained model
	fc_ori_weights_fc1 = net.params[current_fc][0].data
	fc_ori_bias_fc1 = net.params[current_fc][1].data
	fc_weights_fc1 = np.array(fc_ori_weights_fc1)
	fc_bias_fc1 = np.array(fc_ori_bias_fc1)

	# if there is a subsequent FC layer ( i.e., the current FC layer is fc1) then need to get the weights too
	
	if 'fc1' in current_fc:
		# replace the FC layer name from "fc1_...." to "fc2_...."
		next_fc = current_fc.replace("fc1","fc2",1) 

		fc_ori_weights_fc2 = net.params[next_fc][0].data
		fc_ori_bias_fc2 = net.params[next_fc][1].data
		fc_weights_fc2 = np.array(fc_ori_weights_fc2)
		fc_bias_fc2 = np.array(fc_ori_bias_fc2)


		# get Pearson correlation matrix 
		fc_correlation = np.corrcoef(fc_weights_fc1)
		mask =  np.tri(fc_correlation.shape[0], k=-1)
		mask=np.logical_not(mask)

		fc_correlation_diagonal = np.ma.array(fc_correlation, mask=mask) # mask out the upper triangle by setting to 0
		fc_correlation_diagonal = np.ma.filled(fc_correlation_diagonal, 0)

		height, width = fc_correlation_diagonal.shape

		#threshold = 0.6

		for column in range(0,height):
			if (np.amax(np.fabs(fc_correlation_diagonal[:,0])) >= threshold):

				print current_fc
				index = np.argmax(np.fabs(fc_correlation_diagonal[:,0]))
				fc_correlation_diagonal = np.delete(fc_correlation_diagonal, index, axis = 0)

				# update weights on current FC1 (remove row)
				fc_weights_fc1 = np.delete(fc_weights_fc1, index, axis = 0)
				fc_bias_fc1 = np.delete(fc_bias_fc1, index, axis = 0)

				# update weights on next FC2 (remove column) if current fc is FC1
				if 'fc1' in current_fc:
					fc_weights_fc2 = np.delete(fc_weights_fc2, index, axis = 1)

			fc_correlation_diagonal = np.delete(fc_correlation_diagonal, 0, axis = 1)

		# store weights in the new network model
		newnet.params[current_fc.replace("_temp","")][0].data[...] = fc_weights_fc1
		newnet.params[current_fc.replace("_temp","")][1].data[...] = fc_bias_fc1

		if 'fc1' in current_fc:
			#print next_fc
			newnet.params[next_fc.replace("_temp","")][0].data[...] = fc_weights_fc2
			newnet.params[next_fc.replace("_temp","")][1].data[...] = fc_bias_fc2
		
	#####	
	#####	Attention!!!!!!
	#####	Do not need to care about FC2 because this script only focuses on FC1 and the weights and bias are immediately updated within the script for FC2 
	#####
	

newnet.save('_iter_netsurgery.caffemodel')


'''
for current_fc in layer_names_only_fc:
	temp_weights = np.array(newnet.params[current_fc][0].data)
	temp_bias = np.array(newnet.params[current_fc][1].data)
	print current_fc, temp_weights.shape, temp_bias.shape
'''










