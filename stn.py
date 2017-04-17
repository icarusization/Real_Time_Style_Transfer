import scipy.io
import numpy as np
import tensorflow as tf
import scipy.misc
from ops import *
batch_size=8
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
#size=[(1, 64, 64, 64), (1, 64, 64, 64), (1, 64, 64, 64), (1, 64, 64, 64), (1, 32, 32, 64), (1, 32, 32, 128), (1, 32, 32, 128), (1, 32, 32, 128), (1, 32, 32, 128), (1, 16, 16, 128), (1, 16, 16, 256), (1, 16, 16, 256), (1, 16, 16, 256), (1, 16, 16, 256), (1, 16, 16, 256), (1, 16, 16, 256), (1, 16, 16, 256), (1, 16, 16, 256), (1, 8, 8, 256), (1, 8, 8, 512), (1, 8, 8, 512), (1, 8, 8, 512), (1, 8, 8, 512)]
size=[(1, 64, 64, 3),(1, 64, 64, 64), (1, 64, 64, 64), (1, 64, 64, 64), (1, 64, 64, 64), (1, 32, 32, 64), (1, 32, 32, 128), (1, 32, 32, 128), (1, 32, 32, 128), (1, 32, 32, 128), (1, 16, 16, 128), (1, 16, 16, 256), (1, 16, 16, 256), (1, 16, 16, 256), (1, 16, 16, 256), (1, 16, 16, 256), (1, 16, 16, 256), (1, 16, 16, 256), (1, 16, 16, 256), (1, 8, 8, 256), (1, 8, 8, 512), (1, 8, 8, 512), (1, 8, 8, 512)]
#size=[(1, 64, 64, 3),(1, 64, 64, 64), (1, 64, 64, 64), (1, 64, 64, 64), (1, 64, 64, 64), (1, 32, 32, 64), (1, 32, 32, 128), (1, 32, 32, 128), (1, 32, 32, 128), (1, 32, 32, 128), (1, 16, 16, 128), (1, 16, 16, 256), (1, 16, 16, 256), (1, 16, 16, 256), (1, 16, 16, 256), (1, 16, 16, 256), (1, 16, 16, 256), (1, 16, 16, 256), (1, 16, 16, 256), (1, 8, 8, 256), (1, 8, 8, 512), (1, 8, 8, 512), (1, 8, 8, 512)]
size=size[::-1]
class stn(object):
	def __init__(self):
		self.layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

	def build_model(self,input_image):
	    data = scipy.io.loadmat(VGG_PATH)
	    mean = data['normalization'][0][0][0]
	    mean_pixel = np.mean(mean, axis=(0, 1))
	    weights = data['layers'][0]

	    net = {}
	    current = input_image
	    for i, name in enumerate(self.layers):
	        kind = name[:4]
	        if kind == 'conv':
	            kernels, bias = weights[i][0][0][0][0]
	            # matconvnet: weights are [width, height, in_channels, out_channels]
	            # tensorflow: weights are [height, width, in_channels, out_channels]
	            kernels = np.transpose(kernels, (1, 0, 2, 3))
	            bias = bias.reshape(-1)
	            #print name,":",np.shape(kernels)
	            current = _conv_layer(current, kernels, bias)
	        elif kind == 'relu':
	            current = tf.nn.relu(current)
	        elif kind == 'pool':
	            current = _pool_layer(current)
	        net[name] = current

	    assert len(net) == len(self.layers)
	    return net, mean_pixel

	def preprocess(self,image,mean_pixel):
		return image-mean_pixel

class dec(object):
	def __init__(self):
		self.layers = (
		'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2'
    )
		self.layers=self.layers[::-1]
		assert len(self.layers)==len(size)


	def build_model(self,in_input):
	    net = {}
	    current = in_input
	    string="dec_"
	    for i, name in enumerate(self.layers):
	        kind = name[:4]
	        new=(batch_size,)
	        new=new+size[i][1:4:1]
	        size[i]=new
	    	#print i,size[i]
	        #print i,current.get_shape()
	        #print kind
	        if kind == 'conv':
	        	#print size[i]
	        	#current=deconv2d(current,size[i],3, 3, 1,1)
	        	k_h=3
	        	k_w=3
	        	if current.get_shape()[2] != size[i][2]:
	        		d_h=2
	        		d_w=2
	        	else:
	        		d_w=1
	        		d_h=1
	        	output_shape=size[i]
	        	
	        	#print [k_h, k_w,output_shape[-1],int(current.get_shape()[-1])]
	        	#a=tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
	        	current=deconv2d(current,size[i],3,3,d_h,d_w,stddev=0.02,name=string+name)

	        	#a = tf.Variable(tf.random_uniform([k_h, k_w,output_shape[-1],int(current.get_shape()[-1])],-1.0, 1.0))
                 #           initializer=tf.random_normal_initializer(stddev=0.02))
	        	#current= tf.nn.conv2d_transpose(current,a, output_shape=size[i],strides=[1, d_h, d_w, 1])
	        elif kind == 'relu':
	            current = tf.nn.relu(current)
	        #elif kind == 'pool':
	        #    current = _pool_layer(current)
	        net[name] = current
        	#print len(net),len(self.layers)

        	if len(net) is 23:
        		current=tf.nn.tanh(current)
        		net["tanh"] = current
        		return net
		#self.t_vars = tf.trainable_variables()
		#self.saver = tf.train.Saver()
	#return net

def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),padding='SAME')
    return tf.nn.bias_add(conv, bias)


def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='SAME')

#content_image = imread('4.jpg')
model=dec()

#print model.build_model(content_image)
