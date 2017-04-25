from stn import stn
from stn import dec
import numpy as np
import scipy
import tensorflow as tf
from glob import glob
import os
from utils import *
import math
from ops import *
import time
try:
	reduce
except NameError:
	from functools import reduce

CONTENT_LAYER = 'relu4_2'
stn=stn()
dec=dec()
content_features = {}
RES_LAYER='relu1_1'
printlayers=['tanh','conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1']
last_layer="relu4_2"
batch_size=1
lamb=200
con_weight=1
tv_weight=1
epo=1000
layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2'
    )

loss_calc_layers={'relu1_1','relu2_1','relu3_1','relu4_1'}
#loss_calc_layers={'relu4_1'}
def calcul_fea(image_in,net,mean_pixel,image,sess):
	return sess.run(net[CONTENT_LAYER],feed_dict={image: image_in})

def run_dec(feature_in,net,data,sess):
	return sess.run(net[RES_LAYER],feed_dict={data:feature_in})

def calcul_fea_loss(image_in,image2_in,net,mean_pixel,image,sess):
	#g = tf.Graph()
	content_pre = np.array([stn.preprocess(image_in, mean_pixel)])
	content2_pre = np.array([stn.preprocess(image2_in, mean_pixel)])
		#content_features[CONTENT_LAYER] = net[CONTENT_LAYER].eval(
	    #            feed_dict={image: content_pre})
	mean=0
	std=0
	for layer in loss_calc_layers:
		res=sess.run(net[layer],feed_dict={image: content_pre})
		res2=sess.run(net[layer],feed_dict={image: content2_pre})
		mean+=tf.nn.l2_loss(np.mean(res)-np.mean(res2))
		std+=tf.nn.l2_loss(np.std(res)-np.std(res2))
		#print mean,std
	return mean+std

def imread(path):
    return scipy.misc.imread(path).astype(np.float)

def init_enc(image_in):
	shape = (batch_size,) + image_in.shape
    #print "shape is :",shape
	image = tf.placeholder('float', shape=shape)
	#image2 = tf.placeholder('float', shape=shape)
	net, mean_pixel = stn.build_model(image)
	return net, mean_pixel,image


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)


def run_model(content_image_in,style_image):
	sess = tf.Session()
	shape = (batch_size,) + content_image_in.shape

	image_cont = tf.placeholder('float', shape=shape)
	net_enc, mean_pixel = stn.build_model(image_cont)

	image_style = tf.placeholder('float', shape=shape)
	net_forsty,_= stn.build_model(image_style)

	miu,sigma_square=tf.nn.moments(net_enc[last_layer],[1,2])
	sigma_square=sigma_square+tf.abs(tf.truncated_normal(tf.shape(sigma_square)))
	sigma=tf.sqrt(sigma_square)

	miu_sty,sigma_square_sty=tf.nn.moments(net_forsty[last_layer],[1,2])
	sigma_square_sty=sigma_square_sty+tf.abs(tf.truncated_normal(tf.shape(sigma_square_sty)))
	sigma_sty=tf.sqrt(sigma_square_sty)
	adain=sigma_sty*((net_enc[last_layer]-miu)/sigma)+miu_sty


	net_dec=dec.build_model(adain)
	net_after_dec,_= stn.build_model(net_dec[RES_LAYER])
	shape = (batch_size,) + content_image_in.shape

	try:
		init_op = tf.global_variables_initializer()
	except:
		init_op = tf.initialize_all_variables()

	sess.run(init_op)
	content_in=np.array([content_image_in])
	style_in=np.array([style_image])
	img=sess.run(net_dec[RES_LAYER],feed_dict={image_cont:content_in,image_style:style_in})
	scipy.misc.imsave('./imagesave/res.png',inverse_transform(img[0]))

	

if not os.path.exists("imagesave"):
	os.makedirs("imagesave")

content_image = imread('2.jpg')
style_image=imread('1.jpg')
run_model(content_image,style_image)

