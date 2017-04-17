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

CONTENT_LAYER = 'relu4_2'
stn=stn()
dec=dec()
content_features = {}
RES_LAYER='tanh'
printlayers=['tanh','conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1']
batch_size=8
lamb=0.25
epo=1000
layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2'
    )

loss_calc_layers={'relu1_1','relu2_1','relu3_1','relu4_1'}

'''
def calcul_fea(image_in):
	shape = (1,) + image_in.shape
	g = tf.Graph()
	with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
		image = tf.placeholder('float', shape=shape)
		net, mean_pixel = stn.build_model(image)
		#print net,mean_pixel
		content_pre = np.array([stn.preprocess(image_in, mean_pixel)])
		#for CONTENT_LAYER in layers:
		content_features[CONTENT_LAYER] = net[CONTENT_LAYER].eval(
	                feed_dict={image: content_pre})
		return content_features[CONTENT_LAYER]
		#print content_features[CONTENT_LAYER]
			#print CONTENT_LAYER, ":",np.shape(content_features[CONTENT_LAYER])

def run_dec(feature_in):
	g = tf.Graph()
	with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
		data = tf.placeholder('float', shape=np.shape(feature_in))
		#print np.shape(feature_in)
		net= dec.build_model(data)
		#for CONTENT_LAYER in layers:
		try:
			init_op = tf.global_variables_initializer()
		except:
			init_op = tf.initialize_all_variables()

		sess.run(init_op)
		content_features[RES_LAYER] = net[RES_LAYER].eval(
	                feed_dict={data:feature_in})
		return content_features[RES_LAYER]
'''

def calcul_fea(image_in,net,mean_pixel,image,sess):
	#g = tf.Graph()
	#content_pre = np.array([stn.preprocess(image_in, mean_pixel)])
		#content_features[CONTENT_LAYER] = net[CONTENT_LAYER].eval(
	    #            feed_dict={image: content_pre})
	return sess.run(net[CONTENT_LAYER],feed_dict={image: image_in})

def run_dec(feature_in,net,data,sess):
	#g = tf.Graph()
	#with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
		#content_features[RES_LAYER] = net[RES_LAYER].eval(
	    #            feed_dict={data:feature_in})
		#return content_features[RES_LAYER]
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
	print "shape is :",shape
	image = tf.placeholder('float', shape=shape)
	image2 = tf.placeholder('float', shape=shape)
	net, mean_pixel = stn.build_model(image)
	stn.build_model(image2)
	return net, mean_pixel,image

def init_dec(sess):
	data = tf.placeholder('float', shape=(batch_size, 8, 8, 512))
		#print np.shape(feature_in)
	net= dec.build_model(data)
	print len(net)
		#for CONTENT_LAYER in layers:

	return net,data	


def save(checkpoint_dir, step,sess,saver):
	#os.remove('./checkpoint/*')
	model_name ="saver"
	model_dir = "%s_%s_%s" % ("data", batch_size, 1)
	checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

	if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)

	saver.save(sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)
'''def load(checkpoint_dir):
    print(" [*] Reading checkpoints...")

    model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        print(" [*] Success to read {}".format(ckpt_name))
        return True
    else:
        print(" [*] Failed to find a checkpoint")
        return False	'''

def run_model(content_image_in,style_image_in):
	data_style = glob(os.path.join("./data", "style", "*.jpg"))
	data2 = glob(os.path.join("./data", "content", "*.jpg"))
	batch_idxs = min(len(data_style), len(data2)) /batch_size
	print batch_idxs
	sess = tf.Session()
	shape = (batch_size,) + content_image_in.shape
	#image = tf.placeholder('float', shape=shape)
	#net_enc, mean_pixel = stn.build_model(image)
	#sess.run(content_features[CONTENT_LAYER],ffeed_dict={image: content_pre})
	net_enc,mean_pixel,image=init_enc(content_image_in)
	net_dec,data=init_dec(sess)
	net_after_dec,_= stn.build_model(net_dec["tanh"])
	shape = (batch_size,) + content_image_in.shape
	style_image= tf.placeholder('float', shape=shape)
	net_get_ls,_=stn.build_model(style_image)
	loss_c=tf.nn.l2_loss(data-net_after_dec["relu4_2"])
	loss_s=0
	saver = tf.train.Saver()
	for layer in loss_calc_layers:
		m,s=tf.nn.moments(net_after_dec[layer],[1,2,3])
		s=tf.sqrt(s)
		m_2,s_2=tf.nn.moments(net_get_ls[layer],[1,2,3])
		s_2=tf.sqrt(s_2)
			#print m,s
		loss_s+=tf.nn.l2_loss(m-m_2)+tf.nn.l2_loss(s-s_2)

	loss_c/=batch_size
	loss_s/=batch_size
	loss=loss_c+loss_s*lamb

	loss_c_sum=scalar_summary("c_loss",loss_c)
	loss_s_sum=scalar_summary("s_loss",loss_s)
	loss_all_sum=merge_summary([loss_c_sum,loss_s_sum])
	writer = SummaryWriter("./logs", sess.graph)

	counter=1
	t_vars = tf.trainable_variables()
	d_vars = [var for var in t_vars if 'dec_' in var.name]
	#d_name = [var.name for var in d_vars]
	#print d_name
	#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss,var_list=d_vars)
	train_step=tf.train.AdamOptimizer(1e-4, beta1=0.5).minimize(loss, var_list=d_vars)
	try:
		init_op = tf.global_variables_initializer()
	except:
		init_op = tf.initialize_all_variables()

	sess.run(init_op)
	old_content=[]
	start_time=time.time()
	for epoch in xrange(0,epo):
		for index in xrange(0,batch_idxs-1):

			batch_files = data_style[index * batch_size:(index + 1)*batch_size]
			batch_files_c = data2[index * batch_size:(index + 1)*batch_size]
			#batch_files = data[0:8]
			batch = [inverse_transform(get_image(batch_file,64,is_crop=False)) for batch_file in batch_files]
			batch_c = [inverse_transform(get_image(batch_file_c,64,is_crop=False)) for batch_file_c in batch_files_c]
			#print "graph is ",batch[1]
			#batch=np.ndarray(batch)
			#batch=batch/2+1
			#print "counter is",counter #,"batch is ",type(batch)
			#print np.shape(batch)

			content_data=calcul_fea(batch,net_enc,mean_pixel,image,sess)
			content_loss_c=content_data
			style_data=calcul_fea(batch_c,net_enc,mean_pixel,image,sess)

			mean_c=np.mean(content_data)
			std_c=np.std(content_data)
			mean_s=np.mean(style_data)
			std_s=np.std(style_data)
			#print type(content_data),np.shape(content_data)
			content_data=content_data-mean_c
			content_data=content_data/std_c
			new_content=content_data*std_s
			new_content=new_content+mean_s
			#print old_content==new_content
			old_content=new_content
			#print np.shape(new_content)
			#print new_content
			#print "shape of new_content is ",np.shape(new_content)
			#print 'shape2 is ',np.shape(new_content)
			#print "shape is ",np.shape(net_dec["conv1_1"])
			#print net_dec["conv1_1"]
			
			#print np.shape(net_after_dec)


			#sty_pict=run_dec(new_content,net_dec,data,sess)
			#print np.shape(sty_pict[0]),np.shape(content_image_in)
			#sty_data=calcul_fea(sty_pict[0],net_enc,mean_pixel,image,sess)
				#mean+=tf.nn.l2_loss()
				#std+=tf.nn.l2_loss(np.std(res)-np.std(res2))
			#loss_s=calcul_fea_loss(style_image_in,sty_pict[0],net_enc,mean_pixel,image,sess)
			#content_pre = np.array([stn.preprocess(content_image_in, mean_pixel)])
			_,summary_str=sess.run([train_step,loss_all_sum],feed_dict={data: new_content,style_image:batch})
			writer.add_summary(summary_str, counter)
			#compute the loss
			lc=sess.run(loss_c,feed_dict={data: new_content,style_image:batch})
			ls=sess.run(loss_s,feed_dict={data: new_content,style_image:batch})

			counter+=1
			print("Epoch: [%2d] [%4d/%4d] time: %4.4f, c_loss: %.8f, s_loss: %.8f" \
                    % (epoch, index, batch_idxs,
                    time.time() - start_time, lc, ls))
			#for a in printlayers:
			#	print "layer is ",a,"**********",sess.run(net_dec[a],feed_dict={data:new_content})
                
			#img=run_dec(new_content,net_dec,data,sess)
			#print np.shape(img[0])
			#scipy.misc.imsave('./imagesave/train_{:02d}_{:04d}.png'.format(epoch, index),img[0])
			if np.mod(counter,100)==1:
				#save("checkpoint", counter,sess,saver)
				img=run_dec(new_content,net_dec,data,sess)
				#print np.shape(img)
				#print "image is::::::::::::::",img[1]
				#print np.shape(img[1])
				scipy.misc.imsave('./imagesave/train_{:02d}_{:04d}.png'.format(epoch, index),inverse_transform(img[1]))
				#save_images(img, [64, 64],'./imagesave/train_{:02d}_{:04d}.png'.format(epoch, index))
			if np.mod(counter,1000)==2:
				save("checkpoint", counter,sess,saver)
		#print loss_c
	




	







if not os.path.exists("imagesave"):
	os.makedirs("imagesave")
if not os.path.exists("logs"):
	os.makedirs("logs")
if not os.path.exists("checkpoint"):
	os.makedirs("checkpoint")

content_image = imread('4.jpg')
#print np.shape(content_image)
#model=stn()
run_model(content_image,content_image)
#feature=calcul_fea(content_image)
#print np.shape(run_dec(feature))

