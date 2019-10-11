#coding:utf-8
import tensorflow as tf

BATCH_SIZE = 1
IMAGE_SIZE_H = 321
IMAGE_SIZE_W = 481
NUM_CHANNELS = 1
CONV_SIZE = 3
CONV_KERNEL_NUM = 2

def get_weight(shape, name, regularizer):
	""" get weight with regularizer """
	w = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)
	if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l1_regularizer(regularizer)(w))
	return w

def get_bias(shape, name):
	""" get bias """
	b = tf.Variable(tf.zeros(shape), name=name)
	return b

def get_variation(shape):
	""" get variation """
	var = tf.zeros(shape)
	return var

def get_parameter(shape, name, regularizer):
	""" get parameter """
	param = tf.Variable(tf.constant(shape, dtype=tf.float32), name=name)
	if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l1_regularizer(regularizer)(param))
	return param

def conv2d(x, w):
	""" conv2d """
	return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def rot_90(tensor):
	"""  rotation 90 of 3x3 tensor with NHWC """ 
	return tf.transpose(tf.reverse_v2(tensor, [2]), [0, 2, 1, 3])

def rot_180(tensor):
	"""  rotation 180 of 3x3 tensor with NHWC """ 
	return tf.reverse_v2(tensor, [1, 0])

def prox_l2(v, y, mu):
	""" proximal operator of 1/2||x||^2 """
	x = (v + mu*y) / (1.0 + mu)
	return x

def CN(x, w):
	""" CN Layer """
	return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def ZN(a, kappa):
	""" ZN Layer """
	z = tf.maximum(0.0, a - kappa) - tf.maximum(0.0, -a - kappa)
	return z

def MN(Dx, z, u):
	""" MN Layer """
	u = u + Dx - z
	return u

def forward(x, regularizer):
	""" compute graph """
	x = tf.convert_to_tensor(x)
	Dn = tf.reshape(tf.constant([[0 ,0, 0], [0, -1, 1], [0, 0, 0]], dtype=tf.float32), (3, 3, 1))
	Dv = tf.reshape(tf.constant([[0 ,0, 0], [0, -1, 0], [0, 1, 0]], dtype=tf.float32), (3 ,3, 1))
	D = tf.stack([Dn, Dv], axis=3)
	lambd = tf.constant(0.06, dtype=tf.float32)
	mu = tf.constant(0.2, dtype=tf.float32)
	rho = tf.constant(1.0, dtype=tf.float32)
	
	z = get_variation([BATCH_SIZE, IMAGE_SIZE_H, IMAGE_SIZE_W, CONV_KERNEL_NUM])
	u = get_variation([BATCH_SIZE, IMAGE_SIZE_H, IMAGE_SIZE_W, CONV_KERNEL_NUM])
	v = x

	for k in range(60):
		v = v - mu*rho*CN(CN(v, D) - z + u, tf.reshape(rot_180(D), [3, 3, 2, 1]))
		v = prox_l2(v, x, mu)
		conv = CN(v, D)
		z = ZN(conv + u, lambd/rho)
		u = MN(conv, z, u)

	y = v
	return y
