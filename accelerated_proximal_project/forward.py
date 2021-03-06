#coding:utf-8
import tensorflow as tf

BATCH_SIZE = 1
IMAGE_SIZE_H = 321
IMAGE_SIZE_W = 481
NUM_CHANNELS = 1
CONV_SIZE = 3
CONV_KERNEL_NUM = 2
NUM_FILTER = 20

def get_weight(shape, regularizer):
	w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
	if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l1_regularizer(regularizer)(w))
	return w

def get_bias(shape):
	b = tf.Variable(tf.zeros(shape))
	return b

def conv2d(x, w):
	return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def get_variation(shape):
	#var = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
	#var = tf.truncated_normal(shape, stddev=0.1, seed=1)
	var = tf.random_normal(shape, stddev=0.1, seed=1)
	return var

def get_parameter(shape, regularizer):
	#param = tf.Variable(tf.constant(shape, dtype=tf.float32))
	param = tf.constant(shape, dtype=tf.float32)
	if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l1_regularizer(regularizer)(param))
	return param

def rot_90(tensor):
	return tf.transpose(tf.reverse_v2(tensor, [2]), [0, 2, 1, 3])

def rot_180(tensor):
	return tf.reverse_v2(tensor, [1, 0])

def prox_l2(v, y, rho):
	x = (v + rho*y) / (1.0 + rho)
	return x

def forward(x, regularizer):
	#conv_w = get_weight([NUM_FILTER, CONV_SIZE, CONV_SIZE, NUM_CHANNELS, CONV_KERNEL_NUM], regularizer)
	#tf.summary.histogram('conv_w', conv_w)
	Dn = tf.reshape(tf.constant([[0 ,0, 0], [0, -1, 1], [0, 0, 0]], dtype=tf.float32), (3, 3, 1))
	Dv = tf.reshape(tf.constant([[0 ,0, 0], [0, -1, 0], [0, 1, 0]], dtype=tf.float32), (3 ,3, 1))
	D = tf.stack([Dn, Dv], axis=3)
	z = get_variation([BATCH_SIZE, IMAGE_SIZE_H, IMAGE_SIZE_W, NUM_CHANNELS])
	zprev = get_variation([BATCH_SIZE, IMAGE_SIZE_H, IMAGE_SIZE_W, NUM_CHANNELS])
	lambda_ = get_parameter(0.06, regularizer)
	tf.summary.scalar('lambda', lambda_)
	trade_off = get_parameter(0.1, regularizer)
	tf.summary.scalar('trade_off', trade_off)

	for k in range(60):
		v = z + (k/(k + 3))*(z - zprev)
		conv = conv2d(v, D)
		conv = conv / (tf.abs(conv) + 1e-12)
		gradient = lambda_*conv2d(conv, tf.reshape(rot_180(D), [3, 3, 2, 1]))
		tf.summary.histogram('gradient', gradient)
		z = prox_l2(v - trade_off*gradient, x, trade_off)
		zprev = v

	y = z
	return y
