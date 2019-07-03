#coding:utf-8
import tensorflow as tf

BATCH_SIZE = 1
IMAGE_SIZE_H = 321
IMAGE_SIZE_W = 481
NUM_CHANNELS = 1
CONV_SIZE = 3
CONV_KERNEL_NUM = 2

def get_weight(shape, regularizer):
	w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
	if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w

def get_bias(shape):
	b = tf.Variable(tf.zeros(shape))
	return b

def conv2d(x, w):
	return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def get_variation(shape):
	var = tf.truncated_normal(shape, stddev=0.1, seed=1)
	return var

def get_parameter(shape, regularizer):
	param = tf.Variable(tf.constant(shape, dtype=tf.float32))
	if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(param))
	return param

def prox_l2(v, y, rho):
	x = (v + rho*y) / (1.0 + rho)
	return x

def shrinkage(a, kappa):
	z = tf.maximum(0.0, a - kappa) - tf.maximum(0.0, -a - kappa)
	return z

def rot_90(tensor):
	return tf.transpose(tf.reverse_v2(tensor, [2]), [0, 2, 1, 3])

def rot_180(tensor):
	return tf.reverse_v2(tensor, [1, 0])

def forward(x, regularizer):
	Dn = tf.reshape(tf.constant([[0 ,0, 0], [0, -1, 1], [0, 0, 0]], dtype=tf.float32), (3, 3, 1))
	Dv = tf.reshape(tf.constant([[0 ,0, 0], [0, -1, 0], [0, 1, 0]], dtype=tf.float32), (3 ,3, 1))
	D = tf.stack([Dn, Dv], axis=3)
	lambda_  = tf.constant(0.07, dtype=tf.float32)
	proximal_tradeoff = tf.constant(1.0, dtype=tf.float32)
	theta = tf.constant(0.9, dtype=tf.float32)

	temp_x = get_variation([BATCH_SIZE, IMAGE_SIZE_H, IMAGE_SIZE_W, NUM_CHANNELS])
	x_hat = get_variation([BATCH_SIZE, IMAGE_SIZE_H, IMAGE_SIZE_W, NUM_CHANNELS])
	u = get_variation([BATCH_SIZE, IMAGE_SIZE_H, IMAGE_SIZE_W, CONV_KERNEL_NUM])

	for k in range(60):
		u = u + lambda_*conv2d(x_hat, D)
		u = u - shrinkage(u, lambda_)
		gradient = conv2d(u, tf.reshape(rot_180(D), [3, 3, 2, 1]))
		tf.summary.histogram('gradient', gradient)
		xprev = temp_x
		temp_x = prox_l2(temp_x - proximal_tradeoff*gradient, x, proximal_tradeoff)
		x_hat = temp_x + theta*(temp_x - xprev)

	y = temp_x
	return y
