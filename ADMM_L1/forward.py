#coding:utf-8
#cython:language_level=2
import tensorflow as tf

BATCH_SIZE = 1
IMAGE_SIZE_H = 321
IMAGE_SIZE_W = 481
NUM_CHANNELS = 1
CONV_SIZE = 3
CONV_KERNEL_NUM = 2

def get_weight(shape, name, regularizer):
	w = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)
	if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l1_regularizer(regularizer)(w))
	return w

def get_bias(shape, name):
	b = tf.Variable(tf.zeros(shape), name=name)
	return b

def get_variation(shape):
	var = tf.zeros(shape)
	return var

def get_parameter(shape, name, regularizer):
	param = tf.Variable(tf.constant(shape, dtype=tf.float32), name=name)
	if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l1_regularizer(regularizer)(param))
	return param

def conv2d(x, w):
	return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def rot_90(tensor):
	return tf.transpose(tf.reverse_v2(tensor, [2]), [0, 2, 1, 3])

def rot_180(tensor):
	return tf.reverse_v2(tensor, [1, 0])

def Gauss_Seidel(V, Y, Z, U, RHO):
	with tf.Session() as sess:
		v, y, z, u, rho = sess.run([V, Y, Z, U, RHO])
		coeff_1 = rho / (1.0 + 4.0*rho)
		coeff_2 = 1.0 / (1.0 + 4.0*rho)

		for i in range(1, v.shape[1]-1): 
			for j in range(1, v.shape[2]-1):

				v[:,i,j,:] = coeff_1*(v[:,i+1,j,:] + v[:,i-1,j,:] + v[:,i,j+1,:] + v[:,i,j-1,:] + 
				z[:,i-1,j,0:1] - z[:,i,j,0:1] + z[:,i,j-1,1:2] - z[:,i,j,1:2] + 
				u[:,i,j,0:1] - u[:,i-1,j,0:1] + u[:,i,j,1:2] - u[:,i,j-1,1:2]) + coeff_2*y[:,i,j,:]

	return tf.convert_to_tensor(v)

def XN(v, y, z, u, rho):
	rec_v = Gauss_Seidel(v, y, z, u, rho)
	return rec_v

def CN(x, w):
	return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def ZN(a, kappa):
	z = tf.maximum(0.0, a - kappa) - tf.maximum(0.0, -a - kappa)
	return z

def MN(Dx, z, u):
	u = u + Dx - z
	return u

def forward(x, regularizer):
	Dn = tf.reshape(tf.constant([[0 ,0, 0], [0, -1, 1], [0, 0, 0]], dtype=tf.float32), (3, 3, 1))
	Dv = tf.reshape(tf.constant([[0 ,0, 0], [0, -1, 0], [0, 1, 0]], dtype=tf.float32), (3 ,3, 1))
	D = tf.stack([Dn, Dv], axis=3)
	lambda_ = tf.constant(0.05, dtype=tf.float32)
	rho = tf.constant(1.0, dtype=tf.float32)
	
	v = x
	z = get_variation([BATCH_SIZE, IMAGE_SIZE_H, IMAGE_SIZE_W, CONV_KERNEL_NUM])
	u = get_variation([BATCH_SIZE, IMAGE_SIZE_H, IMAGE_SIZE_W, CONV_KERNEL_NUM])

	for k in range(40):
		v = XN(v, x, z, u, rho)
		conv = CN(v, D)
		z = ZN(conv + u, lambda_/rho)
		u = MN(conv, z, u)

	y = v
	return y
