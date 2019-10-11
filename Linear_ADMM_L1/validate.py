#coding:utf-8

import forward
import numpy as np
import pandas as pd
import os, time, glob
import tensorflow as tf
import PIL.Image as Image
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr, compare_ssim

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sigma = 30.0
BATCH_SIZE = 1
out_dir = "./data/output/"
test_dir = "./data/Test/Set68/"
out_dir_noise = "./data/output/img_noise/"
out_dir_denoise = "./data/output/img_denoise/"

def validate():
	""" validate ADMM """
	with tf.Session() as sess:
		global_init_op = tf.global_variables_initializer()
		local_init_op = tf.local_variables_initializer()
		sess.run([global_init_op, local_init_op])
		
		name = []
		psnr = []
		ssim = []
		file_list = glob.glob('{}/*.png'.format(test_dir))
		for file in file_list:
			img_clean = np.array(Image.open(file), dtype='float32') / 255.0	
			img_test = img_clean + np.random.normal(0, sigma / 255.0, img_clean.shape)
			img_test = img_test.astype('float32')

			reshaped_img = np.reshape(img_test, (
				BATCH_SIZE,
				forward.IMAGE_SIZE_H,
				forward.IMAGE_SIZE_W,
				forward.NUM_CHANNELS))

			img_out = forward.forward(reshaped_img, None)
			img_out = np.clip(sess.run(img_out).reshape(img_clean.shape), 0, 1)

			# calculate numeric metrics
			psnr_noise, psnr_denoised = compare_psnr(img_clean, img_test), compare_psnr(img_clean, img_out)
			ssim_noise, ssim_denoised = compare_ssim(img_clean, img_test), compare_ssim(img_clean, img_out)
			psnr.append(psnr_denoised)
			ssim.append(ssim_denoised)

			#save images
			filename = file.split('/')[-1].split('.')[0]
			name.append(filename)
			img_test = Image.fromarray((img_test*255).astype('uint8'))
			img_test.save(out_dir_noise + filename + '_sigma' + '{}_psnr{:.2f}.png'.format(sigma, psnr_noise))
			img_out = Image.fromarray((img_out*255).astype('uint8'))
			img_out.save(out_dir_denoise + filename + '_psnr{:.2f}.png'.format(psnr_denoised))

			print("Validate %d step(s), validate psnr is %gdb." % (len(psnr), psnr_denoised))
			print("Validate %d step(s), max psnr is %gdb." % (len(psnr), max(psnr)))
			print("Validate %d step(s), min psnr is %gdb." % (len(psnr), min(psnr)))

		psnr_avg = sum(psnr) / len(psnr)
		ssim_avg = sum(ssim) / len(ssim)
		name.append('Average')
		psnr.append(psnr_avg)
		ssim.append(ssim_avg)
		print('Aversge PSNR = {0:.2f}, SSIM = {1:.2f}'.format(psnr_avg, ssim_avg))
		pd.DataFrame({'name':np.array(name), 'psnr':np.array(psnr), 'ssim':np.array(ssim)}).to_csv(out_dir + '/metrics.csv', index=True)

def main():
	validate()

if __name__ == '__main__':
	main()
