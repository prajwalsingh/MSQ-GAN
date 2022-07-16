import os
from glob import glob
from natsort import natsorted
import tensorflow as tf
import skimage.io as skio
from skimage.transform import resize
from tifffile import imwrite
import numpy as np
import cv2
from save_figure import save_figure
import h5py
from functools import partial
import tensorflow_io as tfio

import tensorflow as tf
from functools import partial

# class generator:
# 	def __call__(self, file):
# 		for filename in file:
# 			with h5py.File(file, 'r') as hf:
# 				image = hf['image'][()]
# 				code  = hf['code'][()]
# 				# for image, code in hf["image"], hf["code"]:
# 			yield image, code

# def load_complete_data(data_path, label_path, input_res=512, batch_size=16, shuffle_buffer_size=100):
# 	dataset = tf.data.Dataset.list_files(data_path)
# 	# dataset = dataset.map(read_file)
# 	dataset = dataset.from_generator(generator, (tf.uint8, tf.float32), ((None, None, None), (None,)))
# 	# dataset = dataset.map(partial(preprocess_data, resolution=input_res))
# 	dataset = dataset.shuffle(buffer_size=shuffle_buffer_size).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
# 	return dataset

# def read_file(path):
# 	# spec = {
#  #        '/image': tf.TensorSpec(shape=[None, None, None], dtype=tf.uint8),
#  #        '/code': tf.TensorSpec(shape=[None,], dtype=tf.float32),
#  #    }
# 	image = tfio.IODataset.from_hdf5(path, dataset='/image', spec=tf.TensorSpec(shape=[128, 128, 3], dtype=tf.uint8))
# 	image = tf.as_dtype(image)
# 	# code  = tfio.IODataset.from_hdf5(path, dataset='/code', spec=tf.float32)
# 	# h5_tensors = tfio.IOTensor.from_hdf5(path, spec=spec)
# 	# data  = {'image':image, 'code':code}
# 	# image = h5_tensors('/image').to_tensor()
# 	# code  = h5_tensors('/code').to_tensor()
# 	return image

# def preprocess_data(data, code, resolution):
# 	# image = h5_tensors('/image').to_tensor()
# 	# code  = h5_tensors('/code').to_tensor()
# 	# data = tf.image.decode_jpeg(data, channels=3)
# 	data = tf.image.resize(data, (resolution, resolution))
# 	return ( (tf.cast(data, dtype=tf.float32) - 127.5) / 127.5 ), code


def load_batch(data_path, res, batch_size=8):
	paths = natsorted(glob(data_path))
	batch = tf.data.Dataset.from_tensor_slices(paths).map(partial(read_data, res=res)).shuffle(2*batch_size).batch(batch_size, drop_remainder=False)
	return batch


# def load_data(paths, dt, res):
# 	X, C = list(zip(*map(partial(read_data, dt=dt, res=res), paths)))
# 	X = tf.convert_to_tensor(X, dtype=tf.float32)
# 	C = tf.convert_to_tensor(C, dtype=tf.float32)
# 	return X, C


def read_data(path, res):
	# path  = path.numpy().decode('utf-8')
	# code  = tf.cast(np.zeros(shape=(max(dt.values())+1,), dtype=np.float32), dtype=tf.float32)
	# code[dt[path.split(sep='/')[-2]]] = 1
	# print(tf.cast(path, dtype=tf.string).split(sep='/'))
	# code[dt[tf.string.split(tf.convert_to_tensor(path, dtype=tf.string), sep='/')[-2]]] = 1
	image = tf.image.resize(tf.image.decode_jpeg(tf.io.read_file(path), channels=3), (res, res))
	# image = cv2.resize(cv2.imread(path, 1), (res, res))
	# image = (np.float32(image) - 127.5) / 127.5
	image = ( (tf.cast(image, dtype=tf.float32) - 127.5) / 127.5 )
	return image, path

# @tf.function
def vis(model, idx, X, latent, exp_dir, wandb=None):
	if not os.path.isdir(exp_dir+'/results'):
		os.makedirs(exp_dir+'/results/generated/')
		os.makedirs(exp_dir+'/results/autoenc/')

	# H_hat     = model.gcn(graph, H, training=False)
	# latent    = tf.concat([H_hat, latent], axis=-1)
	# latent    = tf.concat([H, latent], axis=-1)
	
	X_fake     = model.gen(latent, training=False) # GEN
	# _, X_recon = model.disc(X, training=False)
	

	X_fake  = save_figure(X_fake, exp_dir+'/results/generated/{}.jpg'.format(idx))
	# X       = save_figure(X, exp_dir+'/results/autoenc/{}_real.jpg'.format(idx))
	# X_recon = save_figure(X_recon, exp_dir+'/results/autoenc/{}_recon.jpg'.format(idx))
	
	# wandb.log({'generated': [wandb.Image(X_fake, caption='Itr: {}'.format(idx))]})
	# wandb.log({'real': [wandb.Image(X, caption='Itr: {}'.format(idx))]})
	# wandb.log({'dis_recon': [wandb.Image(X_recon, caption='Itr: {}'.format(idx))]})
