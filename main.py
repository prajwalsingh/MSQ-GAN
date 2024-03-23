import tensorflow as tf
from utils import vis, load_batch#, load_data
# from utils import vis, load_complete_data
from model import DCGAN, dist_train_step#, train_step
from tqdm import tqdm
import os
import shutil
import pickle
from glob import glob
from natsort import natsorted
import wandb
import numpy as np
# from tensorflow import keras

tf.random.set_seed(45)
np.random.seed(45)

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]= '1'
clstoidx   = {}
idxtocls   = {}


# @tf.function
def get_code(path):
	path  = path.numpy().decode('utf-8')
	code  = np.zeros(shape=(max(clstoidx.values())+1,), dtype=np.float32)
	code[clstoidx[path.split(sep='/')[-2]]] = 1
	return tf.cast(code, dtype=tf.float32)


if __name__ == '__main__':

	# if len(glob('experiments/*'))==0:
	# 	os.makedirs('experiments/experiment_1/code/')
	# 	exp_num = 1
	# else:
	# 	exp_num = len(glob('experiments/*'))+1
	# 	os.makedirs('experiments/experiment_{}/code/'.format(exp_num))

	# exp_dir = 'experiments/experiment_{}'.format(exp_num)
	# for item in glob('*.py'):
	# 	shutil.copy(item, exp_dir+'/code')
	
	gpus = tf.config.list_physical_devices('GPU')
	mirrored_strategy = tf.distribute.MirroredStrategy(devices=['/GPU:0','/GPU:1'], 
		cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
	n_gpus = mirrored_strategy.num_replicas_in_sync
	# print(n_gpus)

	exp_dir = 'experiments/'
	batch_size = 8
	latent_dim = 128
	
	# with open('../dataset/imagenet_clstoidx.pkl', 'rb') as file:
	# 	imagenet_clstoidx = pickle.load(file)
	# with open('../dataset/imagenet_idxtocls.pkl', 'rb') as file:
	# 	imagenet_idxtocls = pickle.load(file)
	# n_class      = len(imagenet_clstoidx)
	#for idx, item in enumerate(natsorted(glob('../../../data/100-shot-grumpy_cat/*')), start=0):
	#	clsname = os.path.basename(item)
	#	clstoidx[clsname] = idx
	#	idxtocls[idx] = clsname

	# data_path   = '../data/100-shot-panda/*' 
	data_path   = 'cropped_jpg/*'
	# train_batch = load_complete_data(data_path, data_path, input_res=256, batch_size=batch_size, shuffle_buffer_size=100)
	train_batch = load_batch(data_path, res=256, batch_size=batch_size)
	lr = 3e-4

	with mirrored_strategy.scope():
		model        = DCGAN()
		model_gopt   = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.2, beta_2=0.5)
		model_copt   = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.2, beta_2=0.5)
		ckpt         = tf.train.Checkpoint(step=tf.Variable(1), model=model, gopt=model_gopt, copt=model_copt)
		ckpt_manager = tf.train.CheckpointManager(ckpt, directory=exp_dir+'/ckpt/gan', max_to_keep=5)
		ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

	# print(ckpt.step.numpy())
	START         = int(ckpt.step.numpy()) // len(train_batch)
	EPOCHS        = 3847#66
	model_freq    = 13#20#7#2#40
	t_visfreq     = 13#20#7#2#1500#40
	latent        = tf.random.uniform(shape=(8, latent_dim), minval=-1, maxval=1)

	# wandb.config.update({
	# 				"learning_rate": lr,
	# 				"epochs"       : EPOCHS,
	# 				"batch_size"   : batch_size,
	# 				"adam_beta_1"  : 0.2,
	# 				"adam_beta_2"  : 0.5,
	# 				"distribution" : 'uniform(-1, 1)',
	# 				'cnn_filter'   : 3,
	# 				'gen_act'      : 'leaky_relu',
	# 				'dis_act'      : 'leaky_relu_with_0.2',
	# 				'diff_aug'     : 'color, translation',
	# 				'conv_bias'    : 'False',
	# 				'dis_type'     : 'Hinge Loss',
	# 				'value_norm'   : 'BatchNorm [G, D]',
	# 				'weight_norm'  : 'SpecNorm [G]'
	# 				})
	
	if ckpt_manager.latest_checkpoint:
		print('Restored from last checkpoint epoch: {0}'.format(START))

	for epoch in tqdm(range(START, EPOCHS)):
		t_gloss = tf.keras.metrics.Mean()
		t_closs = tf.keras.metrics.Mean()

		for idx, (X, C) in enumerate((train_batch), start=1):
		# for idx, path in enumerate(tqdm(train_batch), start=1):
			# X, C = load_data(path, dt=clstoidx, res=128)
			# print(X.shape)
			#C = tf.map_fn(get_code, C, fn_output_signature=tf.float32)
			gloss, closs = dist_train_step(mirrored_strategy, model, model_gopt, model_copt, X, latent_dim, batch_size)
			gloss = tf.reduce_mean(gloss)
			closs = tf.reduce_mean(closs)
			t_gloss.update_state(gloss)
			t_closs.update_state(closs)
			ckpt.step.assign_add(1)
			if (idx%model_freq)==0:
				ckpt_manager.save()
			if (idx%t_visfreq)==0:
				# vis(model, int(ckpt.step.numpy()), X, latent, exp_dir, wandb)
				vis(model, int(ckpt.step.numpy()), X, latent, exp_dir, None)

			# wandb.log({'iter_gen_loss': gloss, 'iter_dis_loss': closs})

			print('t_gloss: {0}\tt_closs: {1}'.format(gloss, closs))
			# break

		with open(exp_dir+'/log.txt', 'a') as file:
			file.write('Epoch: {0}\tT_gloss: {1}\tT_closs: {2}\n'.format(epoch, t_gloss.result(), t_closs.result()))
		print('Epoch: {0}\tT_gloss: {1}\tT_closs: {2}'.format(epoch, t_gloss.result(), t_closs.result()))
		# break
		# wandb.log({'epoch': epoch, 'epoch_gen_loss': gloss, 'epoch_dis_loss': closs})
