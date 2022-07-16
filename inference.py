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
import cv2

tf.random.set_seed(45)
np.random.seed(45)

# wandb.init(project='DCGAN_DiffAug_EDDisc_imagenet_128', entity="prajwal_15")

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '1'
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
	
	# gpus = tf.config.list_physical_devices('GPU')
	# mirrored_strategy = tf.distribute.MirroredStrategy(devices=['/GPU:1', '/GPU:2'], 
	# 	cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
	# n_gpus = mirrored_strategy.num_replicas_in_sync
	# print(n_gpus)

	exp_dir = 'experiments/'
	batch_size = 64
	latent_dim = 128
	
	# with open('../dataset/imagenet_clstoidx.pkl', 'rb') as file:
	# 	imagenet_clstoidx = pickle.load(file)
	# with open('../dataset/imagenet_idxtocls.pkl', 'rb') as file:
	# 	imagenet_idxtocls = pickle.load(file)
	# n_class      = len(imagenet_clstoidx)
	# for idx, item in enumerate(natsorted(glob('data/ImageNet10/*')), start=0):
	# 	clsname = os.path.basename(item)
	# 	clstoidx[clsname] = idx
	# 	idxtocls[idx] = clsname

	data_path   = '../data/100-shot-panda/*'
	# train_batch = load_complete_data(data_path, data_path, input_res=128, batch_size=batch_size, shuffle_buffer_size=100)
	train_batch = load_batch(data_path, res=128, batch_size=batch_size)
	lr = 3e-4

	# with mirrored_strategy.scope():
	model        = DCGAN()
	model_gopt   = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.2, beta_2=0.5)
	model_copt   = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.2, beta_2=0.5)
	ckpt         = tf.train.Checkpoint(step=tf.Variable(1), model=model, gopt=model_gopt, copt=model_copt)
	ckpt_manager = tf.train.CheckpointManager(ckpt, directory=exp_dir+'/ckpt/gan', max_to_keep=10)
	ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

	# print(ckpt.step.numpy())
	START         = int(ckpt.step.numpy()) // len(train_batch) + 1
	EPOCHS        = 670#66
	model_freq    = 200#40
	t_visfreq     = 200#1500#40

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


	if not os.path.isdir('inference_result/'):
		os.makedirs('inference_result/')

	for _ in tqdm(range(1000)):
		latent = tf.random.uniform(shape=(1, latent_dim), minval=-1, maxval=1)
		fake_img = model.gen(latent, training=False)
		fake_img = fake_img[0].numpy()
		fake_img = np.uint8(np.clip(255*(fake_img * 0.5 + 0.5), 0.0, 255.0))
		fake_img = cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR)
		cv2.imwrite('inference_result/{}.png'.format(_), fake_img)
