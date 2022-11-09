import tensorflow as tf
from tensorflow.keras import Model, layers, backend
from tensorflow.keras.constraints import Constraint
from losses import disc_hinge, disc_loss, gen_loss, gen_hinge
from diff_augment import diff_augment
from tensorflow_addons.layers import SpectralNormalization

tf.random.set_seed(45)
# np.random.seed(45)

# https://inf.news/en/tech/ef67a9c98d34e20c1c718239908d9399.html
class PixelNorm(layers.Layer):
	def __init__(self, epsilon=1e-8):
		super(PixelNorm, self).__init__()
		self.epsilon = epsilon

	def call(self, x):
		return tf.divide(x, tf.math.sqrt( tf.reduce_mean( x ** 2, axis=-1, keepdims=True ) + self.epsilon ))

class Generator(Model):
	def __init__(self):
		super(Generator, self).__init__()
		# filters   = [  1024, 512, 256, 128,  64, 32, 16,   8]
		# strides   = [     4,   2,   2,   2,   2,  2,  2,   2]
		filters   = [  1024, 512, 512, 256, 256, 128, 64, 16]
		strides   = [     4,   2,   2,   2,   2,  2,   2,  1]
		# filters   = [  1024, 512, 256, 128,  64, 32, 16]
		# strides   = [     4,   2,   2,   2,   2,  2,  2]
		# filters   = [  1024, 512, 256, 128,  64, 32]
		# strides   = [     4,   2,   2,   2,   2,  2]
		self.cnn_depth  = len(filters)
		
		# Hyperparameter:
		# If only conv  : mean=0.0, var=0.02
		# If using bnorm: mean=1.0, var=0.02
		self.conv  = [layers.Conv2DTranspose(\
					  filters=filters[idx], kernel_size=3,\
		              strides=strides[idx], padding='same',\
		              kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02),\
					  use_bias=False)\
					  for idx in range(self.cnn_depth)]

		self.act   = [layers.LeakyReLU() for idx in range(self.cnn_depth)]

		# self.bnorm = [layers.BatchNormalization() for idx in range(self.cnn_depth)]
		self.bnorm = [PixelNorm() for idx in range(self.cnn_depth)]

		self.last_conv = layers.Conv2D(filters=3, kernel_size=3,\
									   strides=1, padding='same',\
									   activation='tanh',\
									   kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02),\
									   use_bias=False)

	@tf.function
	def call(self, X):
		X = tf.expand_dims(tf.expand_dims(X, axis=1), axis=1)
		X = self.act[0]( self.conv[0]( X ) )

		for idx in range(1, self.cnn_depth):
			X = self.act[idx]( self.bnorm[idx]( self.conv[idx]( X ) ) )
			# X = self.bnorm[idx]( self.act[idx]( self.conv[idx]( X ) ) )
			# X = self.act[idx]( self.conv[idx]( X ) )
		X = self.last_conv(X)
		return X


class Discriminator(Model):
	def __init__(self):
		super(Discriminator, self).__init__()
		# filters    = [16, 32, 64, 128, 256, 512, 1024, 1]
		# strides    = [ 2,  2,  2,   2,   2,   2,    1, 1]
		filters    = [32, 64, 128, 256, 512, 1024, 1]
		strides    = [ 2,  2,   2,   2,   2,    1, 1]
		# filters    = [ 64, 128, 256, 512, 1024, 1]
		# strides    = [  2,   2,   2,   2,    1, 1]
		self.cnn_depth = len(filters)

		self.cnn_conv  = [layers.Conv2D(filters=filters[i], kernel_size=3,\
										strides=strides[i], padding='same',\
										kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02),\
										use_bias=False)\
										for i in range(self.cnn_depth)] 

		self.cnn_bnorm = [layers.BatchNormalization() for _ in range(self.cnn_depth)]

		self.cnn_act   = [layers.LeakyReLU(alpha=0.2) for _ in range(self.cnn_depth)]

		self.flat      = layers.Flatten()

		self.disc_out  = layers.Dense(units=1)

	@tf.function
	def call(self, x, condition=None):
		#x         = self.cnn_merge( x )
		#x         = self.cnn_exp( x )
		mem_bank   = []
		for layer_no in range(self.cnn_depth):
			# print(x.shape)
			x = self.cnn_act[layer_no]( self.cnn_bnorm[layer_no]( self.cnn_conv[layer_no]( x ) ) )
			# x = self.cnn_bnorm[layer_no]( self.cnn_act[layer_no]( self.cnn_conv[layer_no]( x ) ) )
			# x = self.cnn_act[layer_no]( self.cnn_conv[layer_no]( x ) )
			# if layer_no == 0:
			# 	mem_bank.append( x )
			# if layer_no == 1:
			# 	mem_bank.append( x )
			# x = self.cnn_act[layer_no]( self.cnn_conv[layer_no]( x ) )

		# reconst_x = self.autoenc( x )
		
		# condition = tf.expand_dims(tf.expand_dims(condition, axis=1), axis=1)
		# condition = tf.tile(condition, [1, x.shape[1], x.shape[1], 1])
		# x         = tf.concat([x, condition], axis=-1)

		# x = self.cnn_act[layer_no+1]( self.cnn_bnorm[layer_no+1]( self.cnn_conv[layer_no+1]( x ) ) )
		# x = self.cnn_bnorm[layer_no+1]( self.cnn_act[layer_no+1]( self.cnn_conv[layer_no+1]( x ) ) )
		# x = self.cnn_act[layer_no+1]( self.cnn_conv[layer_no+1]( x ) )

		# reconst_x = self.autoenc( x )
		reconst_x   = None

		# x = self.cnn_act[layer_no+2]( self.cnn_bnorm[layer_no+2]( self.cnn_conv[layer_no+2]( x ) ) )
		# reconst_x = self.autoenc( x, mem_bank )

		# x = self.final_act( x )
		# x = self.out( self.flat( x ) )
		x = self.disc_out( self.flat( x ) )

		return x, reconst_x


class Discriminator_128(Model):
	def __init__(self):
		super(Discriminator_128, self).__init__()
		# filters    = [16, 32, 64, 128, 256, 512, 1024, 1]
		# strides    = [ 2,  2,  2,   2,   2,   2,    1, 1]
		# filters    = [32, 64, 128, 256, 512, 1024, 1]
		# strides    = [ 2,  2,   2,   2,   2,    1, 1]
		filters    = [ 64, 128, 256, 512, 1024, 1]
		strides    = [  2,   2,   2,   2,    1, 1]
		self.cnn_depth = len(filters)

		self.cnn_conv  = [layers.Conv2D(filters=filters[i], kernel_size=3,\
										strides=strides[i], padding='same',\
										kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02),\
										use_bias=False)\
										for i in range(self.cnn_depth)] 

		self.cnn_bnorm = [layers.BatchNormalization() for _ in range(self.cnn_depth)]

		self.cnn_act   = [layers.LeakyReLU(alpha=0.2) for _ in range(self.cnn_depth)]

		self.flat      = layers.Flatten()

		self.disc_out  = layers.Dense(units=1)

	@tf.function
	def call(self, x, condition=None):

		for layer_no in range(self.cnn_depth):
			# print(x.shape)
			x = self.cnn_act[layer_no]( self.cnn_bnorm[layer_no]( self.cnn_conv[layer_no]( x ) ) )

		reconst_x   = None

		x = self.disc_out( self.flat( x ) )

		return x, reconst_x


class Discriminator_64(Model):
	def __init__(self):
		super(Discriminator_64, self).__init__()
		# filters    = [16, 32, 64, 128, 256, 512, 1024, 1]
		# strides    = [ 2,  2,  2,   2,   2,   2,    1, 1]
		# filters    = [32, 64, 128, 256, 512, 1024, 1]
		# strides    = [ 2,  2,   2,   2,   2,    1, 1]
		# filters    = [ 64, 128, 256, 512, 1024, 1]
		# strides    = [  2,   2,   2,   2,    1, 1]
		filters    = [ 128, 256, 512, 1024, 1]
		strides    = [   2,   2,   2,    1, 1]
		self.cnn_depth = len(filters)

		self.cnn_conv  = [layers.Conv2D(filters=filters[i], kernel_size=3,\
										strides=strides[i], padding='same',\
										kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02),\
										use_bias=False)\
										for i in range(self.cnn_depth)] 

		self.cnn_bnorm = [layers.BatchNormalization() for _ in range(self.cnn_depth)]

		self.cnn_act   = [layers.LeakyReLU(alpha=0.2) for _ in range(self.cnn_depth)]

		self.flat      = layers.Flatten()

		self.disc_out  = layers.Dense(units=1)

	@tf.function
	def call(self, x, condition=None):

		for layer_no in range(self.cnn_depth):
			# print(x.shape)
			x = self.cnn_act[layer_no]( self.cnn_bnorm[layer_no]( self.cnn_conv[layer_no]( x ) ) )
		reconst_x   = None

		x = self.disc_out( self.flat( x ) )

		return x, reconst_x

class DCGAN(Model):
	def __init__(self):
		super(DCGAN, self).__init__()
		self.gen      = Generator()
		self.disc     = Discriminator()
		self.disc_128 = Discriminator_128()
		self.disc_64  = Discriminator_64()

@tf.function
def dist_train_step(mirrored_strategy, model, model_gopt, model_copt, X, latent_dim=128, batch_size=128):

	diff_augment_policies = "color,translation"
	noise_vector          = tf.random.uniform(shape=(batch_size, latent_dim), minval=-1, maxval=1)
	noise_vector_2        = tf.random.uniform(shape=(batch_size, latent_dim), minval=-1, maxval=1)
	
	# @tf.function
	def train_step_disc(model, model_gopt, model_copt, X, latent_dim=128, batch_size=128):	
		with tf.GradientTape() as ctape:
			# noise_vector = tf.random.uniform(shape=(batch_size, latent_dim), minval=-1, maxval=1)
			# noise_vector = tf.random.uniform(shape=(batch_size, latent_dim), minval=-1, maxval=1)
			# noise_vector = tf.random.normal(shape=(batch_size, latent_dim))

			fake_img     = model.gen(noise_vector, training=False)

			X_aug        = diff_augment(X, policy=diff_augment_policies)
			fake_img     = diff_augment(fake_img, policy=diff_augment_policies)

			D_real, X_recon = model.disc(X_aug, training=True)
			D_fake, _       = model.disc(fake_img, training=True)

			X_aug    = tf.image.resize(X_aug, (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
			fake_img = tf.image.resize(fake_img, (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

			D_real_128, X_recon = model.disc_128(X_aug, training=True)
			D_fake_128, _       = model.disc_128(fake_img, training=True)

			X_aug    = tf.image.resize(X_aug, (64, 64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
			fake_img = tf.image.resize(fake_img, (64, 64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

			D_real_64, X_recon = model.disc_64(X_aug, training=True)
			D_fake_64, _       = model.disc_64(fake_img, training=True)

			# c_loss       = disc_loss(D_real, D_fake) +\
			# 			   tf.reduce_mean(tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)(X_aug, X_recon))
			# c_loss       = disc_hinge(D_real, D_fake) +\
			# 			   tf.reduce_mean(tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)(X_aug, X_recon))
			c_loss       = disc_hinge(D_real, D_fake) +\
						   disc_hinge(D_real_128, D_fake_128) +\
						   disc_hinge(D_real_64, D_fake_64)

		variables = model.disc.trainable_variables
		gradients = ctape.gradient(c_loss, variables)
		model_copt.apply_gradients(zip(gradients, variables))
		return c_loss

	# @tf.function
	def train_step_gen(model, model_gopt, model_copt, X, latent_dim=128, batch_size=128):
		with tf.GradientTape() as gtape:
			# noise_vector = tf.random.uniform(shape=(batch_size, latent_dim), minval=-1, maxval=1)
			# noise_vector = tf.random.normal(shape=(batch_size, latent_dim))
			
			fake_img_o    = model.gen(noise_vector, training=True)
			fake_img_2   = model.gen(noise_vector_2, training=True)
			#D_fake       = model.disc(fake_img, H_hat, training=False)

			fake_img     = diff_augment(fake_img_o, policy=diff_augment_policies)

			D_fake, _     = model.disc(fake_img, training=False)

			fake_img      = tf.image.resize(fake_img, (128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
			D_fake_128, _ = model.disc_128(fake_img, training=False)

			fake_img      = tf.image.resize(fake_img, (64, 64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
			D_fake_64, _  = model.disc_64(fake_img, training=False)
			# g_loss       = gen_loss(D_fake)
			g_loss       = gen_hinge(D_fake) +\
						   gen_hinge(D_fake_128) +\
						   gen_hinge(D_fake_64)

			mode_loss    = tf.divide(tf.reduce_mean(tf.abs(tf.subtract(fake_img_2, fake_img_o))),\
									tf.reduce_mean(tf.abs(tf.subtract(noise_vector_2, noise_vector)))
									)
			mode_loss   = tf.divide(1.0, mode_loss + 1e-5)
			total_loss   = g_loss + 1.0 * mode_loss
			# total_loss   = g_loss

		variables = model.gen.trainable_variables #+ model.gcn.trainable_variables
		gradients = gtape.gradient(total_loss, variables)
		model_gopt.apply_gradients(zip(gradients, variables))
		return g_loss

	per_replica_loss_disc = mirrored_strategy.run(train_step_disc, args=(model, model_gopt, model_copt, X, latent_dim, batch_size,))
	per_replica_loss_gen  = mirrored_strategy.run(train_step_gen, args=(model, model_gopt, model_copt, X, latent_dim, batch_size,))
	
	# print(per_replica_loss_disc)

	# print(mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss_disc, axis=0).numpy())

	discriminator_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss_disc, axis=None)
	generator_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss_gen, axis=None)
	return generator_loss, discriminator_loss
