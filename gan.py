import tensorflow as tf
import numpy as np
import os

from history import History


class GRGAN:
    def __init__(self,
                 input_size,
                 learning_rate_d,
                 learning_rate_g,
                 batch_size,
                 history_size=100):
        """ Creates Glasses Removal GAN class

        Arguments:
            input_size (list): Shape of the input image
            learning_rate_d (float): Learning rate of the discriminator
            learning_rate_g (float): Learning rate of the refiner
            batch_size (int): Batch size
            history_size (int): Size of the history of refined images"""
        self.input_size = input_size
        self.learning_rate_d = learning_rate_d
        self.learning_rate_g = learning_rate_g
        self.batch_size = batch_size
        self.history = History(history_size)
        self.last_run = []

    def generator(self, source, reuse=True, is_training=True):
        # Initializes the refiner
        with tf.variable_scope("generator", reuse=reuse):
            conv_1 = tf.layers.conv2d(
                source,
                filters=64,
                kernel_size=5,
                activation=tf.nn.relu,
                padding='same',
                trainable=is_training,
                name='g_conv_1')

            last_output = conv_1

            # Builds ResNet blocks
            for i in range(6):
                resnet_1 = tf.layers.conv2d(
                    last_output,
                    filters=64,
                    kernel_size=5,
                    activation=tf.nn.relu,
                    padding='same',
                    trainable=is_training,
                    name='g_resnet_{}'.format(i * 2))

                resnet_2 = tf.layers.conv2d(
                    resnet_1,
                    filters=64,
                    kernel_size=5,
                    activation=tf.nn.relu,
                    padding='same',
                    trainable=is_training,
                    name='g_resnet_{}'.format(i * 2 + 1))

                last_output = resnet_2 + last_output

            refined = tf.layers.conv2d(
                last_output,
                filters=1,
                kernel_size=1,
                activation=tf.nn.relu,
                padding='same',
                trainable=is_training,
                name='g_conv_last')

            return refined

    def discriminator(self, source, reuse=True, is_training=True):
        # Initializes discriminator
        with tf.variable_scope("discriminator", reuse=reuse):
            net = tf.layers.conv2d(
                inputs=source,
                filters=96,
                kernel_size=5,
                activation=tf.nn.relu,
                trainable=is_training,
                padding="same",
                name="d_conv1")
            net = tf.layers.max_pooling2d(
                inputs=net, pool_size=2, strides=2, name="d_maxp1")
            net = tf.layers.conv2d(
                inputs=net,
                filters=192,
                kernel_size=3,
                activation=tf.nn.relu,
                trainable=is_training,
                padding="same",
                name="d_conv2")
            net = tf.layers.max_pooling2d(
                inputs=net, pool_size=2, strides=2, name="d_maxp2")
            net = tf.layers.conv2d(
                inputs=net,
                filters=384,
                kernel_size=3,
                activation=tf.nn.relu,
                trainable=is_training,
                padding="same",
                name="d_conv3")
            net = tf.layers.max_pooling2d(
                inputs=net, pool_size=2, strides=2, name="d_maxp3")
            net = tf.layers.conv2d(
                inputs=net,
                filters=256,
                kernel_size=3,
                activation=tf.nn.relu,
                trainable=is_training,
                padding="same",
                name="d_conv4")
            net = tf.layers.conv2d(
                inputs=net,
                filters=256,
                kernel_size=3,
                activation=tf.nn.relu,
                trainable=is_training,
                padding="same",
                name="d_conv5")
            net = tf.layers.max_pooling2d(
                inputs=net, pool_size=2, strides=2, name="d_maxp4")
            out_logit = tf.layers.conv2d(
                inputs=net,
                filters=2,
                kernel_size=5,
                activation=None,
                trainable=is_training,
                name="d_conv7")

            out = tf.nn.softmax(out_logit)

            return out, out_logit, net

    def build_model(self, fake_input, real_input, sample_input):
        # Builds model
        # Defines history images placeholder
        self.history_input = tf.placeholder(
            tf.float32, [self.batch_size // 2] + self.input_size)

        # Defines summary string for the history input
        self.hi = tf.summary.image("history", self.history_input)

        # Defines tf.Dataset for images without glasses
        real_dataset = tf.data.Dataset.from_tensor_slices(real_input)
        # Defines tf.Dataset for images with glasses
        fake_dataset = tf.data.Dataset.from_tensor_slices(fake_input)
        # Defines tf.Dataset for validation images
        sample_dataset = tf.data.Dataset.from_tensor_slices(sample_input)

        # Decodes jpeg, crops input image and makes batches
        real_dataset = real_dataset.map(lambda x: tf.read_file(x)).map(
            lambda x: tf.image.decode_jpeg(x, channels=1)).map(
                lambda x: tf.slice(x, [44, 22, 0], [128, 128, 1])).repeat(
                ).batch(self.batch_size // 2)
        fake_dataset = fake_dataset.map(lambda x: tf.read_file(x)).map(
            lambda x: tf.image.decode_jpeg(x, channels=1)).map(
                lambda x: tf.slice(x, [44, 22, 0], [128, 128, 1])).repeat(
                ).batch(self.batch_size)
        sample_dataset = sample_dataset.map(
            lambda x: (x, tf.read_file(x))).map(
                lambda y, x: (y, tf.image.decode_jpeg(x, channels=1))).map(
                    lambda y, x: (y, tf.slice(x, [44, 22, 0], [128, 128, 1]))
                ).repeat().batch(self.batch_size)

        # Defines iterators for each dataset
        real_iterator = real_dataset.make_one_shot_iterator()
        fake_iterator = fake_dataset.make_one_shot_iterator()
        sample_iterator = sample_dataset.make_one_shot_iterator()

        # Defines refiner inputs for training and inference
        self.generator_inputs = fake_iterator.get_next() / 255
        self.inf_filename, inf_input = sample_iterator.get_next()
        self.generator_inference_inputs = inf_input / 255

        # Defines discriminator inputs for training and pretraining
        discriminator_memory = self.history_input
        self.discriminator_training_input = tf.concat(
            [real_iterator.get_next() / 255, discriminator_memory], 0)
        self.discriminator_pretraining_input = tf.concat(
            [real_iterator.get_next() / 255, discriminator_memory], 0)

        # Defines summary string for the validation input
        self.g_inference_input_summary = tf.summary.image(
            "inference_input",
            self.generator_inference_inputs * 255,
            max_outputs=64)

        # Defines refiner outputs and summary strings
        self.g_training_output = self.generator(
            self.generator_inputs, is_training=True, reuse=False)
        print(self.g_training_output.shape)
        g_inference_output = self.generator(
            self.generator_inference_inputs, is_training=False, reuse=True)
        encode = lambda x: tf.image.encode_jpeg(tf.cast(x * 255, tf.uint8))
        g_inference_output_jpg = tf.minimum(g_inference_output, 1.)
        g_inference_output_jpg = tf.maximum(g_inference_output_jpg, 0.)
        self.g_inference_output_jpeg = tf.map_fn(
            encode, g_inference_output_jpg, dtype=tf.string)
        self.g_inference_output_summary = tf.summary.image(
            "inference_output", g_inference_output * 255, max_outputs=64)

        # Defines discriminator pretraining outputs
        d_pretraining_output, d_pretraining_output_logits, _ = self.discriminator(
            self.discriminator_pretraining_input,
            is_training=True,
            reuse=False)

        # Defines discriminator training outputs
        d_training_output, d_training_output_logits, _ = self.discriminator(
            self.discriminator_training_input,
            is_training=True,
            reuse=True)
        d_refined_output, d_refined_output_logits, _ = self.discriminator(
            self.g_training_output, is_training=True, reuse=True)
        d_inference_output, d_inference_output_logits, _ = self.discriminator(
            self.g_training_output, is_training=False, reuse=True)

        # Defines discriminator losses and their summaries
        d_pretraining_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=d_pretraining_output_logits,
                labels=tf.concat([
                    tf.ones([32, 4, 4], dtype=tf.int32),
                    tf.zeros([32, 4, 4], dtype=tf.int32)
                ], 0)))
        self.d_pretraining_loss_summary = tf.summary.scalar("pretrain_discriminator", d_pretraining_loss)
        d_training_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=d_training_output_logits,
                labels=tf.concat([
                    tf.ones([32, 4, 4], dtype=tf.int32),
                    tf.zeros([32, 4, 4], dtype=tf.int32)
                ], 0)))
        d_refined_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=d_training_output_logits, labels=tf.zeros([64, 4, 4], dtype=tf.int32)))
        self.d_loss = tf.reduce_mean([d_training_loss, d_refined_loss])
        self.d_loss_summary = tf.summary.scalar("discriminator_loss",
                                                self.d_loss)

        # Defines refiner losses and their summaries
        g_adversarial_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=d_inference_output_logits,
            labels=tf.ones([64, 4, 4], dtype=tf.int32))
        mask = np.ones([1, 128, 128, 1])
        mask[:, 34:94] *= 1e-4
        g_regularization_loss = tf.losses.mean_squared_error(
            self.generator_inputs, self.g_training_output, weights=mask)
        self.g_loss = tf.reduce_mean(
            g_adversarial_loss) + 10 * g_regularization_loss
        self.g_loss_summary = tf.summary.scalar("generator_loss",
                                                self.g_loss)

        # Gets all trainable variables
        trainable_vars = tf.trainable_variables()
        # Gets discriminator trainable variables
        d_vars = [
            var for var in trainable_vars if 'd_' in var.name
        ]
        # Gets refiner trainable variables
        g_vars = [
            var for var in trainable_vars if 'g_' in var.name
        ]

        # Defines optimizers
        g_optimizer = tf.train.AdamOptimizer(self.learning_rate_g, beta1=0.5)
        d_optimizer = tf.train.AdamOptimizer(self.learning_rate_d, beta1=0.5)

        # Defines pretraining optimization ops
        self.g_pretrain_optimize = g_optimizer.minimize(
            g_regularization_loss, var_list=g_vars)
        self.d_pretrain_optimize = d_optimizer.minimize(
            d_pretraining_loss, var_list=d_vars)

        # Defines training optimization ops
        self.g_optimize = g_optimizer.minimize(
            self.g_loss, var_list=g_vars)
        self.d_optimize = d_optimizer.minimize(
            self.d_loss, var_list=d_vars)

        # Creates model saver
        self.saver = tf.train.Saver(max_to_keep=10)
        self.counter = 0

    def pretraining_step(self, sess, writer, d_steps=1, g_steps=2):
        # Performs pretraining
        print('Beginning pretrain...')
        last_run = []
        for i in range(g_steps):
            _, run = sess.run(
                [self.g_pretrain_optimize, self.g_training_output])
            self.history.update_history(run[:self.batch_size // 2])
            last_run.append(run[self.batch_size // 2:])
        for i in range(d_steps):
            np.random.shuffle(last_run[i % g_steps])
            fake_input = np.concatenate(
                (self.history.get_images(self.batch_size // 4),
                 last_run[i % g_steps][:self.batch_size // 4]))
            _, summary = sess.run(
                [self.d_pretrain_optimize, self.d_pretraining_loss_summary],
                feed_dict={self.history_input: fake_input})
            writer.add_summary(summary, i)
        print('Ending pretrain...')

    def training_step(self, sess, writer, d_steps=1, g_steps=2):
        # Performs a training step
        last_run = []
        self.counter += 1
        for i in range(g_steps):
            _, run, summary = sess.run([self.g_optimize, self.g_training_output, self.g_loss_summary])
            if i == g_steps - 1:
                writer.add_summary(summary, self.counter)
            last_run = run
        for i in range(d_steps):
            np.random.shuffle(last_run)
            history_input = np.concatenate((self.history.get_images(
                self.batch_size // 4), last_run[:self.batch_size // 4]))
            _, summary, hi = sess.run(
                [self.d_optimize, self.d_loss_summary, self.hi],
                feed_dict={self.history_input: history_input})
            writer.add_summary(hi, self.counter)
            writer.add_summary(summary, self.counter)
        self.history.update_history(last_run[self.batch_size // 2:])

    def validation_step(self, sess, writer):
        # Performs a validation step
        inp, out = sess.run([self.g_inference_input_summary, self.g_inference_output_summary])
        writer.add_summary(inp, self.counter)
        writer.add_summary(out, self.counter)

    def inference_step(self, sess):
        # Performs an inference step
        return sess.run([self.inf_filename, self.g_inference_output_jpeg])

    def save(self, sess, cache_dir, step):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.saver.save(
            sess, os.path.join(cache_dir, 'noglasses.model'), global_step=step)

    def load(self, sess, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(
                next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
        return False, 0

    def load_ckpt(self, sess, checkpoint_path):
        self.saver.restore(sess, checkpoint_path)
