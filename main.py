import argparse
import glob
import tensorflow as tf

from gan import GRGAN

def parse_args():
    # Creates the argument parser
    parser = argparse.ArgumentParser(description='GAN that removes glasses from pictures')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size to be used in training')
    parser.add_argument(
        '--steps',
        type=int,
        default=10000,
        help='Number of training steps to be performed')
    parser.add_argument(
        '--infer',
        help=
        'Performs inference for all images in the samples directory using the latest stored model',
        action='store_true')
    parser.add_argument(
        '--log_dir',
        default='log',
        help='Directory where all the log files reside')
    parser.add_argument(
        '--model_dir',
        default='checkpoint',
        help='Directory where the trained model is saved')
    parser.add_argument(
        '--samples_dir',
        default='samples',
        help='Directory of the images used for validation/inference')
    parser.add_argument(
        '--output_dir',
        default='output',
        help='Directory where the output of the inference will be stored')
    parser.add_argument(
        '--no_glasses_dir',
        default='training',
        help='Directory of the images without glasses for training')
    parser.add_argument(
        '--glasses_dir',
        default='noise',
        help='Directory of the images with glasses for training')

    return parser.parse_args()


def main():
    # Gets parsed arguments
    args = parse_args()
    glasses_dir = args.glasses_dir
    no_glasses_dir = args.no_glasses_dir
    samples_dir = args.samples_dir
    output_dir = args.output_dir
    model_dir = args.model_dir
    log_dir = args.log_dir

    if args.infer:
        # Inference mode
        with tf.Session() as sess:
            # Instantiates GAN class with batch size 1
            network = GRGAN([128, 128, 1], 5e-6, 2e-4, 1)
            # Builds gan model
            network.build_model(
                glob.glob(glasses_dir + '/*.jpg'),
                glob.glob(no_glasses_dir + '/*.jpg'),
                glob.glob(samples_dir + '/*.jpg'))
            # Gets number of images that need to be inferred
            n = len(glob.glob(samples_dir + '/*.jpg'))
            # Loads the latest trained model
            network.load(sess, model_dir)
            for i in range(n):
                # Performs an inference step
                filename, output_str = network.inference_step(sess)
                # Extracts the original filename
                filename = filename[0].split(b'/')[-1].decode("utf-8")
                print(filename)
                # Writes the refined image in the output directory
                output_f = open(output_dir + "/" + filename, "wb")
                output_f.write(output_str[0])
        return
    # Training mode
    with tf.Session() as sess:
        # Instantiates GAN class
        network = GRGAN([128, 128, 1], 5e-6, 2e-4, args.batch_size)
        # Builds model
        network.build_model(
            glob.glob(glasses_dir + '/*.jpg'),
            glob.glob(no_glasses_dir + '/*.jpg'),
            glob.glob(samples_dir + '/*.jpg'))
        # Initializes variables
        tf.global_variables_initializer().run()
        # Creates the log writer
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        # Performs pretrain
        network.pretraining_step(sess, writer, d_steps=300, g_steps=60)
        for i in range(args.steps):
            # Performs a training step
            network.training_step(sess, writer)
            if i % 10 == 0:
                # Runs inference on the validation set every 10 steps
                network.validation_step(sess, writer)
                print(i)
            if i % 1000 == 0:
                # Saves model every 1000 steps
                network.save(sess, model_dir, i)


if __name__ == '__main__':
    main()
