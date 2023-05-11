import os
import os.path as osp
import numpy as np
import tensorflow as tf
import vggish_input
import vggish_params
import vggish_slim
import argparse
from mmcv import ProgressBar
import random


def extract(checkpoint_path, num_secs, audio_dir, npy_dir):
    # Paths to downloaded VGGish files.
    # pca_params_path = 'vggish_pca_params.npz'
    # freq = 1000
    # sr = 44100
    # path of audio files and AVE annotation
    lis = os.listdir(audio_dir)
    random.shuffle(lis)
    len_data = len(lis)

    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        vggish_slim.define_vggish_slim()
        vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)

        pb = ProgressBar(len_data)
        pb.start()
        for n in range(len_data):
            '''feature learning by VGG-net trained by audioset'''
            outfile = os.path.join(npy_dir, os.path.splitext(lis[n])[0] + '.npy')
            if os.path.exists(outfile):
                pb.update()
                continue
            audio_index = os.path.join(audio_dir, lis[n])  # path of your audio files

            try:
                input_batch = vggish_input.wavfile_to_examples(audio_index)
            except:
                pb.update()
                continue
            np.testing.assert_equal(
                input_batch.shape,
                [num_secs, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS])

            [embedding_batch] = sess.run([embedding_tensor],
                                         feed_dict={features_tensor: input_batch})

            # save npy file
            np.save(outfile, embedding_batch)
            pb.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', dest='audio_dir', type=str, default='data/LLP_dataset/audio')
    parser.add_argument('--out_dir', dest='out_dir', type=str, default='data/feats/vggish')
    parser.add_argument('--ckpt_path', dest='ckpt_path', type=str, default='feature_extractor/vggish_model.ckpt')
    parser.add_argument('--num_secs', dest='num_secs', type=int, default=10)
    parser.add_argument("--gpu", dest='gpu', type=str, default='9')
    # parser.add_argument('--log_dir', dest='log_dir', type=str, default='log/extract_audio_feat')

    args = parser.parse_args()
    audio_dir = args.audio_dir
    out_dir = args.out_dir
    ckpt_path = args.ckpt_path
    num_secs = args.num_secs
    # log_dir = args.log_dir

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # set gpu number

    extract(ckpt_path, num_secs, audio_dir, out_dir)
