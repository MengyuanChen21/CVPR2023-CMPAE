import argparse


def build_args():
    feat_dir = 'data/feats/'
    anno_dir = 'data/annotations/'

    parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio-Visual Video Parsing')
    # Path parameters
    parser.add_argument('--audio_dir', type=str, default=feat_dir + 'vggish/', help='audio dir')
    parser.add_argument('--video_dir', type=str, default=feat_dir + 'res152/', help='video dir')
    parser.add_argument('--st_dir', type=str, default=feat_dir + 'r2plus1d_18/', help='video dir')
    parser.add_argument('--label_train', type=str, default=anno_dir + 'AVVP_train.csv', help='weak train csv file')
    parser.add_argument('--label_val', type=str, default=anno_dir + 'AVVP_val_pd.csv', help='weak val csv file')
    parser.add_argument('--label_test', type=str, default=anno_dir + 'AVVP_test_pd.csv', help='weak test csv file')
    parser.add_argument('--eval_audio', type=str, default=anno_dir + 'AVVP_eval_audio.csv')
    parser.add_argument('--eval_video', type=str, default=anno_dir + 'AVVP_eval_visual.csv')
    parser.add_argument("--model_save_dir", type=str, default='save/', help="model save dir")
    parser.add_argument('--noise_ratio_file', type=str, default='noise_ratios.npz')

    # Training parameters
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('-g', '--group_name', type=str, default='default', help='name of experiment group')
    parser.add_argument('-e', '--exp_name', type=str, default='default', help='experiment name')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--warm_up_epoch', type=float, default=0.9, help='warm-up epochs')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='adam')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr_step_size', type=int, default=6)
    parser.add_argument('--lr_gamma', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--log_interval', type=int, default=700, help='how many batches for logging training status')

    # Model parameters
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--v_thres', type=float, default=1.8)
    parser.add_argument('--a_thres', type=float, default=0.6)
    parser.add_argument('--a_smooth', type=float, default=1.0)
    parser.add_argument('--v_smooth', type=float, default=0.9)
    parser.add_argument('--audio_weight', type=float, default=2.0)
    parser.add_argument('--visual_weight', type=float, default=0.6)
    parser.add_argument('--video_weight', type=float, default=0.6)
    parser.add_argument('--nce_weight', type=float, default=1.0)
    parser.add_argument('--clamp', type=float, default=1e-7)
    parser.add_argument('--nce_smooth', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.2, help='feature temperature number')

    parser.add_argument('--mutual-weight', type=float, default=1.0)
    parser.add_argument('--mutual-eta', type=float, default=1.0)
    parser.add_argument('--mutual-lamb', type=float, default=1.0)

    parser.add_argument('--fuse_ratio', type=float, default=0.7)

    parser.add_argument('--start_class', type=int, default=0)
    parser.add_argument('--early_save_epoch', type=int, default=8)

    # Action parameters
    parser.add_argument('-s', '--not_save', action='store_true', help='whether to save model')
    parser.add_argument('-w', '--without_wandb', action='store_true', help='whether to use wandb')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'select_thresholds'],
                        help='with mode to use')
    args = parser.parse_args()
    return args
