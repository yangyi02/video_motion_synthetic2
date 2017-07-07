import argparse
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def parse_args():
    arg_parser = argparse.ArgumentParser(description='unsupervised motion', add_help=False)
    arg_parser.add_argument('--method', default='unsupervised')
    arg_parser.add_argument('--bidirection', action='store_true')

    arg_parser.add_argument('--train', action='store_true')
    arg_parser.add_argument('--test', action='store_true')
    arg_parser.add_argument('--test_gt', action='store_true')

    arg_parser.add_argument('--train_epoch', type=int, default=1000)
    arg_parser.add_argument('--test_epoch', type=int, default=10)
    arg_parser.add_argument('--test_interval', type=int, default=100)
    arg_parser.add_argument('--save_dir', default='./model')

    arg_parser.add_argument('--init_model_path', default='')
    arg_parser.add_argument('--display', action='store_true')

    arg_parser.add_argument('--learning_rate', type=float, default=0.001)

    arg_parser.add_argument('--batch_size', type=int, default=32)
    arg_parser.add_argument('--image_size', type=int, default=32)
    arg_parser.add_argument('--motion_range', type=int, default=1)
    arg_parser.add_argument('--num_frame', type=int, default=2)

    arg_parser.add_argument('--data', default='box')
    arg_parser.add_argument('--num_objects', type=int, default=1)
    arg_parser.add_argument('--bg_noise', type=float, default=0.5)
    arg_parser.add_argument('--fg_noise', type=float, default=0.1)
    arg_parser.add_argument('--bg_move', action='store_true')

    args = arg_parser.parse_args()

    return args

