import argparse
import os
import tensorflow as tf
from model import semi_stargan

# argument parser
parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase',          type=str,   default='train')
parser.add_argument('--dataset',        type=str,   default='celebA')
parser.add_argument('--data_dir',       type=str,   default=os.path.join('.','data','celebA'))
parser.add_argument('--log_dir',        type=str,   default='log')
parser.add_argument('--ckpt_dir',       type=str,   default='checkpoint')
parser.add_argument('--sample_dir',     type=str,   default='sample')
parser.add_argument('--test_dir',       type=str,   default='test')
parser.add_argument('--epoch',          type=int,   default=20)
parser.add_argument('--batch_size',     type=int,   default=8)
parser.add_argument('--image_size',     type=int,   default=128)
parser.add_argument('--image_channel',  type=int,   default=3)
# number of filters
parser.add_argument('--nf',             type=int,   default=64) 
parser.add_argument('--n_label',        type=int,   default=6)
parser.add_argument('--lambda_gp',      type=int,   default=10)
parser.add_argument('--lambda_cls',     type=float, default=1)
parser.add_argument('--lambda_rec',     type=int,   default=10)
parser.add_argument('--lambda_id',      type=float, default=1)
parser.add_argument('--lambda_adv',     type=int,   default=1)
parser.add_argument('--lambda_ucls',    type=int,   default=1)
# learning_rate
parser.add_argument('--lr',             type=float, default=0.0001) 
parser.add_argument('--beta1',          type=float, default=0.5)
parser.add_argument('--continue_train', type=bool,  default=False)
# number of iterations to save files
parser.add_argument('--snapshot',       type=int,   default=500)
# number of iterations to test auxiliary classifier accuracy
parser.add_argument('--snapshot_test',  type=int,   default=5000) 
parser.add_argument('--binary_attrs',   type=str,   default='100')
parser.add_argument('--d_steps',        type=int,   default=5)
parser.add_argument('--c_method',       type=str,   default='Sigmoid')


args = parser.parse_args()

def main(_):
    
    assets_dir = os.path.join(
        '.','assets','label{}_img{}_{}'.format(
            args.n_label, args.image_size, args.dataset))

    args.log_dir = os.path.join(assets_dir, args.log_dir)
    args.ckpt_dir = os.path.join(assets_dir, args.ckpt_dir)
    args.sample_dir = os.path.join(assets_dir, args.sample_dir)
    args.test_dir = os.path.join(assets_dir, args.test_dir)
    
    if args.n_label == 3:
        args.attr_keys = ['Black_Hair','Blond_Hair','Brown_Hair']
    else:
        args.attr_keys = ['Black_Hair','Blond_Hair','Brown_Hair', 'Male', 'Young']
    
    # make directory if not exist
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = semi_stargan(sess,args)
        if args.phase == 'train':
            model.train()
        elif args.phase == 'test':
            model.test()
        elif args.phase == 'test_all':
            model.test_all()
        elif args.phase == 'aux_test':
            model.test_aux_accuracy()
        else:
            raise ValueError(
                "Phase {} does not exist".format(args.phase))
            

# run main function
if __name__ == '__main__':
    tf.app.run()