import os
import numpy as np
import tensorflow as tf
from collections import namedtuple
from tqdm import tqdm
from glob import glob
import random
import math

from module import * 
from util import * 


class semi_stargan(object): # Y module
    def __init__(self,sess,args):
        #
        self.sess = sess
        self.dataset = args.dataset
        self.phase = args.phase # train or test
        self.data_dir = args.data_dir # ./data/celebA
        self.log_dir = args.log_dir # ./assets/log
        self.ckpt_dir = args.ckpt_dir # ./assets/checkpoint
        self.sample_dir = args.sample_dir # ./assets/sample
        self.test_dir = args.test_dir # ./assets/test
        self.epoch = args.epoch # 100
        self.batch_size = args.batch_size # 16
        self.batch_size_half = args.batch_size//2
        self.image_size = args.image_size # 64
        self.image_channel = args.image_channel # 3
        self.nf = args.nf # 64
        self.n_label = args.n_label # 10
        self.lambda_gp = args.lambda_gp
        self.lambda_cls = args.lambda_cls # 1
        self.lambda_rec = args.lambda_rec # 10
        self.lambda_ucls = args.lambda_ucls #10
        self.lr = args.lr # 0.0001
        self.beta1 = args.beta1 # 0.5
        self.continue_train = args.continue_train # False
        self.snapshot = args.snapshot # 100
        self.snapshot_test = args.snapshot_test
        self.binary_attrs = args.binary_attrs
        self.c_method = args.c_method
        self.training = args.phase == 'train'
        self.attr_keys = args.attr_keys
        self.continue_epoch = 0
        self.continue_iteration = 0
        
        OPTIONS = namedtuple('OPTIONS', ['batch_size', 'image_size', 'nf', 
            'n_label', 'lambda_gp'])
        self.options = OPTIONS(self.batch_size, self.image_size, self.nf, 
            self.n_label, self.lambda_gp)
        
        # select discriminator
        if self.n_label == 5:
            self.discriminator = semi_c_discriminator_Y_deep
        else:
            self.discriminator = semi_c_discriminator_Y

        # build model & make checkpoint saver 
        self.build_model()
        self.saver = tf.train.Saver()
        
    def build_model(self): 

        # [batch,h,w,c + n_cls]
        self.real_img = tf.placeholder(tf.float32,
            [None, self.image_size, self.image_size, self.image_channel],
            name='input_images')
        self.unlab_img = tf.placeholder(tf.float32,
            [None, self.image_size, self.image_size, self.image_channel],
            name='input_unlab_images')
        self.real_atr = tf.placeholder(tf.float32,
            [None, self.n_label], name='input_images_attributes')
        self.fake_atr = tf.placeholder(tf.float32,
            [None, self.n_label], name='target_images_attributes')
        
        
        self.epsilon = tf.placeholder(tf.float32, [None,1,1,1], 
            name='gp_random_num')
        self.lr_decay = tf.placeholder(tf.float32, None, name='lr_decay')
        self.pi_weight = tf.placeholder(tf.float32, None, name='pi_weight')
        self.lambda_id = tf.placeholder(tf.float32, None, name='lambda_id')

        self.mask = tf.reduce_sum(tf.cast(
            tf.equal(self.real_atr,self.fake_atr), tf.float32),axis=1)
        self.mask = tf.cast(tf.equal(self.mask,self.n_label),tf.float32)
        # generate image
        # real_A is already concatenate with n_cls ? 
        fake_atr_tile    = tf.tile(tf.reshape(self.fake_atr, 
            [-1,1,1,self.n_label]),[1,self.image_size,self.image_size,1])
        real_img_concat  = tf.concat((self.real_img, fake_atr_tile), axis=3)
        unlab_img_concat = tf.concat((self.unlab_img, fake_atr_tile), axis=3)
        all_img_concat = tf.concat((real_img_concat,unlab_img_concat),axis=0)
        
        
        all_fake_img = generator(all_img_concat, self.options, False, name='gen')
        self.fake_img = all_fake_img[:self.batch_size_half]
        self.fake_unlab_img = all_fake_img[self.batch_size_half:]

        # discriminate image
        # src: real or fake, cls: domain classification 
        all_real_img = tf.concat((self.real_img,self.unlab_img),axis=0)
        src_all_real_img , cls_all_real_img = self.discriminator(all_real_img, 
            self.options, False, training=self.training, name='disc')
        self.src_real_img  = src_all_real_img[:self.batch_size_half]
        self.src_unlab_img = src_all_real_img[self.batch_size_half:]
        self.cls_real_img  = cls_all_real_img[:self.batch_size_half]
        self.cls_unlab_img = cls_all_real_img[self.batch_size_half:]
        
        all_real_img.set_shape((
            self.batch_size,self.image_size,self.image_size,self.image_channel))
        
        with tf.name_scope("real_img_flipping"):
            all_real_img_list = tf.unstack(all_real_img)
            all_real_img_list = ([tf.image.flip_left_right(i) 
                for i in all_real_img_list])
            all_real_img_flip = tf.stack(all_real_img_list)

        src_all_real_img2 , cls_all_real_img2 = self.discriminator(
            all_real_img_flip, self.options, True, 
            training=self.training, name='disc')
        self.src_real_img2  = src_all_real_img2[:self.batch_size_half]
        self.src_unlab_img2 = src_all_real_img2[self.batch_size_half:]
        self.cls_real_img2  = cls_all_real_img2[:self.batch_size_half]
        self.cls_unlab_img2 = cls_all_real_img2[self.batch_size_half:]

        ###
        src_all_fake_img, cls_all_fake_img  = self.discriminator(all_fake_img, 
            self.options, True , training=self.training, name='disc')
        self.src_fake_img = src_all_fake_img[:self.batch_size_half]
        self.src_fake_unlab_img = src_all_fake_img[self.batch_size_half:]
        self.cls_fake_img = cls_all_fake_img[:self.batch_size_half]
        self.cls_fake_unlab_img = cls_all_fake_img[self.batch_size_half:]

        
        all_fake_img.set_shape((
            self.batch_size,self.image_size,self.image_size,self.image_channel))
        with tf.name_scope("fake_img_flipping"):
            all_fake_img_list = tf.unstack(all_fake_img)
            all_fake_img_list = ([tf.image.flip_left_right(i) 
                for i in all_fake_img_list])
            all_fake_img_flip = tf.stack(all_fake_img_list)
       
        
        src_all_fake_img2, cls_all_fake_img2 = self.discriminator(
            all_fake_img_flip, self.options, True , 
            training=self.training, name='disc')
        self.src_fake_img2 = src_all_fake_img2[:self.batch_size_half]
        self.src_fake_unlab_img2 = src_all_fake_img2[self.batch_size_half:]
        self.cls_fake_img2 = cls_all_fake_img2[:self.batch_size_half]
        self.cls_fake_unlab_img2 = cls_all_fake_img2[self.batch_size_half:]
        
        real_atr_tile   = tf.tile(tf.reshape(self.real_atr, 
            [-1,1,1,self.n_label]),[1,self.image_size,self.image_size,1])
        fake_img_concat = tf.concat((self.fake_img, real_atr_tile), axis=3)
        
        if self.c_method == 'Softmax':
            unlab_atr = tf.one_hot(
                tf.argmax(self.cls_unlab_img,axis=1),self.n_label)
        else:
            predicted = tf.nn.sigmoid(self.cls_unlab_img)
            unlab_atr = self.threshold(predicted)

        unlab_atr_tile = tf.tile(tf.reshape(unlab_atr, [-1,1,1,self.n_label]),
            [1,self.image_size,self.image_size,1])
        fake_unlab_img_concat = tf.concat(
            (self.fake_unlab_img, unlab_atr_tile), axis=3)
        
        fake_all_img_concat = tf.concat(
            (fake_img_concat,fake_unlab_img_concat),axis=0)
        recon_all_img = generator(fake_all_img_concat, 
            self.options, True, name='gen')
        self.recon_img = recon_all_img[:self.batch_size_half]
        self.recon_unlab_img = recon_all_img[self.batch_size_half:]
        
        
        ### sample
        self.fake_img_sample = generator(real_img_concat, 
            self.options, True, name='gen')
        fake_img_sample_concat = tf.concat(
            (self.fake_img_sample, real_atr_tile), axis=3)
        self.recon_img_sample = generator(fake_img_sample_concat, 
            self.options, True, name='gen')

        # loss
        ## discriminator loss ##
        ### adversarial loss
        self.gp_loss = wgan_gp_loss(all_real_img, all_fake_img, 
            self.options,self.discriminator)
        
        self.d_loss_fake = (tf.reduce_mean(src_all_fake_img)  
            + tf.reduce_mean(src_all_fake_img2))/2
        self.d_loss_real = (-tf.reduce_mean(src_all_real_img) 
            - tf.reduce_mean(src_all_real_img2))/2
        self.d_adv_loss = self.d_loss_fake + self.d_loss_real + self.gp_loss
            
        ### domain classification loss
        if self.c_method =='Softmax':
            self.real_cls_loss       = (
                cls_loss_SoftCE(self.cls_real_img , self.real_atr) + 
                cls_loss_SoftCE(self.cls_real_img2, self.real_atr))/2
            self.fake_cls_loss       = (
                cls_loss_SoftCE(self.cls_fake_img , self.fake_atr) + 
                cls_loss_SoftCE(self.cls_fake_img2, self.fake_atr))/2
            self.fake_unlab_cls_loss = (
                cls_loss_SoftCE(self.cls_fake_unlab_img,self.fake_atr) + 
                cls_loss_SoftCE(self.cls_fake_unlab_img2,self.fake_atr))/2
        else:
            self.real_cls_loss       = (
                cls_loss(self.cls_real_img , self.real_atr) + 
                cls_loss(self.cls_real_img2, self.real_atr))/2
            self.fake_cls_loss       = (
                cls_loss(self.cls_fake_img , self.fake_atr) + 
                cls_loss(self.cls_fake_img2, self.fake_atr))/2
            self.fake_unlab_cls_loss = (
                cls_loss(self.cls_fake_unlab_img,self.fake_atr) + 
                cls_loss(self.cls_fake_unlab_img2,self.fake_atr))/2

        self.unlab_pi_loss = (l2_loss(cls_all_real_img,cls_all_real_img2)+\
            l2_loss(cls_all_fake_img,cls_all_fake_img2))/2

        ### disc loss function
        self.d_loss = self.d_adv_loss \
                    + self.real_cls_loss \
                    + self.pi_weight *self.unlab_pi_loss 
        
        ## generator loss ##
        ### reconstruction loss
        self.recon_loss       = recon_loss(self.real_img , self.recon_img)
        self.identity_loss    = recon_loss_with_mask(
            self.real_img, self.fake_img, self.mask)
        self.recon_loss_unlab = recon_loss(self.unlab_img, self.recon_unlab_img)
        
        
        ### adv loss
        self.g_adv_loss = (
            -tf.reduce_mean(src_all_fake_img) 
            -tf.reduce_mean(src_all_fake_img2))/2
        
        ### gen loss function
        self.g_loss = self.g_adv_loss \
                    + self.lambda_rec * (self.recon_loss)  \
                    + self.lambda_cls * (self.fake_cls_loss 
                        + self.fake_unlab_cls_loss)/2 \
                    + self.lambda_id  * (self.identity_loss)

        # withoud identity loss
        self.g_loss2= self.g_adv_loss \
                    + self.lambda_rec * (self.recon_loss 
                        +self.recon_loss_unlab)/2  \
                    + self.lambda_cls * (self.fake_cls_loss 
                        + self.fake_unlab_cls_loss)/2
        
        # trainable variables
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'disc' in var.name]
        for var in self.d_vars: print(var.name)
        self.g_vars = [var for var in t_vars if 'gen' in var.name]
        for var in self.g_vars: print(var.name)
        # optimizer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.d_optim  = tf.train.AdamOptimizer(
                self.lr * self.lr_decay, beta1=self.beta1).minimize(
                self.d_loss, var_list=self.d_vars)
            self.g_optim  = tf.train.AdamOptimizer(
                self.lr * self.lr_decay, beta1=self.beta1).minimize(
                self.g_loss, var_list=self.g_vars)
            self.g_optim2 = tf.train.AdamOptimizer(
                self.lr * self.lr_decay, beta1=self.beta1).minimize(
                self.g_loss2, var_list=self.g_vars)
        
        self.src_real_img_test , self.cls_real_img_test = self.discriminator(
            self.real_img, self.options, True, training=self.training,name='disc')
        self.acc = self.compute_accuracy(
            self.cls_real_img_test,self.real_atr,self.c_method)
    
    def train(self):
        # summary setting
        self.summary()
        
        # load train data list & load attribute data
        data_files = load_data_list(self.data_dir)
        unlab_data_files = load_data_list(self.data_dir,phase='unlabel')
        self.attr_names, self.attr_list = attr_extract(
            self.data_dir)
        
        # variable initialize
        self.sess.run(tf.global_variables_initializer())
        
        # load or not checkpoint
        if self.continue_train and self.checkpoint_load():
            print(" [*] before training, Load SUCCESS ")
        else:
            print(" [!] before training, no need to Load ")
        
        #batch_idxs = len(data_files) // self.batch_size_half # 182599
        batch_idxs = len(unlab_data_files) // self.batch_size_half # 182599
        small_batch_idxs = len(data_files) // self.batch_size_half

        self.continue_epoch = self.continue_iteration//batch_idxs
        count = self.continue_epoch*batch_idxs

        num_iteration = batch_idxs*self.epoch
        rampup_length = num_iteration//3
        rampdown_length = rampup_length
        #train
        for epoch in range(self.epoch-self.continue_epoch):
            # get lr_decay
            epoch = epoch + self.continue_epoch
            if epoch < self.epoch / 2:
                lr_decay = 1.0
            else:
                lr_decay = float(self.epoch - epoch) / float(self.epoch / 2)
            print('lr',lr_decay)

            # data shuffle at the begining of an epoch
            
            np.random.shuffle(unlab_data_files)
            
            for idx in tqdm(range(batch_idxs)):
                count += 1
                pi_weight = self.rampup(count,rampup_length)
                unlab_data_list = (unlab_data_files[
                    idx * self.batch_size_half : (idx+1) * self.batch_size_half])
                if idx % small_batch_idxs == 0:
                    np.random.shuffle(data_files)
                sidx = idx%small_batch_idxs
                data_list = (data_files[
                    sidx * self.batch_size_half : (sidx+1) * self.batch_size_half])

                attr_list  = ([self.attr_list[os.path.basename(val)] 
                    for val in data_list])
                attr_list_ = np.copy(attr_list)
                np.random.shuffle(attr_list_)
                attr_list_[0] = attr_list[0] #make sure at least one identity
                
                # get batch images and labels
                # Only reserve attrs that is listed in attr_keys.
                real_atr, fake_atr = preprocess_attr(
                    self.attr_names, attr_list, attr_list_, self.attr_keys)
                if self.n_label>3:
                    fake_atr = np.zeros((self.batch_size_half, self.n_label))
                    r = np.random.randint(4, size=self.batch_size_half)
                    fake_atr[np.arange(self.batch_size_half), r] = 1
                    #fake_atr = np.array(fake_atr)
                    fake_atr[:,3:] = np.random.randint(
                        2, size=(self.batch_size_half,self.n_label-3))
                # Read images
                real_img = preprocess_image(
                    data_list, self.image_size, phase='train')
                unlab_img = preprocess_image(
                    unlab_data_list, self.image_size, phase='train')
                
                # update D network for 5 times
                feed = { 
                        self.real_img: real_img, 
                        self.unlab_img: unlab_img, 
                        self.real_atr: np.array(real_atr), 
                        self.fake_atr: np.array(fake_atr), 
                        self.lr_decay: lr_decay, 
                        self.pi_weight: pi_weight 
                       }
                _, d_loss, d_summary,unlab_cls_loss, gp_loss = self.sess.run(
                    [self.d_optim, self.d_loss, self.d_sum, self.unlab_pi_loss, 
                    self.gp_loss ], feed_dict = feed)

                # updatae G network for 1 time
                # adding identity loss for first several epoch, which stablize
                # the training.
                if (idx+1) % 5 == 0:
                    if epoch == 0:
                        lambda_id = 1.0
                    elif epoch == 1:
                        lambda_id = 0.5
                    else:
                        lambda_id = 0

                    feed = { 
                            self.real_img: real_img, 
                            self.unlab_img: unlab_img, 
                            self.real_atr: np.array(real_atr), 
                            self.fake_atr: np.array(fake_atr), 
                            self.lr_decay: lr_decay, 
                            self.pi_weight: pi_weight,
                            self.lambda_id:lambda_id 
                           }
                    if epoch >= 3: #self.epoch / 8:
                        _, g_loss, g_summary = self.sess.run(
                            [self.g_optim2, self.g_loss2, self.g_sum], 
                            feed_dict = feed)
                    else:
                        _, g_loss, g_summary = self.sess.run(
                            [self.g_optim, self.g_loss,  self.g_sum],
                            feed_dict = feed)
                                
                # summary
                    self.writer.add_summary(g_summary, count)
                self.writer.add_summary(d_summary, count)
                
                # save checkpoint and samples
                if count % self.snapshot == 0:
                    print("Epoch:%02d, Iter: %06d, g_loss: %4.4f, \
                        d_loss: %4.4f, unlab_loss: %4.4f, gp_loss: %4.4f" % 
                        (epoch, count, g_loss, d_loss,unlab_cls_loss,gp_loss))
                    
                    # checkpoint
                    self.checkpoint_save(count)
                    # save samples (from test dataset)
                    self.sample_save(count)

                
                if count % self.snapshot_test == 0:
                    print('Now Doing Testing...\n')
                    test_files = glob(os.path.join(self.data_dir, 'test', '*'))
                    test_batch_idxs = len(test_files) // self.batch_size
                    over_all_acc = 0
                    for idx_ in tqdm(range(test_batch_idxs)):
                        test_list  = (test_files[
                            idx_ * self.batch_size : (idx_+1) * self.batch_size])
                        attr_list = ([self.attr_list[os.path.basename(val)] 
                            for val in test_list])

                    
                        real_atr, _ = preprocess_attr(
                            self.attr_names, attr_list, attr_list, 
                            self.attr_keys)
                        real_img = preprocess_image(
                            test_list, self.image_size, phase='test')

                        feed = {
                                self.real_img: real_img, 
                                self.real_atr: real_atr 
                               }
                        batch_acc = self.sess.run(self.acc, feed_dict = feed)
                        over_all_acc += batch_acc
                    
                    over_all_acc = over_all_acc/test_batch_idxs
                    print('overall accuracy: %3.3f'%(over_all_acc))
                
        
        
    def test(self):
        # check if attribute available
        if not len(self.binary_attrs) == self.n_label:
            print ("binary_attr length is wrong! \
                The length should be {}".format(self.n_label))
            return
        
        # load or not checkpoint
        if self.phase=='test' and self.checkpoint_load():
            print(" [*] before training, Load SUCCESS ")
            
            self.attr_names, self.attr_list = attr_extract(self.data_dir)
            test_files = glob(os.path.join(self.data_dir, 'test', '*'))
            test_list = random.sample(test_files, 10)
            attr_list = ([self.attr_list[os.path.basename(val)] 
                for val in test_list])
            real_atr, _ = preprocess_attr(
                self.attr_names, attr_list, attr_list, self.attr_keys)
            fake_atr = [float(i) for i in list(self.binary_attrs)] * len(test_list)
            fake_atr = np.array(fake_atr)
            fake_atr = np.reshape(fake_atr,[-1,self.n_label])
            real_img = preprocess_image(test_list, self.image_size, phase='test')
            # generate fakeB
            feed = { 
                    self.real_img: real_img, 
                    self.real_atr: real_atr,
                    self.fake_atr: fake_atr 
                   }
            fake_img,recon_img = self.sess.run(
                [self.fake_img_sample,self.recon_img_sample], feed_dict = feed)
            
            # save samples
            test_file = os.path.join(self.test_dir, 'test.jpg')
            save_images(real_img, fake_img, recon_img, self.image_size, 
                test_file, num=10)

        else:
            print(" [!] before training, no need to Load ")

    def test_all(self):
        """
        Generate images base on all kinds of attributes combination
        Currently only CelebA 'hair color (n_label == 3)'
        and 'hair color + age + gender (n_label == 5)' are implemented.
        """
        if self.phase=='test_all' and self.checkpoint_load():
            print(" [*] before training, Load SUCCESS ")
            num_sample = 100
            test_files = glob(os.path.join(self.data_dir, 'test', '*'))
            self.attr_names, self.attr_list = attr_extract(self.data_dir)
            if self.n_label>3:
                test_list = []
                test_atr = []
                attr_list = ([self.attr_list[os.path.basename(val)] 
                    for val in test_files])
                real_atr = preprocess_attr_single(
                    self.attr_names, attr_list,self.attr_keys)
                for idx,value in enumerate(real_atr):
                    if sum(value[0:3])==1:
                        test_list.append(test_files[idx])
                        test_atr.append(value)
                print (len(test_list))
                test_list = test_list[:num_sample]
                test_atr = test_atr[:num_sample]

            else:
                test_list = test_files[:num_sample]
                attr_list = ([self.attr_list[os.path.basename(val)] 
                    for val in test_files])
                
            
            # get batch images and labels
            # Only reserve attrs that is listed in attr_keys.
            real_atr = preprocess_attr_single(
                self.attr_names, attr_list, self.attr_keys) 
            real_img = preprocess_image(
                test_list, self.image_size, phase='test') # Read images
            
            for idx,img in enumerate(real_img):
                # generate fakeB
                if self.c_method=='Sigmoid':
                    num_img = 9
                    fake_atr = test_atr[idx]
                    fake_atr = np.tile(fake_atr,(num_img,1))
                    fake_atr[:3,:3] = np.identity(3) # hair color
                    fake_atr[3,3] = 0 if fake_atr[3,3] else 1 #convert gender
                    fake_atr[4,4] = 0 if fake_atr[4,4] else 1 #aged
                    fake_atr[5,:3] = [1,0,0] #black hair
                    fake_atr[5,3] = 0 if fake_atr[5,3] else 1 #convert gender
                    fake_atr[6,:3] = [1,0,0] #black hair
                    fake_atr[6,4] = 0 if fake_atr[6,4] else 1 #aged hair
                    fake_atr[7,3] = 0 if fake_atr[7,3] else 1 #gender + aged
                    fake_atr[7,4] = 0 if fake_atr[7,4] else 1
                    fake_atr[8,:3] = [1,0,0] #black hair
                    fake_atr[8,3] = 0 if fake_atr[8,3] else 1 # gender
                    fake_atr[8,4] = 0 if fake_atr[8,4] else 1 # aged
                else:
                    fake_atr = np.identity(self.n_label)
                    num_img = self.n_label

                org_img = img.copy()
                img = np.reshape(img,
                    [1,self.image_size,self.image_size,self.image_channel])
                img = np.repeat(img,num_img,axis=0)
                feed = { 
                        self.real_img: img, 
                        self.real_atr: np.array(real_atr), 
                        self.fake_atr: np.array(fake_atr) 
                       }
                fake_img = self.sess.run(self.fake_img_sample, feed_dict = feed)
                fake_img = list(fake_img)
                # save samples
                file_name = os.path.basename(test_list[idx])
                test_file = os.path.join(self.test_dir, file_name)
                img_list = [org_img]
                img_list = img_list+fake_img
                save_images_test(img_list, self.image_size, test_file, 
                    num=1,col=num_img+1)

        else:
            print(" [!] before training, no need to Load ")
    
    def test_aux_accuracy(self):
        """
        Calculate the auxiliary classifier's classification accuracy.
        """
        if self.phase=='aux_test' and self.checkpoint_load():
            print(" [*] before training, Load SUCCESS ")
            
            self.attr_names, self.attr_list = attr_extract(self.data_dir)
            test_files = glob(os.path.join(self.data_dir, 'test', '*'))

            batch_idxs = len(test_files) // self.batch_size
            over_all_acc = 0
            for idx in tqdm(range(batch_idxs)):
                test_list = (
                    test_files[idx*self.batch_size : (idx+1)*self.batch_size])
                attr_list = ([self.attr_list[os.path.basename(val)] 
                    for val in test_list])
                
                real_atr, _ = preprocess_attr(
                    self.attr_names, attr_list, attr_list, self.attr_keys)
                
                real_img = preprocess_image(
                    test_list, self.image_size, phase='test')
                feed = { self.real_img: real_img, self.real_atr: real_atr }
                batch_acc = self.sess.run(self.acc, feed_dict = feed)
                over_all_acc += batch_acc
            print('overall accuracy: %3.3f'%(over_all_acc/batch_idxs))
        else:
            print(" [!] before training, no need to Load ")

    def compute_accuracy(self, x, y, method='Sigmoid'):
        if method == 'Sigmoid':
            x = tf.nn.sigmoid(x)
            predicted = self.threshold(x)
            correct = tf.cast(tf.equal(predicted, y),tf.float32)
            accuracy = tf.reduce_mean(correct) * 100.0
        else:
            x = tf.argmax(x,axis=1)
            y = tf.argmax(y,axis=1)
            correct = tf.cast(tf.equal(x, y),tf.float32)
            accuracy = tf.reduce_mean(correct) * 100.0
        return accuracy

    def threshold(self,x):
        ans = tf.cast(tf.greater(x,0.5),tf.float32)
        return ans

    def summary(self):
        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        
        # session : discriminator
        sum_d_1 = tf.summary.scalar('disc/adv_loss', self.d_adv_loss)
        sum_d_2 = tf.summary.scalar('D/real_cls_loss', self.real_cls_loss)
        sum_d_3 = tf.summary.scalar('D/d_loss', self.d_loss)
        sum_d_4 = tf.summary.scalar('D/unlab_pi_loss',self.unlab_pi_loss)
        sum_acc = tf.summary.scalar('D/acc',self.acc)
        sum_d_5 = tf.summary.scalar('D/gp_loss',self.gp_loss)
        sum_d_6 = tf.summary.scalar('D/real_loss',self.d_loss_real)
        sum_d_7 = tf.summary.scalar('D/fake_loss',self.d_loss_fake)
        self.d_sum = tf.summary.merge(
            [sum_d_1,sum_d_2, sum_d_3,sum_d_4,sum_acc,sum_d_5,sum_d_6,sum_d_7])
        
        # session : generator
        sum_g_1 = tf.summary.scalar('G/adv_loss', self.g_adv_loss)
        sum_g_2 = tf.summary.scalar('G/fake_cls_loss', self.fake_cls_loss)
        sum_g_3 = tf.summary.scalar('G/recon_loss', self.recon_loss)
        sum_g_4 = tf.summary.scalar('G/g_loss', self.g_loss)
        sum_g_5 = tf.summary.scalar('G/fake_unlab_cls_loss',
            self.fake_unlab_cls_loss)
        sum_g_6 = tf.summary.scalar('G/recon_unlab_loss',self.recon_loss_unlab)
        sum_g_7 = tf.summary.scalar('G/identity_loss',self.identity_loss)
        self.g_sum = tf.summary.merge(
            [sum_g_1,sum_g_2, sum_g_3, sum_g_4,sum_g_5,sum_g_6,sum_g_7])
       
    
    def checkpoint_load(self):
        print(" [*] Reading checkpoint...")
        print(self.ckpt_dir)
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            print ('found!',ckpt_name) #stargan.model-72000
            self.continue_iteration = int(ckpt_name.split('-')[-1])
            self.saver.restore(
                self.sess, os.path.join(self.ckpt_dir, ckpt_name))
            return True
        else:
            return False
        
        
    def checkpoint_save(self, step):
        model_name = "stargan.model"
        self.saver.save(self.sess,
                        os.path.join(self.ckpt_dir, model_name),
                        global_step=step)
        
        
    def sample_save(self, step):
        """
        Save samples periodically during training.
        """
        num_sample = self.n_label
        test_files = glob(os.path.join(self.data_dir, 'test', '*'))
        
        test_list = random.sample(test_files, num_sample)
        attr_list = [self.attr_list[os.path.basename(val)] for val in test_list]

        if self.c_method == 'Sigmoid':
            fake_atr = np.identity(num_sample)
            fake_atr[:,-1]=1
            fake_atr[-1,-1]=0
            fake_atr[3:,0]=1
        else:
            fake_atr = np.identity(num_sample)
        # get batch images and labels
        real_atr = preprocess_attr_single(
            self.attr_names, attr_list, self.attr_keys)
        real_img = preprocess_image(test_list, self.image_size, phase='test')
                        
        feed = { 
                    self.real_img: real_img, 
                    self.real_atr: np.array(real_atr), 
                    self.fake_atr: np.array(fake_atr) 
               }

        fake_img,recon_img = self.sess.run(
            [self.fake_img_sample,self.recon_img_sample], feed_dict = feed)
        
        # save samples
        sample_file = os.path.join(self.sample_dir, '%06d.jpg'%(step))
        save_images(real_img, recon_img, fake_img, 
            self.image_size, sample_file, num=num_sample)
    
    def rampup(self,iteration,rampup_length,max_val=2):
        if iteration < rampup_length:
            p = max(0.0, float(iteration)) / float(rampup_length)
            p = 1.0 - p
            return max_val*math.exp(-p*p*5.0)
        else:
            return max_val

    def rampdown(self,iteration,rampdown_length,num_iteration):
        if iteration >= (num_iteration - rampdown_length):
            p = ((iteration - (num_iteration - rampdown_length)) / 
                float(rampdown_length))
            p = 1.0 - p
            return 1-math.exp(-p*p*5.0)
        else:
            return 1.0