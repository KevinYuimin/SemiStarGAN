import tensorflow as tf
from ops import conv2d, deconv2d, relu, lrelu, instance_norm, tanh, entropy, batch_norm
import numpy as np

def generator(images, options, reuse=False, name='gen'):
    # reuse or not
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "CONSTANT") #CONSTANT
            y = instance_norm(conv2d(y, dim, ks, s, 
                padding='VALID', name=name+'_c1'), name+'_in1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "CONSTANT")
            y = instance_norm(conv2d(y, dim, ks, s, 
                padding='VALID', name=name+'_c2'), name+'_in2')
            return y + x
            
        # down sampling
        c0 = tf.pad(images, [[0, 0], [3, 3], [3, 3], [0, 0]], "CONSTANT")
        c1 = relu(instance_norm(conv2d(c0,   options.nf, ks=7, s=1, 
                padding='VALID', name='gen_ds_conv1'), 'in1_1'))
        c2 = relu(instance_norm(conv2d(c1, 2*options.nf, ks=4, s=2, 
                name='gen_ds_conv2'), 'in1_2'))
        c3 = relu(instance_norm(conv2d(c2, 4*options.nf, ks=4, s=2, 
                name='gen_ds_conv3'), 'in1_3'))
        
        # bottleneck
        r1 = residule_block(c3, options.nf*4, name='g_r1')
        r2 = residule_block(r1, options.nf*4, name='g_r2')
        r3 = residule_block(r2, options.nf*4, name='g_r3')
        r4 = residule_block(r3, options.nf*4, name='g_r4')
        r5 = residule_block(r4, options.nf*4, name='g_r5')
        r6 = residule_block(r5, options.nf*4, name='g_r6')

        # up sampling
        d1 = relu(instance_norm(deconv2d(r6, options.nf*2, 4, 2, 
                name='g_us_dconv1'), 'g_d1_in'))
        d2 = relu(instance_norm(deconv2d(d1, options.nf  , 4, 2, 
                name='g_us_dconv2'), 'g_d2_in'))
        d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "CONSTANT")
        pred = tf.nn.tanh(conv2d(d2, 3, 7, 1, padding='VALID', name='g_pred_c'))
        
        return pred

def semi_c_discriminator_Y(
    images, options,reuse=False, repeat_num=6, stddev=0.15, 
    training = True,d_rate=0.5,name='cls'):
    # currently best model for hair colors
    if training:
        gaussian_noise = tf.random_normal(
            shape=tf.shape(images),mean=0.0, stddev=stddev)
        images = images + gaussian_noise

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
            
        # input & hidden layer
        h1   = lrelu(conv2d(images, options.nf, ks=4, s=2, name='hid_conv1')) #32
        h2   = lrelu(conv2d(h1,   2*options.nf, ks=4, s=2, name='hid_conv2')) #16
        h3_d = lrelu(conv2d(h2,   4*options.nf, ks=4, s=2, name='dis_conv3')) #08
        h4_d = lrelu(conv2d(h3_d, 8*options.nf, ks=4, s=2, name='dis_conv4')) #04
        
        h2_c = tf.layers.dropout(h2,rate=d_rate,training=training, name='cls_drop1')
        h3_c = lrelu(batch_norm(conv2d(h2_c,   4*options.nf, ks=4, s=2, 
                name='cls_conv3'),is_training=training,name='cbn1')) #8
        h4_c = lrelu(batch_norm(conv2d(h3_c,   4*options.nf, ks=4, s=2, 
                name='cls_conv4'),is_training=training,name='cbn2')) #4
        h4_c = tf.layers.dropout(h4_c,rate=d_rate,training=training, 
                name='cls_drop2')
        h5_c = lrelu(batch_norm(conv2d(h4_c,   8*options.nf, ks=4, s=2, 
                name='cls_conv5'),is_training=training,name='cbn3')) #2
        h6_c = lrelu(batch_norm(conv2d(h5_c,   8*options.nf, ks=4, s=2, 
                name='cls_conv6'),is_training=training,name='cbn4')) #1
        h6_c = tf.layers.dropout(h6_c,rate=d_rate,training=training, 
                name='cls_drop3')
        h7_c = lrelu(batch_norm(conv2d(h6_c,   4*options.nf, ks=1, s=1, 
                name='cls_conv7'),is_training=training,name='cbn5'))
        h8_c = lrelu(batch_norm(conv2d(h7_c,   2*options.nf, ks=1, s=1, 
                name='cls_conv8'),is_training=training,name='cbn6'))
        
        src = conv2d(h4_d, 1, ks=3, s=1, name='disc_conv7_patch')
        aux = tf.reduce_mean(h8_c, [1, 2], keep_dims=True, name='GlobalPool')
        aux = conv2d(aux, options.n_label, ks=1, s=1, name='cls_conv9_aux')
        aux = tf.reshape(aux,[-1,options.n_label])

        return src,aux

def semi_c_discriminator_Y_deep(
    images, options,reuse=False, repeat_num=6, stddev=0.15, 
    training = True,name='cls'):

    if training:
        gaussian_noise = tf.random_normal(
            shape=tf.shape(images),mean=0.0, stddev=stddev)
        images = images + gaussian_noise

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
            
        # input & hidden layer
        h1   = lrelu(conv2d(images, options.nf, ks=4, s=2, name='hid_conv1')) #64
        h2   = lrelu(conv2d(h1,   2*options.nf, ks=4, s=2, name='hid_conv2')) #32
        
        h2_c = tf.layers.dropout(h2  ,rate=0.5,training=training, name='cls_drop1')
        
        h3_d = lrelu(conv2d(h2,     4*options.nf, ks=4, s=2, name='dis_conv3')) #16
        h4_d = lrelu(conv2d(h3_d,   4*options.nf, ks=4, s=2, name='dis_conv4')) #08
        h5_d = lrelu(conv2d(h4_d,   8*options.nf, ks=4, s=2, name='dis_conv5')) #04
        h6_d = lrelu(conv2d(h5_d,   8*options.nf, ks=4, s=2, name='dis_conv6')) #04
       
        h3_c = lrelu(batch_norm(conv2d(h2_c,   4*options.nf, ks=4, s=2, 
                name='cls_conv3'),is_training=training,name='cbn1'))
        h4_c = lrelu(batch_norm(conv2d(h3_c,   4*options.nf, ks=4, s=2, 
                name='cls_conv4'),is_training=training,name='cbn2'))
        h4_c = tf.layers.dropout(h4_c,rate=0.5,training=training, 
                name='cls_drop2')
        h5_c = lrelu(batch_norm(conv2d(h4_c,   8*options.nf, ks=4, s=2, 
                name='cls_conv5'),is_training=training,name='cbn3'))
        h6_c = lrelu(batch_norm(conv2d(h5_c,   8*options.nf, ks=4, s=2, 
                name='cls_conv6'),is_training=training,name='cbn4'))
        h6_c = tf.layers.dropout(h6_c,rate=0.5,training=training, name='cls_drop3')
        h7_c = lrelu(batch_norm(conv2d(h6_c,   4*options.nf, ks=1, s=1, 
                name='cls_conv7'),is_training=training,name='cbn5'))
        h8_c = lrelu(batch_norm(conv2d(h7_c,   2*options.nf, ks=1, s=1, 
                name='cls_conv8'),is_training=training,name='cbn6'))
       
        src = conv2d(h6_d, 1, ks=3, s=1, name='disc_conv7_patch')
        aux = tf.reduce_mean(h8_c, [1, 2], keep_dims=True, name='GlobalPool')
        aux = conv2d(aux, options.n_label, ks=1, s=1, name='cls_conv9_aux')
        aux = tf.reshape(aux,[-1,options.n_label])
        
        return src,aux
        


def wgan_gp_loss(real_img, fake_img, options,discriminator_=None): #gradient penalty
    alpha = tf.random_uniform(
        shape=[options.batch_size,1,1,1], 
        minval=0.,
        maxval=1.
    )


    hat_img = alpha * real_img + (1.-alpha) * fake_img
    if discriminator_:
        gradients = tf.gradients(discriminator_(
            hat_img, options, reuse=True, name='disc')[0], xs=[hat_img])[0]
    else:
        gradients = tf.gradients(semi_c_discriminator_Y(
            hat_img, options, reuse=True, name='disc')[0], xs=[hat_img])[0]
    
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
    gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.))
    
    return options.lambda_gp * gradient_penalty
        

def cls_loss(logits, labels):
    # sigmoid cross entropy return [batchsize,n_label]
    return tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits,labels=labels),axis=1))

def cls_loss_SoftCE(logits, labels):
    # softmax cross entropy return [batchsize]
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits,labels=labels))

def recon_loss(image1, image2):
    return tf.reduce_mean(tf.abs(image1 - image2))

def recon_loss_with_mask(image1,image2,mask):
    diff = tf.reduce_mean(tf.abs(image1 - image2),axis=[1,2,3])
    diff = diff*mask
    loss = tf.reduce_sum(diff)/tf.reduce_sum(mask)
    return loss

def l2_loss(z1,z2):
    return tf.reduce_mean(tf.square(z1-z2))