# coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import time
import sys
import argparse
import utils
from tensorflow.python.ops import data_flow_ops
import random
import tensorflow as tf
import numpy as np
import DentNet_ATT
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

boundaries = [30000, 60000, 100000]
learning_rates = [0.001, 0.001, 0.0005, 0.0001]
inputsize = 128
classnum = 9490
featuresize = 512
weightdecay =5e-4

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs_base_dir', type=str, help='log',
                        default=r'.\savefile')
    parser.add_argument('--pretrained_model', type=str, help='pretrained model')
    parser.add_argument('--param_num', type=str,default=True, help='# of params')
    parser.add_argument('--data_path', type=str, help='labeltxt.',
                        default=r".\train.txt")
    parser.add_argument('--max_nrof_epochs', type=int, help='总代数.', default=100)
    parser.add_argument('--image_size_h', type=int, help='尺寸.', default=128)
    parser.add_argument('--image_size_w', type=int, help='尺寸.', default=128)
    parser.add_argument('--batch_size', type=int, help='batch图片数.', default=16)
    parser.add_argument('--people_per_batch', type=int, help='每批类数.', default=16)
    parser.add_argument('--images_per_person', type=int, help='每类张数.', default=1)
    parser.add_argument('--epoch_size', type=int, help='一代的批次总数.', default=1500)
    parser.add_argument('--embedding_size', type=int, help='特征维度.', default=512)
    parser.add_argument('--transform', help='预处理增强', action='store_true',default=True)
    parser.add_argument('--opt', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='优化算法 ', default='MOM')
    parser.add_argument('--seed', type=int, help='Random seed.', default=666)
    return parser.parse_args(argv)
    
def main(args):

    """basic info"""
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), "logs")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.logs_base_dir), "models")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    argumentfile = os.path.join(log_dir, 'arguments.txt')
    utils.write_arguments_to_file(args, argumentfile)
    src_path, _ = os.path.split(os.path.realpath(__file__))
    # utils.store_revision_info(src_path, log_dir, ' '.join(sys.argv))
    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    if args.pretrained_model:
        print('Pre-trained model: %s' % os.path.expanduser(args.pretrained_model))

    """trainset"""
    # np.random.seed(seed=args.seed)
    train_set = utils.get_dataset(args.data_path,classnum,argumentfile)

    """开启图"""
    with tf.Graph().as_default():
        # tf.set_random_seed(args.seed)

        """占位符"""
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int64, shape=(None,1), name='labels')

        """通过队列读取批量数据"""
        input_queue = data_flow_ops.FIFOQueue(capacity=1000, dtypes=[tf.string, tf.int64], shapes=[(1,), (1,)],
                                              shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])

        nrof_preprocess_threads = 17
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()  # 路径、标签出队
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=1)  # uint8 with shape [h, w, num_channels]

                image = tf.image.crop_to_bounding_box(image, 0, 0, 850, 2100)
                image = tf.image.resize_images(image, (args.image_size_h, args.image_size_w), method=1)

                if args.transform:
                    if random.random() < 1:
                        height, width, d = image.shape
                        mask = utils.geterasebox(height, width)
                        image = tf.multiply(image,mask)

                image = tf.cast(image, tf.float32)
                images.append(image)
            images_and_labels.append([images, label])

        image_batch, labels_batch = tf.train.batch_join(images_and_labels, batch_size=batch_size_placeholder,
                                                        shapes=[(args.image_size_h, args.image_size_w, 1), ()],
                                                        enqueue_many=True,
                                                        capacity=4 * nrof_preprocess_threads * args.batch_size,
                                                        allow_smaller_final_batch=False)
        tf.summary.image("trainimage",image_batch,max_outputs=16)

        # 阶梯下降学习率
        with tf.name_scope("lr"):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            learning_rate = tf.train.piecewise_constant(global_step, boundaries=boundaries, values=learning_rates)
        """运行网络 输出特征向量"""
        model = DentNet_ATT.DentNet_ATT(image_batch,classnum,True)
        embeddings = model.dropout2
        Cosin_logits = cosineface_losses(embedding=embeddings,labels=labels_batch,out_num=classnum)
        # loss
        with tf.name_scope("loss"):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels_batch, logits=Cosin_logits, name='cross_entropy_per_example')
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
            L2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()]) * 0.001 #0.001需要对比设置
            total_loss = tf.add_n([cross_entropy_mean+ L2_loss*weightdecay], name='total_loss')

        # optimizer
        with tf.name_scope('optimizer'):
            if args.opt == 'ADAGRAD':
                optimizer = tf.train.AdagradOptimizer(learning_rate)
            elif args.opt == 'ADADELTA':
                optimizer = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
            elif args.opt == 'ADAM':
                optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
            elif args.opt == 'RMSPROP':
                optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
            elif args.opt == 'MOM':
                optimizer = tf.train.MomentumOptimizer(learning_rate, 0.95, use_nesterov=True)
            else:
                raise ValueError('Invalid optimization algorithm')
            train_op = optimizer.minimize(total_loss, global_step=global_step)

        # accuracy 计算准确度
        with tf.name_scope("total_accuracy"):
            prob = tf.nn.softmax(Cosin_logits)
            one_hot_label = tf.one_hot(labels_batch, classnum)
            correct_pred = tf.equal(tf.argmax(prob, 1), tf.argmax(one_hot_label, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Tensorboard
        tf.summary.scalar('loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('lr', learning_rate)

        saver = tf.train.Saver(max_to_keep=200)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(log_dir)

        """开启session"""
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder: True})
            sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder: True})
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=sess)

            if args.pretrained_model:
                print('Restoring pretrained model: %s' % args.pretrained_model)
                saver.restore(sess, os.path.expanduser(args.pretrained_model))
            if args.param_num:
                count_params()
            summary_writer.add_graph(sess.graph)  # 把图的数据写入

            # Training
            epoch = 0
            while epoch < args.max_nrof_epochs:
                batch_number = 0  # 一代内的步数
                while batch_number < args.epoch_size:

                    """选择训练样本"""
                    thestep = sess.run(global_step)
                    image_paths, num_per_class, batch_truelabel = sample_people(train_set, args.people_per_batch,
                                                                                args.images_per_person,log_dir,thestep)

                    nrof_examples = args.people_per_batch * args.images_per_person
                    image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 1))
                    sess.run(enqueue_op,{image_paths_placeholder: image_paths_array, labels_placeholder: batch_truelabel})
                    nrof_batches = int(np.ceil(nrof_examples / args.batch_size))

                    """计算特征向量"""
                    for i in range(nrof_batches):
                        batch_size = min(nrof_examples - i * args.batch_size, args.batch_size)
                        emb, lab, err, _, ministep, acc, s, lrr = sess.run(
                            [embeddings, labels_batch, total_loss, train_op,
                             global_step, accuracy, summary_op, learning_rate],
                            feed_dict={batch_size_placeholder: batch_size,
                                       phase_train_placeholder: True})
                        if (ministep + 1) % 100 == 0:
                            timenow = str(time.strftime('%Y-%m-%d %H:%M:%S'))
                            print(timenow,"step:",ministep+1,"lr:",lrr,"Loss:",err,"TrainAcc:",acc)
                            summary_writer.add_summary(s, ministep)

                        # save model
                        checkpoint_name = os.path.join(model_dir, 'models-step' + str(ministep+1) + '.ckpt')
                        if (ministep + 1) % 1000 == 0:
                            print("Saving checkpoint of model:", checkpoint_name)
                            saver.save(sess, checkpoint_name)
                    batch_number += 1
                epoch += 1
    return

"""选择样本"""
def sample_people(dataset, people_per_batch, images_per_person,log_dir,thestep):
    nrof_images = people_per_batch * images_per_person
    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)

    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    truelabel = np.zeros((nrof_images,1))
    # Sample images from these classes until we have enough
    while len(image_paths) < nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images - len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index][j] for j in idx]
        sampled_class_indices += [class_index] * nrof_images_from_class
        truelabel[i] = class_index
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i += 1

    labeltxt = os.path.join(log_dir,"label.txt")
    with open(labeltxt,"a",encoding="utf-8") as labelfile:
        labelfile.write(str(thestep)+":"+str(sampled_class_indices)+"\n")
    return image_paths, num_per_class, truelabel


def cosineface_losses(embedding, labels, out_num, w_init=None, s=30., m=0.35):
    with tf.variable_scope('cosineface_loss'):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos_theta - m
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t_m = tf.subtract(cos_t, m, name='cos_t_m')

        mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        output = tf.add(s * tf.multiply(cos_t, inv_mask), s * tf.multiply(cos_t_m, mask), name='cosineface_loss_output')
    return output
    
def count_params():
    total_params=0
    for variable in tf.trainable_variables():
        shape=variable.get_shape()
        params=1
        for dim in shape:
            params=params*dim.value
        total_params+=params
    print("Total training params: %.2fM" % (total_params / 1e6))


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
