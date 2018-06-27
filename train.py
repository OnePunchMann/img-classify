import tensorflow as tf
from input_data import Parser, Generator
from model import VGG16
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2


if __name__ == '__main__':
    BATCH_SIZE = 10
    # 用于生成tfrecord文件
    # source = './data'
    # dest = './tmp/'
    # gen = Generator(source, dest)
    # gen.generate()

    source = './tmp/'
    # dest最开始是用作测试的打印图片目录
    dest = './tmp/dest/'
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 256, 256, 3])
    y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    par = Parser(source, dest)
    image, label = par.parse_batch(BATCH_SIZE, 1, 50, 5)
    vgg16 = VGG16()
    y = vgg16.VGG16_conv(x=x, keep_prob=0.5, trainable=True, name='vgg16')
    loss = vgg16.losses(y, y_)
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    with tf.Session() as sess:  # 开始一个会话
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # 启动多线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)

        for step in range(1000):
            # image_down = np.asarray(image_down.eval(), dtype='uint8')
            # plt.imshow(image.eval())
            # plt.show()
            images, labels = sess.run([image, label])
            # print(type(images))
            # print(labels)
            _, tra_loss = sess.run([train_step, loss], feed_dict={x: images, y_: labels})
            if step%10 == 0:
                print('Step %d, train loss = %f' % (step, tra_loss))
            # single, l = sess.run([image, label])  # 在会话中取出image和label
            # img = Image.fromarray(single, 'RGB')  # 这里Image是之前提到的
            # img.save(os.path.join(par.dest, str(i)) + '_''Label_' + str(l) + '.jpg')  # 存下图片
            # print(single,l)
        coord.request_stop()
        coord.join(threads)

