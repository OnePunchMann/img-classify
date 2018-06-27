# -*- coding = utf-8 -*-
import tensorflow as tf
import os
from PIL import Image




class Generator:

    def __init__(self, source, dest):
        print('Generator init')
        self.source = source
        self.dest = dest


    def generate(self):

        # 存放图片个数
        bestnum = 100000
        # 第几个图片
        num = 0
        # 第几个TFRecord文件
        recordfilenum = 0
        ftrecordfilename = ("traindata.tfrecords-%.3d" % recordfilenum)
        writer = tf.python_io.TFRecordWriter(os.path.join(self.dest, ftrecordfilename))
        for img_name in os.listdir(self.source):
            num = num + 1
            if num > bestnum:
                num = 1
                recordfilenum = recordfilenum + 1
                # tfrecords格式文件名
                ftrecordfilename = ("traindata.tfrecords-%.3d" % recordfilenum)
                writer = tf.python_io.TFRecordWriter(os.path.join(self.dest, ftrecordfilename))
            # print('路径',class_path)
            # print('第几个图片：',num)
            # print('文件的个数',recordfilenum)
            # print('图片名：',img_name)
            if img_name[0] == 'c':
                label = 0
            elif img_name[0] == 'd':
                label = 1
            img_path = os.path.join(self.source, img_name)  # 每一个图片的地址
            img = Image.open(img_path, 'r')
            size = img.size
            img_raw = img.tobytes()  # 将图片转化为二进制格式
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                    'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[size[0]])),
                    'img_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[size[1]]))
                }))
            writer.write(example.SerializeToString())  # 序列化为字符串
        writer.close()

class Parser:

    def __init__(self, source, dest):

        print("parser init")
        self.data_files = tf.gfile.Glob(os.path.join(source, "traindata.tfrecords-*"))
        print("data_files is :", self.data_files)
        self.dest = dest


    def parse(self):

        filename_queue = tf.train.string_input_producer(self.data_files, shuffle=True)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'img_raw': tf.FixedLenFeature([], tf.string),
                                               'img_width': tf.FixedLenFeature([], tf.int64),
                                               'img_height': tf.FixedLenFeature([], tf.int64),
                                           })  # 取出包含image和label的feature对象
        image = tf.decode_raw(features['img_raw'], tf.uint8)
        height = tf.cast(features['img_height'], tf.int32)
        width = tf.cast(features['img_width'], tf.int32)
        label = tf.cast(features['label'], tf.int32)
        channel = 3
        image = tf.reshape(image, [height, width, channel])
        return image, label

    def parse_batch(self, batch_size, num_threads=1, capacity=1000, min_after_dequeue=200):

        image, label = self.parse()
        image = tf.image.resize_images(image, [256, 256])
        images, labels = tf.train.shuffle_batch([image, label],
                                                batch_size=batch_size,
                                                num_threads=num_threads,
                                                capacity=capacity,
                                                min_after_dequeue=min_after_dequeue
                                                )
        return images, labels
