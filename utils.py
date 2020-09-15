# coding:utf-8
"""Functions for building network.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from subprocess import Popen, PIPE
import tensorflow as tf
import numpy as np
from six import iteritems
import random

def geterasebox(height,width):
    boxh = random.randrange(15, 35)
    boxw = random.randrange(15, 35)
    lh = random.randrange(20, height - 40)
    lw = random.randrange(20, width - 40)
    image_erase = np.ones((height, width, 1), np.float)
    image_erase[lh:lh + boxh, lw:lw + boxw, :] = 0
    image_erase = tf.convert_to_tensor(image_erase,tf.uint8)
    return image_erase

"""解析路径文件"""
def get_dataset(path,classnum,argumentfile):
    file = open(path, "r", encoding="utf-8")
    lines = file.readlines()
    picnumber = len(lines)
    dataset = [[[] for col in range(0)] for row in range(classnum)]
    for i in range(picnumber):
        path, label = lines[i].strip().split(" ")
        try:
            if os.path.exists(path):
                dataset[int(label)].append(path)
        except Exception as e:
            print("classname is not right->",e)
            raise

    # 删除无图片数据的索引位置
    i = 0
    while (i < len(dataset)):
        # print(i, len(dataset[i]))
        if len(dataset[i]) < 1:
            del dataset[i]
            i = i
        else:
            i += 1
    showtxt = "总图片数: "+str(picnumber)+" 预计类别数: "+str(classnum)+" 实际类别数: "+str(len(dataset))
    print(showtxt)
    with open(argumentfile,"a",encoding="utf-8") as f:
        f.write(showtxt)

    return dataset

def store_revision_info(src_path, output_dir, arg_string):
    try:
        # Get git hash
        cmd = ['git', 'rev-parse', 'HEAD']
        gitproc = Popen(cmd, stdout=PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_hash = stdout.strip()
    except OSError as e:
        git_hash = ' '.join(cmd) + ': ' + e.strerror

    try:
        # Get local changes
        cmd = ['git', 'diff', 'HEAD']
        gitproc = Popen(cmd, stdout=PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_diff = stdout.strip()
    except OSError as e:
        git_diff = ' '.join(cmd) + ': ' + e.strerror

    # Store a text file in the log directory
    rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
    with open(rev_info_filename, "w") as text_file:
        text_file.write('arguments: %s\n--------------------\n' % arg_string)
        text_file.write('tensorflow version: %s\n--------------------\n' % tf.__version__)  # @UndefinedVariable
        text_file.write('git hash: %s\n--------------------\n' % git_hash)
        text_file.write('%s' % git_diff)


def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in iteritems(vars(args)):
            f.write('%s: %s\n' % (key, str(value)))

