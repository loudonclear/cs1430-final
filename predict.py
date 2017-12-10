import tensorflow as tf
import numpy as np
import os
import csv
import cv2

c1out = 12
fc1out = 256

HEIGHT = 480
WIDTH = 640
OUTDIM = 2
BATCH = 20
ITERS = 100




dataFile = "/gazePredictions.csv"
test_index = "test_1430_1.txt"
train_index = "train_1430_1.txt"

index = []
with open(train_index) as ind:
    for line in ind:
        dirToView = line.rstrip('\n')
        try:
            with open(dirToView + dataFile) as f:
                readCSV = csv.reader(f, delimiter=',')
                for row in readCSV:
                    index.append(row)
        except:
            pass

tot = sum(1 for line in index)
start = 0


def next_batch(size):
    global start
    if start + size >= tot:
        start = start + size - len(imgs_arr)

    img_batch = []
    coor_batch = []
    batchIndex = index[start:start+size]

    for row in batchIndex:
        frameFilename = row[0]
        # frameTimestamp = row[1]

        # Tobii has been calibrated such that 0,0 is top left and 1,1 is bottom right on the display.
        tobiiLeftEyeGazeX = float(row[2])
        tobiiLeftEyeGazeY = float(row[3])
        tobiiRightEyeGazeX = float(row[4])
        tobiiRightEyeGazeY = float(row[5])

        tobiiEyeGazeX = (tobiiLeftEyeGazeX + tobiiRightEyeGazeX) / 2
        tobiiEyeGazeY = (tobiiLeftEyeGazeY + tobiiRightEyeGazeY) / 2


        img = cv2.imread(frameFilename)
        img_batch += [img]

        coor_batch += [[tobiiEyeGazeX, tobiiEyeGazeY]]

    start += size
    return img_batch, coor_batch


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


imgs = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, 3])
coords = tf.placeholder(tf.float32, shape=[None, 2])

wc1 = weight_variable([5, 5, 3, c1out])
bc1 = bias_variable([c1out])
h_conv1 = tf.nn.relu(conv2d(imgs, wc1) + bc1)
h_pool1 = max_pool_2x2(h_conv1)

wc2 = weight_variable([3, 3, c1out, c1out*2])
bc2 = bias_variable([c1out*2])
h_conv2 = tf.nn.relu(conv2d(h_pool1, wc2) + bc2)
h_pool2 = max_pool_2x2(h_conv2)

wfc1 = weight_variable([int(HEIGHT/4*WIDTH/4*c1out*2), fc1out])
bfc1 = bias_variable([fc1out])
h_pool2_flat = tf.reshape(h_pool2, [-1, int(HEIGHT/4*WIDTH/4*c1out*2)])
fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, wfc1) + bfc1)

wfc2 = weight_variable([fc1out, OUTDIM])
bfc2 = bias_variable([OUTDIM])
out_coords = tf.matmul(fc1, wfc2) + bfc2

loss = tf.reduce_sum(tf.square(out_coords - coords))
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

squared_dists = []
for i in range(ITERS):
    img_batch, coor_batch = next_batch(BATCH)
    if i % 2 == 0:
        l = sess.run(loss, feed_dict={imgs: img_batch, coords: coor_batch})
        print(l)
        squared_dists.append(l)
    sess.run(train_step, feed_dict={imgs: img_batch, coords: coor_batch})

plt.plot(range(1, ITERS + 1), squared_dists)
plt.xlabel('iteration')
plt.ylabel('squared distance')
plt.show()
