#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def display(Xrow, ImageName):
    '''Display a digit by first reshaping it from the row-vector into the image.  '''
    plt.imshow(np.reshape(Xrow, (28, 28)))
    plt.gray()
    plt.savefig(str(ImageName) + ".png")
    plt.show()


def loadData(digit, num):
    '''
    Loads all of the images into a data-array (for digits 0 through 5).

    The training data has 5000 images per digit and the testing data has 200,
    but loading that many images from the disk may take a while.  So, you can
    just use a subset of them, say 200 for training (otherwise it will take a
    long time to complete.

    Note that each image as a 28x28 grayscale image, loaded as an array and
    then reshaped into a single row-vector.

    Use the function display(row-vector) to visualize an image.

    '''
    X = np.zeros((num, 784), dtype=np.uint8)  # 784=28*28
    for i in range(num):

        pth = 'D://bigdata2/train%d/%05d.pgm' % (digit,i)
        with open(pth, 'rb') as infile:
            #header = infile.readline()
            #header2 = infile.readline()
            #header3 = infile.readline()
            image = np.fromfile(infile, dtype=np.uint8).reshape(1, 784)

        X[i, :] = image

    print('\n')
    return X


def loadAllData(num):

    X = np.zeros((num, 784), dtype=np.uint8)  # 784=28*28
    for k in range(3):
        for i in range(int(num / 3)):
            pth = '/Users/meredith/assignment/train%d/%05d.pgm' % (k, i)
            with open(pth, 'rb') as infile:
                image = np.fromfile(infile, dtype=np.uint8).reshape(1, 784)
            X[i * (k + 1), :] = image

    print('\n')
    return X

# for question2
num = 5000
matrix = loadData(0, num)
'''
# for question3
matrix = loadData(2, num)
# for question4
num = 15000
matrix = loadAllData(num)
'''
features = 20
mean = matrix.mean(axis=0)
matrix = matrix - mean
display(mean, 'meanImage')
cov = np.dot(matrix, matrix.T) / num
evs, vec = np.linalg.eigh(cov)
index = np.argsort(evs)
index = index[::-1]

# show eigenvalues
x = range(1, 101)
y = [0] * 100
for i in range(100):
    y[i] = evs[index[i]]
plt.plot(x, y, 'bo')
plt.show()

# show the first 20 eigenvectors(as image)
eigen = [0] * features
for i in range(features):
    eigen[i] = vec[index[i]].tolist()
data = np.dot(np.transpose(matrix), np.transpose(eigen))
for j in range(features):
    display(np.transpose(data)[j], j + 1)