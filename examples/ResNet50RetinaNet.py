#!/usr/bin/env python
# coding: utf-8

# ## Load necessary modules
# import keras
import keras

import sys

sys.path.insert(0, '../')

import socket
import json

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

# use this to change which GPU to use
gpu = 0

# set the modified tf session as backend in keras
setup_gpu(gpu)

# ## Load RetinaNet model

# In[ ]:


# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('..', 'snapshots', 'mymodel.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
# model = models.convert_model(model)

# print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
                   7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
                   13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
                   21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
                   28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
                   34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
                   39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
                   46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',
                   53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                   60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
                   67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
                   73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
                   79: 'toothbrush', 80: 'n02701002'}

# ## Run detection on example

# In[ ]:


# Connect
host = '183.109.109.130'
port = 8080
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((host, port))

# file receive
def fileRcv(sock, i):
    data_transferred = 0
    filename = str(i) + '.jpg'

    sizeStr = sock.recv(1024)
    sizeStr = sizeStr.decode()
    print('size: ' + sizeStr)
    size = int(sizeStr)

    sock.sendall('rack'.encode('utf-8'))

    data = sock.recv(1024)
    with open('./images/' + filename, 'wb') as f:
        try:
            while data:
                data_transferred += len(data)

                f.write(data)
                if data_transferred >= size:
                    break
                elif data_transferred + 1024 >= size:
                    rcvTemp = size - data_transferred
                    data = sock.recv(rcvTemp)
                else:
                    data = sock.recv(1024)
        except Exception as e:
            print(e)

        print('파일 [%s] 전송종료. 전송량 [%d]' % (filename, data_transferred))


def getFileFromServer():
    ack = sock.recv(1024)
    ack = ack.decode()
    ret = False
    sndStr = 'reack'

    if ack == 'notsnd':
        print('결과 이미지 보내지 않음 \n')
    else:
        print('결과 이미지 보냄 \n')
        ret = True

    sock.sendall(sndStr.encode('utf-8'))
    for i in range(1, 5):
        print(i, ' 번째 파일')
        fileRcv(sock, i)

    return ret


def getFileSize(fileName, directory):
    fileSize = os.path.getsize(directory + "/" + fileName)
    return str(fileSize)


def imgSnd():
    filename = ['1.png', '2.png', '3.png', '4.png']
    print('파일 [%s] 전송 시작...' % filename)
 
    for i in range(4):
        sock.send(getFileSize(filename[i], './results_imgs').encode())
        reack = sock.recv(1024)
        if reack.decode('utf-8') != 'rack':
            raise Exception
        with open(os.path.join('./results_imgs' , filename[i]), 'rb') as f:
            try:
                data_transferred = 0
                data = f.read(1024)
                while data:
                    data_transferred += sock.send(data)
                    data = f.read(1024)

            except Exception as e:
                print(i)

        print('전송완료[%s], 전송량[%d]' % (filename[i], data_transferred))
    print('종료')


# Counting Data
countDic = [{'car': 0, 'ambulance': 0, 'person': 0}, {'car': 0, 'ambulance': 0, 'person': 0},
            {'car': 0, 'ambulance': 0, 'person': 0}, {'car': 0, 'ambulance': 0, 'person': 0}]
countDicIndex = 0



while True:
    for i in range(0, 4):
        countDic[i]['car'] = 0
        countDic[i]['ambulance'] = 0
        countDic[i]['person'] = 0
    countDicIndex = 0

    reSock = getFileFromServer()

    # load image
    for i in range(1, 5):
        name = './images/' + str(i) + '.jpg'
        image = read_image_bgr(name)

        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)

        # process image
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)

        # correct for image scale
        boxes /= scale

        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < 0.4:
                break

            color = label_color(label)

            b = box.astype(int)
            draw_box(draw, b, color=color)

            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_caption(draw, b, caption)

            if labels_to_names[label] == 'person':
                countDic[countDicIndex]['person'] = countDic[countDicIndex].get('person') + 1
            elif labels_to_names[label] == 'car':
                countDic[countDicIndex]['car'] = countDic[countDicIndex].get('car') + 1
            elif labels_to_names[label] == 'n02701002':
                countDic[countDicIndex]['ambulance'] = countDic[countDicIndex].get('ambulance') + 1

        countDicIndex = countDicIndex + 1

        draw = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
        draw = cv2.resize(draw, (480, 270))
        cv2.imwrite('./results_imgs/' + str(i) + '.png', draw)

    print(countDic)
    if reSock :
      imgSnd()

    s = json.dumps(countDic)
    try:
        s = s.encode('utf-8')
        sock.sendall(str(len(s)).encode('utf-8'))
        
        ack = sock.recv(1024)

        ack = ack.decode()
        if ack == 'ack' : 
            sock.sendall(s)

    except socket.timeout as e:
        print(e)

# plt.figure(figsize=(15, 15))
# plt.axis('off')
# plt.imshow(draw)
# plt.show()


# In[ ]:

