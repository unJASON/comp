import keras
import keras.layers
import keras.utils
from keras import backend as K
import PIL.Image as Image
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
import cv2
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Lambda
from keras.models import Model
# from keras.utils import plot_model
import cv2
import numpy as np
import tensorflow as tf

boxSize = 32


def createYOLOModel():
    inPuts = Input((416, 416, 3))
    inPuts2 = Input((13, 13, 5))
    dic = {
        # 'filters':64,
        # 'kernel_size':(7,7),
        'padding': 'same',
    }
    x = Conv2D(64, (3, 3), **dic)(inPuts)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(64, (3, 3), **dic)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (1, 1), **dic)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(128, (3, 3), **dic)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(128, (1, 1), **dic)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(128, (3, 3), **dic)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (1, 1), **dic)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(128, (3, 3), **dic)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(256, (3, 3), **dic)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(256, (1, 1), **dic)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(256, (3, 3), **dic)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(256, (1, 1), **dic)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(512, (3, 3), **dic)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(512, (1, 1), **dic)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(512, (3, 3), **dic)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(512, (1, 1), **dic)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(1024, (3, 3), **dic)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(1024, (1, 1), **dic)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(1024, (3, 3), **dic)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(1024, (1, 1), **dic)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(1024, (1, 1), **dic)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(1024, (3, 3), **dic)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(1024, (1, 1), **dic)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(1024, (1, 1), **dic)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(5, (1, 1), **dic, name='predict_layer')(x)

    x = Lambda(loss, output_shape=(1,), name='yolo_loss')([x, inPuts2])
    model = Model(inputs=[inPuts] + [inPuts2], outputs=x)

    # plot_model(model, to_file='model.png', show_shapes=True)
    return model


# 数据预处理
def mode(imgObjMap, imgPath):
    flag = True

    for imgObjK in imgObjMap.keys():
        imgObjK = imgObjK[:-7]
        img = cv2.imread(imgPath + imgObjK + '.bmp')  # img (h,w,channel)
        # print(img.shape)
        resImg = cv2.resize(img, (416, 416), interpolation=cv2.INTER_CUBIC)
        resAns = np.zeros((13, 13, 5))
        for obj in imgObjMap[imgObjK + '.gt.xml']:
            posx = int(obj['x']) * 416 // img.shape[1]  # 左上坐标x
            posy = int(obj['y']) * 416 // img.shape[0]  # 左上角坐标y
            posxw = (int(obj['x']) + int(obj['width'])) * 416 // img.shape[1]  # 右下角坐标x
            posyh = (int(obj['y']) + int(obj['height'])) * 416 // img.shape[0]  # 右下角坐标y
            posCenter = (int((posx + posxw) // 2), int((posy + posyh) // 2))  # center position
            resAns[posCenter[0] // boxSize, posCenter[1] // boxSize, 0] = 1  # pc
            resAns[posCenter[0] // boxSize, posCenter[1] // boxSize, 1] = posCenter[0] % boxSize / boxSize  # x
            resAns[posCenter[0] // boxSize, posCenter[1] // boxSize, 2] = posCenter[1] % boxSize / boxSize  # y
            resAns[posCenter[0] // boxSize, posCenter[1] // boxSize, 3] = float(obj['width']) * 416 / img.shape[
                1] / boxSize  # boxw相对Box大小的width
            resAns[posCenter[0] // boxSize, posCenter[1] // boxSize, 4] = float(obj['height']) * 416 / img.shape[
                0] / boxSize  # boxy相对Box大小的with

            # resAns[posCenter[0] // boxSize, posCenter[1] // boxSize, 3] = float(obj['width']) / img.shape[1]  # boxw相对Box大小的width
            # resAns[posCenter[0] // boxSize, posCenter[1] // boxSize, 4] = float(obj['height']) / img.shape[0]  # boxh相对Box大小的height
        if flag:
            resImags = np.asarray(resImg).reshape(1, 416, 416, 3)
            resAnswers = np.asarray(resAns).reshape(1, 13, 13, 5)
            flag = False
        else:
            resImags = np.append(resImags, [np.asarray(resImg)], axis=0)
            resAnswers = np.append(resAnswers, [np.asarray(resAns)], axis=0)
    return resImags, resAnswers


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def showPredictExample(testImages, true_data, data):
    for l in range(testImages.shape[0]):
        img = cv2.cvtColor(np.asarray(Image.fromarray(np.uint8(testImages[l]))), cv2.COLOR_RGB2BGR)
        for j in range(data[l].shape[0]):
            for k in range(data[l].shape[1]):
                # if sigmoid(data[l][j][k][0]) >0.6:
                # if sigmoid(data[l][j][k][0])/np.max( sigmoid(data[l][:][:][0]) ) > 0.2:
                if true_data[l][j][k][0] == 1:
                    # 画点

                    cv2.rectangle(img, (
                        int((j + sigmoid(data[l][j][k][1])) * boxSize), int((k + sigmoid(data[l][j][k][2])) * boxSize)),
                                  (int((j + sigmoid(data[l][j][k][1])) * boxSize + 1),
                                   int(((k + sigmoid(data[l][j][k][2]))) * boxSize + 1)),
                                  color=(255, 255, 255), thickness=2)
                    # 画框
                    cv2.rectangle(img,
                                  (int((j + sigmoid(data[l][j][k][1])) * boxSize - 0.5 * np.exp(
                                      data[l][j][k][3]) * boxSize),
                                   int((k + sigmoid(data[l][j][k][2])) * boxSize - 0.5 * np.exp(
                                       data[l][j][k][4]) * boxSize))
                                  , (int(
                            (j + sigmoid(data[l][j][k][1])) * boxSize + 0.5 * np.exp(data[l][j][k][3]) * boxSize),
                                     int((k + sigmoid(data[l][j][k][2])) * boxSize + 0.5 * np.exp(
                                         data[l][j][k][4]) * boxSize))
                                  , color=(255, 255, 255), thickness=2)
                # if true_data[l][j][k][0] ==1:
                #     #  画点
                #     cv2.rectangle(img, (
                #     int((j + true_data[l][j][k][1]) * boxSize), int((k + true_data[l][j][k][2]) * boxSize)),
                #                   (int((j + true_data[l][j][k][1]) * boxSize + 1),
                #                    int(((k + true_data[l][j][k][2])) * boxSize + 1)),
                #                   color=(0, 0, 0), thickness=2)
                #     # 画框
                #     cv2.rectangle(img,
                #                   (int((j + true_data[l][j][k][1]) * boxSize - 0.5 * (true_data[l][j][k][3]) * boxSize),
                #                    int((k + true_data[l][j][k][2]) * boxSize - 0.5 * (true_data[l][j][k][4]) * boxSize))
                #                   ,
                #                   (int((j + true_data[l][j][k][1]) * boxSize + 0.5 * (true_data[l][j][k][3]) * boxSize),
                #                    int((k + true_data[l][j][k][2]) * boxSize + 0.5 * (true_data[l][j][k][4]) * boxSize))
                #                   , color=(0, 0, 0), thickness=2)
        print("predict:", l)
        draw_confidence(data=data[l], pre=True)
        print("true:", l)
        draw_confidence(data=true_data[l], pre=False)
        # draw_line(img)
        cv2.imshow('fig1', img)
        cv2.waitKey(0)


def confidence():
    pass


def draw_confidence(data, pre=False):
    for j in range(data.shape[0]):
        for k in range(data.shape[1]):
            # 输出confidence
            if (k != 12):
                if pre:
                    print('%.2f' % sigmoid(data[k][j][0]), end=' ')
                else:
                    print('%.2f' % data[k][j][0], end=' ')
            else:
                if pre:
                    print('%.2f' % sigmoid(data[k][j][0]))
                else:
                    print('%.2f' % data[k][j][0])


def draw_line(img):
    for i in range(13):
        cv2.line(img, (boxSize * i, 0), (boxSize * i, boxSize * 13), color=(0, 0, 0), thickness=2)
        cv2.line(img, (0, boxSize * i), (boxSize * 13, boxSize * i), color=(0, 0, 0), thickness=2)


def loss_v2(y_true, y_pre):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # 转成tensor
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pre = tf.convert_to_tensor(y_pre, dtype=tf.float32)
    loss_final = loss([y_pre, y_true]).eval(session=sess)
    sess.close()
    return loss_final
    # y_true_pc = y_true[..., 0:1]
    # y_pre_pc = y_pre[..., 0:1]
    # y_pre_xy = y_pre[..., 1:3]
    # y_pre_wh = y_pre[..., 3:5]
    # y_true_xy = y_true[..., 1:3]
    # y_true_wh = y_true[..., 3:5]
    # xy_loss = 5 * K.sum(y_true_pc * K.square(K.sigmoid(y_pre_xy) - y_true_xy))
    # wh_loss = 5 * K.sum(y_true_pc * K.square(K.sqrt(K.exp(y_pre_wh)) - K.sqrt(y_true_wh)))
    # pc_loss = K.sum(K.binary_crossentropy(y_true_pc, y_pre_pc, from_logits=True))
    # xy_loss=xy_loss.eval(session=sess)
    # wh_loss=wh_loss.eval(session=sess)
    # pc_loss=pc_loss.eval(session=sess)
    # sess.close()
    # return xy_loss+wh_loss+pc_loss


def loss(y):
    y_pre = y[0]
    y_true = y[1]
    y_true_pc = y_true[..., 0:1]
    y_pre_pc = y_pre[..., 0:1]
    y_pre_xy = y_pre[..., 1:3]
    y_pre_wh = y_pre[..., 3:5]
    y_true_xy = y_true[..., 1:3]
    y_true_wh = y_true[..., 3:5]
    xy_loss = 5 * K.sum(y_true_pc * K.square(K.sigmoid(y_pre_xy) - y_true_xy))
    # perform well :adam(1e-5)
    wh_loss = 5 * K.sum(y_true_pc * K.square(K.sqrt(K.exp(y_pre_wh)) - K.sqrt(y_true_wh)))

    # 计算各个cell里的iou,返回size(batch_size,x,y,1)
    ious = findIOUs(y_true_pc, y_true_xy, y_true_wh, y_pre_pc, y_pre_xy, y_pre_wh)

    # 忽略大于0.5的iou,因为出现了非0且重复检测的框,但不做 loss 惩罚
    iou_mask = K.cast(ious < 0.5, K.dtype(y_pre_pc))
    conf_loss = K.sum(
        0.5 * (1 - y_true_pc) * K.square(K.sigmoid(y_pre_pc) - y_true_pc) * iou_mask + y_true_pc * K.square(
            K.sigmoid(y_pre_pc) - y_true_pc))

    # pc_loss = K.sum((1 - y_true_pc) * K.square(K.sigmoid(y_pre_pc) - y_true_pc)
    #                 + y_true_pc * K.square(K.sigmoid(y_pre_pc) - y_true_pc))
    return xy_loss + wh_loss + conf_loss


def findIOUs(y_true_pc, y_true_xy, y_true_wh, y_pre_pc, y_pre_xy, y_pre_wh):

    batchsize = K.shape(y_true_wh)[0]  # 必须用tf的shape否则不能进行下去了
    grid = generate_base_cell_pos(y_true_wh.shape[1], y_true_wh.shape[2])
    ious = tf.TensorArray(K.dtype(y_true_xy), size=1, dynamic_size=True,infer_shape=False)
    def loop_body(x, ious):
        # 真实值加上偏移量算IOU
        bool_y_true_pc = K.cast(y_true_pc[x][..., 0],tf.bool)
        box_xy = tf.boolean_mask(y_true_xy[x][..., 0:2] + grid, bool_y_true_pc)
        box_wh = tf.boolean_mask(y_true_wh[x][..., 0:2], bool_y_true_pc)

        result = box_iou(y_pre_xy[x][..., 0:2] + grid, y_pre_wh[x], box_xy, box_wh)
        result = K.max(result,axis=-1,keepdims=True)
        ious = ious.write(x, result)

        return x + 1, ious

    _, ious = K.control_flow_ops.while_loop(lambda x, *args: x < batchsize, loop_body, [0, ious])
    ious = ious.stack()
    return ious  # return size(batch,size,size,1)


def generate_base_cell_pos(x, y):
    grid_x = K.tile(K.reshape(K.arange(0, stop=x), [-1, 1, 1]), [1, x, 1])
    grid_y = K.tile(K.reshape(K.arange(0, stop=y), [1, -1, 1]), [y, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, tf.float32)
    return grid  # size = (x,y,2)


def box_iou(y_pre_xy, y_pre_wh, y_true_xy, y_true_wh):
    # y_pre_xy:(x,y,2)
    # y_pre_wh:(x,y,2)
    # y_true_xy:(?,2)
    # y_true_wh:(?,2)

    # y_true_xy = K.expand_dims(y_true_xy,axis=0)
    # y_true_wh = K.expand_dims(y_true_wh,axis=0)
    y_pre_wh = K.expand_dims(y_pre_wh, -2)
    y_pre_xy = K.expand_dims(y_pre_xy, -2)
    # A = K.sigmoid(y_pre_xy[..., 0:1]) + K.exp(y_pre_wh[..., 0:1]) / 2
    # B = y_true_xy[..., 0:1] + y_true_wh[..., 0:1] / 2
    # x_max = K.maximum(y_true_xy[..., 0:1] + y_true_wh[..., 0:1] / 2,
    #                   K.sigmoid(y_pre_xy[..., 0:1]) + K.exp(y_pre_wh[..., 0:1]) / 2)
    # y_max = K.maximum(y_true_xy[..., 1:2] + y_true_wh[..., 1:2] / 2,
    #                   K.sigmoid(y_pre_xy[..., 1:2]) + K.exp(y_pre_wh[..., 1:2]) / 2)
    # x_min = K.minimum(y_true_xy[..., 0:1] - y_true_wh[..., 0:1] / 2,
    #                   K.sigmoid(y_pre_xy[..., 0:1]) - K.exp(y_pre_wh[..., 0:1]) / 2)
    # y_min = K.minimum(y_true_xy[..., 1:2] - y_true_wh[..., 1:2] / 2,
    #                   K.sigmoid(y_pre_xy[..., 1:2]) - K.exp(y_pre_wh[..., 1:2]) / 2)
    xy_max = K.maximum(y_true_xy + y_true_wh / 2,
                       K.sigmoid(y_pre_xy) + K.exp(y_pre_wh) / 2)

    xy_min = K.minimum(y_true_xy - y_true_wh,
                       K.sigmoid(y_pre_xy) - K.exp(y_pre_wh) / 2)

    # uni_w = K.maximum(y_true_wh[..., 0:1] + K.exp(y_pre_wh[..., 0:1]) - (xy_max[0] - xy_min[0]), 0.)
    # uni_h = K.maximum(y_true_wh[..., 1:2] + K.exp(y_pre_wh[..., 1:2]) - (xy_max[1] - xy_min[1]), 0.)
    uni_wh = K.maximum(y_true_wh + K.exp(y_pre_wh) - (xy_max - xy_min), 0.)
    # print(uni_wh[0].shape)
    # print(uni_wh[:][:][:][0].shape)
    # print(uni_wh[...,0].shape)
    # print(uni_wh[:,:,:,0].shape)
    ious = (uni_wh[..., 0] * uni_wh[..., 1]) / (
                y_true_wh[..., 0] * y_true_wh[..., 1] + K.exp(y_pre_wh[..., 0]) * K.exp(
            y_pre_wh[..., 1]) - uni_wh[..., 0] * uni_wh[..., 1])

    return ious  # return size =(x,y,?)
