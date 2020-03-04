from __future__ import division, print_function, absolute_import

import os
import random
import numpy as np
import tensorflow as tf

from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', dest='dataset_name', default='cuhk03')
    parser.add_argument('--model', '-m', dest='model', default='./model/cuhk03.ckpt')
    return parser.parse_args()


def extract_feature(dataset_name, dir_path, net):
    imgs = []
    xs = []
    infos = []
    img_names = os.listdir(dir_path)
    for image_name in tqdm(img_names):
        if '.jpg' not in image_name:
            continue

        arr = image_name.replace('.jpg', '').split('_')
        person = int(arr[0])

        if dataset_name == 'market':
            if person < 1 or person > 200:
                continue
            camera = int(arr[1][1])
        elif dataset_name == 'cuhk03':
            camera = int(arr[1])
        image_path = os.path.join(dir_path, image_name)
        img = image.load_img(image_path, target_size=(224, 224))
        imgs.append(img)
        x = image.img_to_array(img)
        x = preprocess_input(x)
        xs.append(x)
        infos.append((person, camera))

    xs = np.asarray(xs)
    features = net.predict(xs, verbose=1)
    features = features.reshape(features.shape[0], -1)
    return imgs, features, infos


def FRR2th(y, tX, tY):
    idx = np.abs(tY - y).argmin()
    if y == tY[idx]:
        x = tX[idx]
    else:
        idx = idx-1 if y < tY[idx] else idx
        x = ((y-tY[idx])*tX[idx+1] + (tY[idx+1]-y)*tX[idx])/(tY[idx+1]-tY[idx])
    return x


def th2FAR(x, fX, fY):
    if x > fX[0]:
        return fY[0]
    if x < fX[-1]:
        return fY[-1]
    idx = np.abs(fX - x).argmin()
    if x == fX[idx]:
        y = fY[idx]
    else:
        idx = idx+1 if x < fX[idx] else idx
        if fX[idx-1] == fX[idx]:
            y = fY[idx]
        else:
            y = ((x-fX[idx])*fY[idx-1] + (fX[idx-1]-x)*fY[idx])/(fX[idx-1]-fX[idx])
    return y


if __name__ == "__main__":
    args = arg_parser()
    dataset_name = args.dataset_name
    if dataset_name == 'market':
        DATASET = './dataset/Market'
        TEST = os.path.join(DATASET, 'bounding_box_test')
        QUERY = os.path.join(DATASET, 'query')
    elif dataset_name == 'cuhk03':
        DATASET = './dataset/CUHK03/detected'
        TEST = os.path.join(DATASET, 'val')
        QUERY = os.path.join(DATASET, 'query')


    # use GPU to calculate the similarity matrix
    query_t = tf.placeholder(tf.float32, (None, None))
    test_t = tf.placeholder(tf.float32, (None, None))
    query_t_norm = tf.nn.l2_normalize(query_t, dim=1)
    test_t_norm = tf.nn.l2_normalize(test_t, dim=1)
    tensor = tf.matmul(query_t_norm, test_t_norm, transpose_a=False, transpose_b=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)

    # load model
    net = load_model(args.model)
    net = Model(input=net.input, output=net.get_layer('avg_pool').output)

    test_i, test_f, test_info = extract_feature(dataset_name, TEST, net)
    if dataset_name == 'market':
        query_i, query_f, query_info = extract_feature(dataset_name, QUERY, net)
    elif dataset_name == 'cuhk03':
        query_i = test_i.copy()
        query_f = test_f.copy()
        query_info = test_info.copy()

    result = sess.run(tensor, {query_t: query_f, test_t: test_f})
    result_argsort = np.argsort(result, axis=1)

    #  ========  calc mAP and Rank 1, 3, 5, 10 acc  ========
    match = []
    junk = []

    for q_index, (qp, qc) in enumerate(tqdm(query_info)):
        tmp_match = []
        tmp_junk = []
        for t_index, (tp, tc) in enumerate(test_info):
            if tp == qp and qc != tc:
                tmp_match.append(t_index)
            elif tp == qp or tp == -1:
                tmp_junk.append(t_index)
        match.append(tmp_match)
        junk.append(tmp_junk)

    TEST_NUM = len(test_f)
    QUERY_NUM = len(query_f)

    mAP = 0.0
    CMC = np.zeros([len(query_info), len(test_info)])
    for idx in range(len(query_info)):
        recall = 0.0
        precision = 1.0
        hit = 0.0
        cnt = 0
        ap = 0.0
        YES = match[idx]
        IGNORE = junk[idx]
        for i in list(reversed(range(0, TEST_NUM))):
            k = result_argsort[idx][i]
            if k in IGNORE:
                continue
            else:
                cnt += 1
                if k in YES:
                    CMC[idx, cnt-1:] = 1
                    hit += 1

                tmp_recall = hit/len(YES)
                tmp_precision = hit/cnt
                ap = ap + (tmp_recall - recall)*((precision + tmp_precision)/2)
                recall = tmp_recall
                precision = tmp_precision
            if hit == len(YES):
                break
        mAP += ap

    rank_1 = np.mean(CMC[:, 0])
    rank_3 = np.mean(CMC[:, 2])
    rank_5 = np.mean(CMC[:, 4])
    rank_10 = np.mean(CMC[:, 9])
    mAP /= QUERY_NUM
    print('====================================')
    print('====================================')
    print()
    print('1: {}, 3: {}, 5: {}, 10: {}, mAP: {}'.format(rank_1, rank_3, rank_5, rank_10, mAP))
    print()
    print('====================================')
    print('====================================')

    #  ========  draw ROC curve  ========

    tpair = []
    fpair = []

    for q_index, (qp, qc) in enumerate(tqdm(query_info)):
        for t_index, (tp, tc) in enumerate(test_info):
            if tp == qp and qc != tc:
                tpair.append((q_index, t_index))
            elif tp == qp or tp == -1:
                continue
            else:
                fpair.append((q_index, t_index))

    random.seed(0)
    tpair_sample = random.sample(tpair, 5000)
    fpair_sample = random.sample(fpair, 5000)

    tscore = np.asarray([result[q][t] for q, t in tpair_sample])
    fscore = np.asarray([result[q][t] for q, t in fpair_sample])
    N = len(tscore)
    tX = np.sort(tscore)
    tY = np.asarray([(i+1)/N for i in range(N)])

    N = len(fscore)
    fX = np.sort(fscore)[::-1]
    fY = np.asarray([(i+1)/N for i in range(N)])

    N = 1000
    FRR = [i/N for i in range(N+1)]
    FAR = [th2FAR(FRR2th(frr, tX, tY), fX, fY) for frr in FRR]
    FAR[0] = 1

    model = args.model.split('/')[-1].replace('.ckpt', '')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(FRR, FAR, color="black", label='{}->{}'.format(model, dataset_name))
    ax.set_xlabel("FRR(False Rejection Rate)")
    ax.set_ylabel("FAR(False Acceptance Rate)")
    ax.set_title("The relationship between FRR and FAR")
    plt.grid()
    plt.legend()
    plt.savefig("./result/ROC_curve_{}_{}.png".format(dataset_name, model))
    np.save('result/FRR_{}_{}.npy'.format(dataset_name, model), FRR)
    np.save('result/FAR_{}_{}.npy'.format(dataset_name, model), FAR)
