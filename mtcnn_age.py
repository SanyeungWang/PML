#import argparse,cv2
import os
import os.path as path
from mtcnn.mtcnn import MTCNN
import matplotlib as plt
from pylab import *
import math
import cv2
import numpy as np
from collections import defaultdict
from PIL import Image, ImageDraw
from matplotlib.pyplot import imshow

# Last revision: 16:50 21-7-2019

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

imdb_train = './IMDB_WIKI/train'
imdb_val = './IMDB_WIKI/val'
imdb_wiki = './DATASETS/IMDB_WIKI/IMDB_WIKI_Crop_All_382326'
morph = './DATASETS/MORPH/MORPH'

def test_image(dir=imdb_wiki, crop_factor=0, index_min=0, index_max=300000):
    mtcnn = MTCNN()
    files = os.listdir(dir)
    print('min is {}'.format(index_min))
    for i, file in enumerate(files[index_min:index_max]):
        print(i+index_min)
        imgpath = dir + "/" + file
        img = cv2.imread(imgpath)
        # img = cv2.resize(img, (224, 224))
        try:
            a = mtcnn.detect_faces(img)
        except Exception as e:
            pass
            print('error of mtcnn thsi img {}'.format(file))
            continue
        img_doc = path.join(dir, 'new', os.path.splitext(file)[0]+'.jpg')
        index_0_img_doc = path.join(dir, 'bad', os.path.splitext(file)[0]+'.jpg')
        if os.path.exists(img_doc) or os.path.exists(index_0_img_doc):
            continue
        # print('total box: %s . image name: %s' % (len(a), imgpath))
        if len(a) >= 1:
            max_scores_index = 0
            for i in range(len(a)):
                if a[i]['confidence'] > a[max_scores_index]['confidence']:
                    max_scores_index = i
            # result = [bbox[max_scores_index], scores[max_scores_index], landmarks[max_scores_index]]
            # print(result)
            box = a[max_scores_index]['box']
            scor = a[max_scores_index]['confidence']
            pts = a[max_scores_index]['keypoints']
            # print(box, scor, pts)
            # box = box.astype('int32')
            # print('x1= %s; y1 = %s; x2 = %s; y2 = %s; score = %s' % (box[1], box[0], box[3], box[2], scor))
            # img = cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 3)
            # bounding box
            # left_top
            box_x1 = box[0]
            box_y1 = box[1]
            # cv2.circle(img, (box_x1, box_y1), 1, (0, 255, 0), 2)
            # right_bottom
            box_x2 = box[0]+box[2]  # box3 是width box2 是heigh
            box_y2 = box[0]+box[3]
            # img = cv2.circle(img, (box_x2, box_y2), 1, (0, 255, 0), 2)
            crop_margin = crop_factor
            # size of face
            orig_size_x = box[2]
            orig_size_y = box[3]
            # coordinate of face bounding box with magin
            margin_box_x1 = box_x1 - orig_size_x * crop_margin
            margin_box_y1 = box_y1 - orig_size_y * crop_margin
            margin_box_x2 = box_x2 + orig_size_x * crop_margin
            margin_box_y2 = box_y2 + orig_size_y * crop_margin
            # pts = pts.astype('int32')
            # five landmarks
            # left eye
            left_eye = pts['left_eye']
            x1 = left_eye[0]
            y1 = left_eye[1]
            # right eye
            right_eye = pts['right_eye']
            x2 = right_eye[0]
            y2 = right_eye[1]

            img_height = img.shape[0]
            img_width = img.shape[1]
            img_channel = img.shape[2]
            print(img_height, img_width, img_channel)
            cropped_img = img[margin_box_y1: margin_box_y2, margin_box_x1: margin_box_x2]
            # cv2.imwrite(img_doc, cropped_img)

            # rotation image
            eye_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            dy = y2 - y1
            dx = x2 - x1
            angle = math.atan2(dy, dx) * 180. / math.pi
            rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
            rotated_img = cv2.warpAffine(img, rotate_matrix, (img_width, img_height))
            # cv2.imwrite(img_doc, rotated_img)
            # rotated_img = cv2.rectangle(rotated_img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 3)
            # origin_img = Image.fromarray(img)
            # plt.figure()
            # plt.subplot(1, 2, 1)
            # plt.imshow(origin_img)
            # plt.subplot(1, 2, 2)
            # plt.imshow(rotated_img)
            # plt.show()

            if margin_box_y1 < 0:
                margin_box_y1 = 0
            if margin_box_x1 < 0:
                margin_box_x1 = 0
            try:
                cv2.imwrite(img_doc, cropped_img)
            except Exception as e:
                pass
                continue
            cropped_img = rotated_img[margin_box_y1: margin_box_y2, margin_box_x1: margin_box_x2]
            try:
                cv2.imwrite(img_doc, cropped_img)
            except Exception as e:
                pass
                continue
        else:
            # pass
            try:
                cv2.imwrite(index_0_img_doc, img)
            except Exception as e:
                pass
                continue
            # with open('Validation_box_more1.txt', 'a+') as file_txt:
            #     file_txt.write('%s\n' % imgpath)


if __name__ == '__main__':
    a = 10000
    test_image(dir=cacd, index_min=a, index_max=a+10000)

    # test_camera()
