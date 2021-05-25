# original author: Zylo117
# modified by: Marek Sicha

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import time
import torch
import cv2
import numpy as np
import tensorflow as tf

from torch.backends import cudnn

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, plot_one_box


compound_coef = 2
num_classes_detector = 1
force_input_size = None  # set None to use default size

# Image's path
img_path = 'path to your test image'

# Classifier´s model path
model_classifier = tf.keras.models.load_model("saved_model/")

# Detector´s model path
model_detector = 'efficientdet-d2_72_36000.pth'

anchor_ratios = [(1.0, 1.0), (1.0, 1.0), (1.0, 0.9)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

# Detector
threshold = 0.5
iou_threshold = 0.5

# Classifier
classifier_threshold = 0.8

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

color_list = standard_to_bgr(STANDARD_COLORS)

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

if use_cuda:
    x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
else:
    x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
model = EfficientDetBackbone(compound_coef=compound_coef,
                             num_classes=num_classes_detector,
                             ratios=anchor_ratios, scales=anchor_scales)
model.load_state_dict(torch.load(model_detector))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

with torch.no_grad():
    features, regression, classification, anchors = model(x)

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    out = postprocess(x,
                      anchors, regression, classification,
                      regressBoxes, clipBoxes,
                      threshold, iou_threshold)


def classifier_img(img):
    img = cv2.equalizeHist(cv2.cvtColor(cv2.resize(img, (32, 32)), cv2.COLOR_BGR2GRAY))
    img = img.reshape(1, 32, 32, 1)
    prediction = np.amax(model_classifier.predict(img))
    class_ID = np.argmax(model_classifier.predict(img), axis=-1)

    return(img, prediction, class_ID)


def get_class_name(class_ID):
    classes = open('classes_cz.txt', 'r')
    class_line = classes.read()
    class_name = class_line.splitlines()

    return(class_name[class_ID[0]])


def display(preds, imgs, imshow=True, imwrite=True):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        imgs[i] = imgs[i].copy()
        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int32)
            _, prediction, class_ID = classifier_img((imgs[i][y1:y2, x1:x2]))
            class_name = get_class_name(class_ID)
            if prediction <= classifier_threshold:
                continue
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=class_name,
                         score=prediction, color=color_list[(int(class_ID))])

        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if imwrite:
            cv2.imwrite(f'test/img_inferred_d{compound_coef}_this_repo_{i}.jpg', imgs[i])


out = invert_affine(framed_metas, out)

display(out, ori_imgs, imshow=True, imwrite=False)

print('running speed test...')
with torch.no_grad():
    print('test1: model inferring and postprocessing')
    print('inferring image for 10 times...')
    t1 = time.time()
    for _ in range(10):
        _, regression, classification, anchors = model(x)

        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)
        out = invert_affine(framed_metas, out)
        for j in range(len(out[0]['rois'])):
            x1, y1, x2, y2 = out[0]['rois'][j].astype(np.int32)
            _, prediction, class_ID = classifier_img(ori_imgs[0][y1:y2, x1:x2])
            class_name = get_class_name(class_ID)

    t2 = time.time()
    tact_time = (t2 - t1) / 10
    print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')
