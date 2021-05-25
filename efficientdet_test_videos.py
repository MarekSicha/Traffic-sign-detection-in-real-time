# original author: Zylo117
# modified by: Marek Sicha

"""
Simple Inference Script of EfficientDet-Pytorch for detecting objects on webcam
"""
import time
import torch
import cv2
import numpy as np
import tensorflow as tf

from torch.backends import cudnn

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess_video, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, plot_one_box

compound_coef = 2
num_classes_detector = 1
force_input_size = None  # set None to use default size

# Video's path
video_src = 0  # set int to use webcam, set str to read from a video file

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

anchor_ratios = [(1.0, 1.0), (1.0, 1.0), (1.0, 0.9)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

color_list = standard_to_bgr(STANDARD_COLORS)

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=num_classes_detector,
                             ratios=anchor_ratios, scales=anchor_scales)
model.load_state_dict(torch.load(model_detector))
model.requires_grad_(False)
model.eval()


if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()


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


def display(preds, imgs):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            return imgs[i]

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int32)
            _, prediction, class_ID = classifier_img((imgs[i][y1:y2, x1:x2]))
            class_name = get_class_name(class_ID)
            if prediction <= classifier_threshold:
                continue
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=class_name,
                         score=prediction, color=color_list[(int(class_ID))])

        return imgs[i]


regressBoxes = BBoxTransform()
clipBoxes = ClipBoxes()

# Video capture
cap = cv2.VideoCapture(video_src)
prev_frame_time = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    new_frame_time = time.time()
    ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)

    out = invert_affine(framed_metas, out)
    img_show = display(out, ori_imgs)
    cv2.putText(img_show, str(int(1/(new_frame_time-prev_frame_time))),
                (4, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (100, 255, 0), 1, cv2.LINE_AA)
    prev_frame_time = new_frame_time
    cv2.imshow('frame', img_show)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

