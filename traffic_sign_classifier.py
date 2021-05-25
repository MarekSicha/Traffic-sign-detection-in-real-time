# Author: Marek Sicha

import numpy as np
import cv2
import tensorflow as tf

from timeit import default_timer as timer

# Classifier´s model path
model = tf.keras.models.load_model('saved_model/')

# Image´s path
imgpath = 'path to your test image'


def get_class_name(class_ID):
    classes = open('classes_cz.txt', 'r')
    class_line = classes.read()
    class_name = class_line.splitlines()

    return(class_name[class_ID[0]])


win_name = 'Output'
img = cv2.imread(imgpath)
img_orig = img
img = cv2.equalizeHist(cv2.cvtColor(cv2.resize(img, (32, 32)), cv2.COLOR_BGR2GRAY))
img = img.reshape(1, 32, 32, 1)

start = timer()
prediction = model.predict(img)
class_ID = np.argmax(model.predict(img), axis=-1)
end = timer()


classID = get_class_name(class_ID)
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.moveWindow(win_name, 400, 400)
cv2.resizeWindow(win_name, 600, 600)
img_orig = cv2.copyMakeBorder(img_orig, 100, 100, 100, 100,
                              cv2.BORDER_CONSTANT, value=[255, 255, 255])

probability = np.amax(prediction)
probability = probability * 100

cv2.putText(img_orig, ('Class: ' + classID), (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 0, cv2.LINE_AA)
cv2.putText(img_orig, ('Probability: ' + (str(probability)[:4]) + ' %'),
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 0, cv2.LINE_AA)
cv2.imshow(win_name, img_orig)

print('Probability: ' + str(probability) + ' %')
print('Class: ' + classID)
print(end - start)
cv2.waitKey(0)
cv2.destroyAllWindows()
