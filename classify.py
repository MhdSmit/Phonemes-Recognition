# USAGE
# python classify.py --model pokedex.model --labelbin lb.pickle --image examples/charmander_counter.png

# import the necessary packages
from keras.preprocessing.image import img_to_array
from os import listdir
from os.path import isfile,join
from keras.models import load_model
import imutils
from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import img_to_array

import numpy as np
import argparse
import random
import pickle
import cv2
import sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras import backend as k
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
config=tf.ConfigProto()
config.gpu_options.allow_growth = True
k.set_session(sess)
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
args0="C:\\Users\\smit\\PycharmProjects\\pokemon - Copy\\pokedex_second_test.model"
args1="C:\\Users\\smit\\PycharmProjects\\pokemon - Copy\\lb1_second_test.pickle"
args2="C:\\Users\\smit\\PycharmProjects\\pokemon - Copy\\examples\\lettererset7Spect_flat.png"
ar = np.array([args0, args1, args2])
args = ar
# load the image
image = cv2.imread(args[2])
output = image.copy()
 
# pre-process the image for classification
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network and the label
# binarizer
print("[INFO] loading network...")
model = load_model(args[0])
lb = pickle.loads(open(args[1], "rb").read())

# classify the input image
print("[INFO] classifying image...")
proba = model.predict(image)[0]
print(proba)
idx = np.argmax(proba)
label = lb.classes_[idx]
# we'll mark our prediction as "correct" of the input image filename
# contains the predicted label text (obviously this makes the
# assumption that you have named your testing image files this way)
filename = args[2][args[2].rfind(os.path.sep) + 1:]
correct = "correct" if filename.rfind(label) != -1 else "incorrect"

# build the label and draw the label on the image
label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
output = imutils.resize(output, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# show the output image
print("[INFO] {}".format(label))
#cv2.imshow("Output", output)
#cv2.waitKey(0)
#####
phonems_name = ['aa','ae','ah','ao','aw','ax','ax-h','axr','ay','b','ch','d','dh','dx','eh','el','en','eng','er','ey','f','g','hh','hv','ih',
                'ix','iy','jh','k','l','m','n','ng','nx','ow','oy','p','q','r','s','sh','t','th','uh','uw','ux','v','w','y','z','zh']
result = []
for i in range(0,51):
    print(i)
    foldername = 'letter_'+phonems_name[i]
    mypath = 'C:\\Users\\smit\\PycharmProjects\\pokemon - Copy\\spec_letters_10\\' + foldername + '\\'
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    count = 0
    dis = 0
    predict = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for idx, val in enumerate(files):
        image1 = cv2.imread(mypath + val)
        output = image1.copy()
        # pre-process the image for classification
        image1 = cv2.resize(image1, (96, 96))
        image1 = image1.astype("float") / 255.0
        image1 = img_to_array(image1)
        image1 = np.expand_dims(image1, axis=0)
        proba1 = model.predict(image1)
        idx = np.argmax(proba1)
        label = lb.classes_[idx]
        predict[idx] = predict[idx] + 1
        
    result.append(predict)

print('######')
print(count)
print('######')
print(dis)
print(predict)
#np.save('matmat.txt',predict)
np.savetxt('matmat.txt',result,'%.d')
