# from rest_framework.request import Request
# from rest_framework.response import Response
from urllib import request

from django.shortcuts import render, redirect
from django import forms
import cv2
import numpy as np
import time
import os
import mediapipe as mp
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import ImageFont, ImageDraw, Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import pandas as pd
import pickle
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau
import pandas as pd
from tensorflow.keras.models import load_model
# Create your views here.

def home(reguest):
    # ml = pickle.load(open('../FinalProject/ml_model.pkl', 'rb'))
    # # print(ml)
    # if request.method == 'POST' and 'run_script' in request.POST:
    #     # import function to run
    #     from FinalProject.FinalApp.views import getPredictions
    #
    #     # call function
    #     getPredictions()
    #
    #     # return user to required page
    #     return render(request, 'index.html', {'products': getPredictions()})
    #     # return HttpResponseRedirect(reverse(FinalApp:getPredictions)
    return render(reguest,'index.html')


def getPredictions(reguest):
    train = pd.read_csv('/Users/user/Desktop/FinalBootcamp/FinalProject/FinalProject/train_ar_last1.csv')
    test = pd.read_csv('/Users/user/Desktop/FinalBootcamp/FinalProject/FinalApp/Ar_test_last1.csv')
    train.dropna(inplace=True)
    test.dropna(inplace=True)

    train_df = pd.read_csv("/Users/user/Desktop/FinalBootcamp/FinalProject/FinalProject/train_ar_last1.csv")
    test_df = pd.read_csv("/Users/user/Desktop/FinalBootcamp/FinalProject/FinalApp/Ar_test_last1.csv")

    y_train = train_df['lable']
    y_test = test_df['lable']
    del train_df['lable']
    del test_df['lable']

    from sklearn.preprocessing import LabelBinarizer
    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.fit_transform(y_test)

    x_train = train_df.values
    x_test = test_df.values

    x_train = x_train / 255
    x_test = x_test / 255

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(x_train)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5,
                                                min_lr=0.00001)

    model = Sequential()
    model.add(Conv2D(75, (3, 3), strides=1, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))
    model.add(Conv2D(50, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))
    model.add(Conv2D(25, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2, 2), strides=2, padding='same'))
    model.add(Flatten())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=24, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()

    history = model.fit(datagen.flow(x_train, y_train, batch_size=128), epochs=20, validation_data=(x_test, y_test),
                        callbacks=[learning_rate_reduction])

    model.save('signmodel.h5')
    model = load_model('signmodel.h5')

    mphands = mp.solutions.hands
    hands = mphands.Hands()
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    _, frame = cap.read()

    h, w, c = frame.shape

    img_counter = 0
    analysisframe = ''
    # keywords = {'0':'ain','2':'aleff', '3':'bb', '4':'dal','5':'dha', '6':'dhad', '7':'fa', '8':'gaaf',
    #            '9':'ghain', '11':'haa','12':'jeem','13':'kaaf', '14':'khaa', '15':'la', '16':'laam',
    #           '17':'meem', '18':'nun', '19':'ra', '20':'saad', '21':'seen',  '24':'taa',
    #          '25':'thaa', '26':'thal'}

    # letterpred = ['0', '2', '3', '4', '5', '6', '7', '8', '9', '11', '12',
    #          '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '24',
    #         '25', '26']

    letterpred = ['ع', 'أ', 'ب', 'د', 'ظ', 'ض', 'ف', 'ق', 'غ', 'ه', 'ج',
                  'ك', 'خ', 'لا', 'ل', 'م', 'ن', 'ر', 'ص', 'س', 'ش', 'ت',
                  'ث', 'ذ']
    while True:
        _, frame = cap.read()

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("تم ايقاف الكاميرا ...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            analysisframe = frame
            showframe = analysisframe
            cv2.imshow("Frame", showframe)
            framergbanalysis = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2RGB)
            resultanalysis = hands.process(framergbanalysis)
            hand_landmarksanalysis = resultanalysis.multi_hand_landmarks
            if hand_landmarksanalysis:
                for handLMsanalysis in hand_landmarksanalysis:
                    x_max = 0
                    y_max = 0
                    x_min = w
                    y_min = h
                    for lmanalysis in handLMsanalysis.landmark:
                        x, y = int(lmanalysis.x * w), int(lmanalysis.y * h)
                        if x > x_max:
                            x_max = x
                        if x < x_min:
                            x_min = x
                        if y > y_max:
                            y_max = y
                        if y < y_min:
                            y_min = y
                    y_min -= 20
                    y_max += 20
                    x_min -= 20
                    x_max += 20
    analysisframe = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2GRAY)
    analysisframe = analysisframe[y_min:y_max, x_min:x_max]
    analysisframe = cv2.resize(analysisframe, (28, 28))

    nlist = []
    rows, cols = analysisframe.shape
    for i in range(rows):
        for j in range(cols):
            k = analysisframe[i, j]
            nlist.append(k)

    datan = pd.DataFrame(nlist).T
    colname = []
    for val in range(784):
        colname.append(val)
    datan.columns = colname

    pixeldata = datan.values
    pixeldata = pixeldata / 255
    pixeldata = pixeldata.reshape(-1, 28, 28, 1)
    prediction = model.predict(pixeldata)
    predarray = np.array(prediction[0])
    letter_prediction_dict = {letterpred[i]: predarray[i] for i in range(len(letterpred))}
    predarrayordered = sorted(predarray, reverse=True)
    high1 = predarrayordered[0]
    high2 = predarrayordered[1]
    high3 = predarrayordered[2]
    for key, value in letter_prediction_dict.items():
        if value == high1:
            print("Predicted Character 1: ", key)
            print('Confidence 1: ', 100 * value)

        elif value == high2:
            print("Predicted Character 2: ", key)
            print('Confidence 2: ', 100 * value)
        elif value == high3:
            print("Predicted Character 3: ", key)
            print('Confidence 3: ', 100 * value)
        time.sleep(5)

        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        hand_landmarks = result.multi_hand_landmarks
        if hand_landmarks:
            for handLMs in hand_landmarks:
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in handLMs.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x > x_max:
                        x_max = x
                    if x < x_min:
                        x_min = x
                    if y > y_max:
                        y_max = y
                    if y < y_min:
                        y_min = y
                y_min -= 20
                y_max += 20
                x_min -= 20
                x_max += 20
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        cap.release()
        cv2.destroyAllWindows()

    pickle.dump(model, open("ml_model2.pkl", "wb"))
    return render(reguest, 'index.html')