import keras
import numpy as np
import cv2

model=keras.models.load_model('model_best_2.h5')

def detect_face_emotion(img):
    y_pred=model.predict(img.reshape(1,224,224,3))
    return y_pred[0]

def draw_label(img,text,pos,bg_color):
    text_size=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1,cv2.FILLED)
    end_x=pos[0]+text_size[0][0]-2
    end_y=pos[0]+text_size[0][1]-2

    cv2.rectangle(img,pos,(end_x,end_y),bg_color,cv2.FILLED)
    cv2.putText(img,text,pos,cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1,cv2.LINE_AA)

haar=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_face(img):
    coords=haar.detectMultiScale(img)
    return coords

def detector(frame):
    frame = cv2.flip(frame, 1)
    img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    coords = detect_face(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    for x, y, w, h in coords:
        cv2.rectangle(frame, (x-20, y-20), (x + w+20, y + h+20), (255, 0, 0), 3)

    cropped_img = img1[y-20:y + h+20, x-20:x + w+20]
    img = cv2.resize(cropped_img, (224, 224))
    y_pred = detect_face_emotion(img)
    y_pred = np.argmax(y_pred)

    if (y_pred == 0):
        draw_label(frame, "angry", (25, 25), (0, 0, 255))
    elif y_pred == 1:
        draw_label(frame, "disgust", (25, 25), (0, 255, 255))
    elif y_pred == 2:
        draw_label(frame, "fear", (25, 25), (255, 0, 255))
    elif y_pred == 3:
        draw_label(frame, "happy", (25, 25), (0, 255, 0))
    elif y_pred == 4:
        draw_label(frame, "neutral", (25, 25), (255, 255, 255))
    elif y_pred == 5:
        draw_label(frame, "sad", (25, 25), (255, 0, 0))
    else:
        draw_label(frame, "surprise", (25, 25), (255, 255, 0))

    return frame