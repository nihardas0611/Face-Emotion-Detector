import streamlit as st
import cv2
from detector import detector


cap = cv2.VideoCapture(0)
st.title('Face Emotion Detector')

frame_placeholder=st.empty()
stop_button=st.button('STOP')

while cap.isOpened() and not stop_button:
    ret,frame=cap.read()

    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame=detector(frame)
    frame_placeholder.image(frame,channels='RGB')

    if cv2.waitKey(1) & 0xFF==ord('q') or stop_button:
        break

frame_placeholder.write('Video Capture has Ended')
cap.release()
cv2.destroyAllWindows()