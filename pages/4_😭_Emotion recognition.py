import cv2
import numpy as np
from keras.models import load_model
import streamlit as st
import av
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Đặt tiêu đề trang và biểu tượng trang
st.set_page_config(page_title="Emotion recognition", page_icon="😭")
cap = cv2.VideoCapture(0)
# Load model
model=load_model('./pages/model_file_30epochs.h5')

# Load bộ phân loại khuôn mặt Haar Cascade
faceDetect=cv2.CascadeClassifier('./pages/haarcascade_frontalface_default.xml')

# Tạo từ điển nhãn cảm xúc
labels_dict={0:'Angry',1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert the frame to BGR format
        frame = cv2.cvtColor(frame.to_ndarray(format="rgb24"), cv2.COLOR_RGB2BGR)
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = faceDetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        # Draw rectangles around the faces and label the emotions
        for (x, y, w, h) in faces:
            sub_face_img = gray[y:y + h, x:x + w]
            resized = cv2.resize(sub_face_img, (48, 48))
            normalize = resized / 255.0
            reshaped = np.reshape(normalize, (1, 48, 48, 1))
            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
            cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # Convert the frame back to RGB format for display on the web
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)

# Tạo một đối tượng VideoTransformer và chạy ứng dụng web
webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)