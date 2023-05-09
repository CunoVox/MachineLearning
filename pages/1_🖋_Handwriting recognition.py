import streamlit_drawable_canvas as canvas
import tkinter as tk
from PIL import ImageTk, Image
import streamlit as st
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import model_from_json 
from tensorflow.keras.optimizers import SGD 
import cv2
import pandas as pd
# Load model architecture and weights
model_architecture = "./pages/digit_config.json"
model_weights = "./pages/digit_weight.h5"
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)
optim = SGD()
model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"])

mnist = keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_test_image = X_test
RESHAPED = 784
X_test = X_test.reshape(10000, RESHAPED)
X_test = X_test.astype('float32')
#normalize in [0,1]
X_test /= 255
index = []
def main():
    root = tk.Tk()

    st.set_page_config(page_title="Handwriting recognition", page_icon="🖋", layout="wide")
    
   # st.title(':blue[Nhận dạng chữ số viết Tay]  :✏️')
    st.markdown('<h1 style="color: #000000;">Handwriting recognition</h1>', unsafe_allow_html=True)
    # st.geometry('520x550')
    index = None
    image_tk = None
    # cvs_digit = canvas.st_canvas(width=421, height=285, background_color="white")
    
    
    
    # lbl_ket_qua = st.label(width=42, height=11, font=('Consolas', 14))
    
    
    # Tạo một đối tượng Text Element
    lbl_ket_qua = st.empty()

# Cài đặt chiều rộng và chiều cao của Text Element
    lbl_ket_qua.width = 42
    lbl_ket_qua.height = 11

# =============================================================================
# # Đặt nội dung và kiểu chữ của Text Element
#     lbl_ket_qua.markdown("Nội dung của nhãn")
#     lbl_ket_qua.markdown(f"<span style='font-family: Consolas; font-size: 14px;'>Nội dung của nhãn với kiểu chữ Consolas và kích thước 14px</span>", unsafe_allow_html=True)
# =============================================================================
    
    
    # btn_tao_anh = st.button('Tao anh', width=9)
    # btn_nhan_dang = st.button('Nhan dang', width=9)

    # btn_tao_anh = st.button('Tạo ảnh', key='btn_tao_anh', help='Tooltip cho nút Tạo ảnh')
    # btn_nhan_dang = st.button('Nhận dạng', key='btn_nhan_dang', help='Tooltip cho nút Nhận dạng')
    
# Sử dụng CSS để điều chỉnh chiều rộng của nút
    st.write('<style>div.row-widget.stButton > div{width: 50%;}</style>', unsafe_allow_html=True)
    
    
    index = []

  

    # Hàm ghi giá trị của mảng vào file txt
    def ghi_gia_tri_vao_file(mang, ten_file):
        with open(ten_file, 'w') as f:
            for gia_tri in mang:
                f.write(str(gia_tri) + '\n')
    
    # Hàm đọc giá trị từ file txt vào một mảng
    def doc_gia_tri_tu_file(ten_file):
        mang = []
        with open(ten_file, 'r') as f:
            for line in f:
                gia_tri = line.strip()
                mang.append(gia_tri)
        return mang

    if st.button('Generate Digit handwritten'):
        index = np.random.randint(0, 9999, 150)
        digit_random = np.zeros((10*28, 15*28), dtype=np.uint8)
        # if len(index) > 0:
        for i in range(0, 150):
                # if index[i] < len(X_test_image):
                m = i // 15
                n = i % 15
                digit_random[m*28:(m+1)*28, n*28:(n+1)*28] = X_test_image[index[i]]
        cv2.imwrite('./pages/digit_random.jpg', digit_random)
        image = Image.open('./pages/digit_random.jpg')
        st.image(image, width=600)
        ghi_gia_tri_vao_file(index, './pages/datamang.txt')
        # lbl_ket_qua.text('')
        print(index)
    if st.button('Recognition'):
        image = Image.open('./pages/digit_random.jpg')
        st.image(image, width=600)
        ketqua=[]
        index = doc_gia_tri_tu_file('./pages/datamang.txt')
        index=list(map(int, index))
        print(index)

        
        
        X_test_sample = np.zeros((150, 784), dtype=np.float32)
        if len(index) > 0:
            for i in range(0, 150):
                if index[i] < len(X_test):
                    X_test_sample[i] = X_test[index[i]]
        prediction = model.predict(X_test_sample)
        s = ''
        for i in range(0,150):
                ket_qua = np.argmax(prediction[i])
                s = s + str(ket_qua).ljust(5)
                if (i+1) % 15 == 0:
                    s = s + '\n'
        
        st.text(s)
        


if __name__ == "__main__":
    main()

