import pickle 
import streamlit as st
import numpy as np
from PIL import Image

scaler_file = pickle.load(open('iris_scaler.pkl', 'rb')) 
model_file = pickle.load(open('iris_model.pkl', 'rb'))

def pred_output(user_input):
    scaled_input = scaler_file.transform(np.array(user_input).reshape(-1,4))
    ypred = model_file.predict(scaled_input)
    return ypred[0]

def main(): 
    st.title('NHÓM 3 - ĐỀ 13 - PHÂN LOẠI HOA DIÊN VĨ')
    st.divider()

    # Input Variables
    sepalLength = st.text_input('NHẬP CHIỀU DÀI ĐÀI HOA (cm):')
    sepalWidth = st.text_input('NHẬP CHIỀU RỘNG ĐÀI HOA (cm):')
    petalLength = st.text_input('NHẬP CHIỀU DÀI CÁNH HOA (cm):')
    petalWidth = st.text_input('NHẬP CHIỀU RỘNG CÁNH HOA (cm):')

    if sepalLength.isalpha() or sepalWidth.isalpha() or petalLength.isalpha() or petalWidth.isalpha():
        st.error("Vui lòng nhập giá trị số thực!")

    # Button to predict
    if st.button('Phân Loại:'):
        user_input = [sepalLength, sepalWidth, petalLength, petalWidth]
        make_prediction = pred_output(user_input)
        species = ""
        if("Iris-virginica" == make_prediction):
            species = "Loài hoa: Iris Virginica"
            image = Image.open('./iris_virginica.jpg')
        elif("Iris-versicolor" == make_prediction):
            species = "Loài hoa: Iris Versicolor"
            image = Image.open('./iris_versicolor.jpg')
        elif("Iris-setosa" == make_prediction):
            species = "Loài hoa: Iris Setosa"
            image = Image.open('./iris_setosa.jpg')
        else:
            species = "Loài hoa chưa xác định"

        st.text(species)
        st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    
if __name__ == '__main__':
    main()

