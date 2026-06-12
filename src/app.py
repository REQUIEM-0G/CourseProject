import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import cv2

def center_digit(img):
    # знаходимо центр мас
    M = cv2.moments(img)
    if M['m00'] > 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # обчислюємо зсув, щоб центр мас став у точку (14, 14)
        rows, cols = img.shape
        shift_x = np.round(cols / 2.0 - cx).astype(int)
        shift_y = np.round(rows / 2.0 - cy).astype(int)
        
        # зсуваємо зображення
        M_shift = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        centered_img = cv2.warpAffine(img, M_shift, (cols, rows))
        return centered_img
    return img

# налаштування сторінки
st.set_page_config(page_title="Розпізнавання цифр", layout="centered")

st.title("Розпізнавання рукописних цифр")

# ініціалізація стану сесії для збереження результатів розпізнавання
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None  # тут зберігатимемо [клас, впевненість, масив ймовірностей, зображення]

# завантаження моделі
@st.cache_resource
def load_my_model():
    # чи існує файл 'mnist_model.h5'
    model = tf.keras.models.load_model('model/mnist_model.h5')
    return model

try:
    model = load_my_model()
    st.success("Модель успішно завантажена!")
except Exception as e:
    st.error(f"Помилка завантаження моделі: {e}")
    st.stop()

# інструкція
st.sidebar.header("Інструкція")
st.sidebar.write("""
1. Завантажте фото або намалюйте цифру.
2. Натисніть кнопку розпізнавання.
3. Результат залишиться на екрані до наступного аналізу.
""")

# функція передобробки для завантаженого фото
def preprocess_uploaded_image(img):
    img = img.convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    img_array = img_array.astype('float32') 
    img_array = np.expand_dims(img_array, axis=(0, -1))
    return img_array

# завантаження файлу
uploaded_file = st.file_uploader("Оберіть зображення цифри...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Завантажене зображення', width=150)
    
    if st.button('Розпізнати завантажену цифру'):
        processed_img = preprocess_uploaded_image(image)
        prediction = model.predict(processed_img)
        
        # зберігаємо в session_state
        st.session_state.prediction_data = {
            'label': np.argmax(prediction),
            'confidence': np.max(prediction) * 100,
            'chart': prediction[0],
            'source': 'upload'
        }

st.write("---")
st.write("### Або намалюйте цифру:")

# малювальна панель
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",  
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    if st.button('Розпізнати намальовану цифру'):
        img = canvas_result.image_data.astype('uint8')
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        
        cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(cnts) > 0:
            # знаходимо контур
            cnt = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            digit = img[y:y+h, x:x+w]
            
            # додаємо рівномірні відступи, щоб цифра була по центру квадрата
            size = max(w, h) + 40
            pad_img = np.zeros((size, size), dtype="uint8")
            dx = (size - w) // 2
            dy = (size - h) // 2
            pad_img[dy:dy+h, dx:dx+w] = digit
            
            # змінюємо розмір до 28x28
            img_28 = cv2.resize(pad_img, (28, 28), interpolation=cv2.INTER_AREA)

            img_28 = center_digit(img_28)
            
            # м'яке розмиття
            img_28 = cv2.GaussianBlur(img_28, (3, 3), 0)
            
            # підсилення контрасту
            _, img_28 = cv2.threshold(img_28, 50, 255, cv2.THRESH_BINARY)
            # повторне легке розмиття для згладжування після порогу
            img_28 = cv2.GaussianBlur(img_28, (3, 3), 0)

            st.image(img_28, width=100, caption="Модель бачить це")

            # підготовка для моделі
            img_array = img_28.astype('float32') 
            img_array = np.expand_dims(img_array, axis=(0, -1))
            
            prediction = model.predict(img_array)
            
            st.session_state.prediction_data = {
                'label': int(np.argmax(prediction)),
                'confidence': float(np.max(prediction) * 100),
                'chart': prediction[0],
                'source': 'canvas',
                'debug_img': img_28 # передаємо оброблене зображення
            }
        else:
            st.warning("Будь ласка, спочатку намалюйте щось!")

# виведення результатів розпізнавання
if st.session_state.prediction_data is not None:
    res = st.session_state.prediction_data
    
    st.write("---")
    st.header(f"Результат розпізнавання: {res['label']}")
    st.write(f"Впевненість моделі: **{res['confidence']:.2f}%**")
    
    # якщо це був малюнок, показує, як його обробила система
    if 'debug_img' in res:
        st.image(res['debug_img'], width=100, caption="Як модель бачить ваш малюнок")
    
    st.subheader("Розподіл ймовірностей:")
    st.bar_chart(res['chart'])
    
    # кнопка очищення результатів
    if st.button("Очистити результат"):
        st.session_state.prediction_data = None
        st.rerun()

# футер
st.markdown("---")
st.caption("Використані технології: TensorFlow, Keras, Streamlit, OpenCV.")