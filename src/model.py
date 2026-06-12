import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns

dataset_path = 'src/trainingSet/trainingSet' # шлях до папки з підпапками 0-9

# параметри зображень та навчання
IMG_SIZE = (28, 28)
BATCH_SIZE = 32
EPOCHS = 10

full_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    shuffle=True # перемішуємо перед поділом
)

# отримуємо загальну кількість батчів
DATASET_SIZE = len(full_ds)

train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.15 * DATASET_SIZE)
test_size = DATASET_SIZE - train_size - val_size

# поділ на вибірки
train_ds = full_ds.take(train_size)
remaining = full_ds.skip(train_size)
val_ds = remaining.take(val_size)
test_ds = remaining.skip(val_size)

print(f"Батчів на навчання: {train_size}")
print(f"Батчів на валідацію: {val_size}")
print(f"Батчів на тест: {test_size}")

data_augmentation = tf.keras.Sequential([
  layers.RandomRotation(0.1), # випадковий поворот на 10%
  layers.RandomTranslation(0.1, 0.1), # випадковий зсув
  layers.RandomZoom(0.1), # випадкове збільшення
])

# кешування для швидкості
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# створення архітектури CNN
model = models.Sequential([
    # нормалізація даних (з 0-255 у 0-1)
    layers.Input(shape=(28, 28, 1)),
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(28, 28, 1)),
    
    # перший згортковий шар
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # другий згортковий шар
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # перетворення у вектор та повнозв'язні шари
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2), # для запобігання перенавчанню
    layers.Dense(10, activation='softmax') # 10 класів (цифри 0-9)
])

# компіляція моделі
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# навчання моделі
print("\nПочаток навчання...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# збереження моделі для веб-застосунку
model.save('model/mnist_model.h5')
print("\nМодель збережена як mnist_model.h5")

# візуалізація графіків навчання
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Точність навчання')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Втрати навчання')
plt.legend()
plt.show()

# оцінка якості та Confusion Matrix
print("\nГенерація звіту класифікації...")
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')
accuracy = accuracy_score(y_true, y_pred)

print("\n--- Основні метрики моделі ---")
print(f"Accuracy (Точність загальна): {accuracy:.4f}")
print(f"Precision (Точність прогнозу): {precision:.4f}")
print(f"Recall (Повнота):             {recall:.4f}")
print(f"F1-Score (Збалансована):      {f1:.4f}")
print("------------------------------\n")

print("Детальний звіт по кожному класу:")
print(classification_report(y_true, y_pred))

# побудова Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Прогноз')
plt.ylabel('Реальність')
plt.title('Матриця помилок')
plt.show()

# перевірка на випадкових зображеннях
def plot_actual_vs_predicted(dataset, model, n=5):
    random_ds = dataset.shuffle(buffer_size=1000)
    plt.figure(figsize=(15, 4))

    for images, labels in random_ds.take(1):
        predictions = model.predict(images, verbose=0)

        batch_size = images.shape[0]
        random_indices = np.random.choice(batch_size, n, replace=False)

        for i, idx in enumerate(random_indices):
            plt.subplot(1, n, i+1)
            img = images[idx].numpy().astype("uint8").squeeze()
            plt.imshow(img, cmap='gray')  
                      
            pred_label = np.argmax(predictions[idx])
            true_label = labels[idx].numpy()
            
            color = 'green' if pred_label == true_label else 'red'
            plt.title(f"Пр: {pred_label} | Реал: {true_label}", color=color)
            plt.axis('off')
    plt.show()

print("\nДемонстрація прогнозів:")
plot_actual_vs_predicted(test_ds, model)