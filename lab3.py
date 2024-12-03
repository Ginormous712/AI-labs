import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import string

# Введемо набір тексту для тренування
text = open('text.txt', 'r', encoding='utf-8').read().lower()

# Очистимо текст
text = text.translate(str.maketrans('', '', string.punctuation))  # видалимо пунктуацію

# Створимо словник символів
chars = sorted(set(text))  # всі унікальні символи в тексті
char_to_index = {char: index for index, char in enumerate(chars)}  # словник символ -> індекс
index_to_char = {index: char for index, char in enumerate(chars)}  # індекс -> символ

# Перетворимо текст в числові значення
input_text = [char_to_index[char] for char in text]

# Підготовка до моделі: розділяємо на вхідні та цільові дані
sequence_length = 100  # довжина кожної послідовності
sequences = []
next_chars = []

for i in range(len(input_text) - sequence_length):
    sequences.append(input_text[i:i+sequence_length])
    next_chars.append(input_text[i+sequence_length])

X = np.array(sequences)
y = np.array(next_chars)

# Створюємо модель
model = tf.keras.Sequential([
    layers.Embedding(input_dim=len(chars), output_dim=128, input_length=sequence_length),
    layers.LSTM(128, return_sequences=False),
    layers.Dense(len(chars), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.summary()

# Навчання моделі
model.fit(X, y, epochs=10, batch_size=64)

def generate_text(model, seed_text, length=100):
    input_sequence = [char_to_index[char] for char in seed_text]
    generated_text = seed_text

    for _ in range(length):
        input_array = np.array([input_sequence])
        predicted_probs = model.predict(input_array, verbose=0)
        predicted_index = np.argmax(predicted_probs)
        predicted_char = index_to_char[predicted_index]
        generated_text += predicted_char
        input_sequence.append(predicted_index)
        input_sequence = input_sequence[1:]

    return generated_text

# Генерація тексту
seed_text = "this is"
generated_text = generate_text(model, seed_text, length=200)
print(generated_text)

