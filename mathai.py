import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense
import numpy as np
import json
import time
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Przygotowanie danych
X = np.array([i for i in range(10)], dtype=np.float32)
y = np.array([0 if i % 2 == 0 else 1 for i in range(10)], dtype=np.float32)

# Definicja modelu
model = Sequential([
    Dense(10, input_shape=(1,), activation='relu'),
    Dense(1, activation='sigmoid')
])

# Kompilacja modelu
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Trening modelu
model.fit(X, y, epochs=100, verbose=1)

# Zapisywanie modelu do pliku JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Zapisywanie wag modelu do pliku .weights.h5
model.save_weights("model.weights.h5")

print("Model has been saved to 'model.json' and 'model.weights.h5'.")

# Wczytywanie modelu z pliku JSON
with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.weights.h5")

# Kompilacja wczytanego modelu
loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Loaded model from file")

# Funkcja do kontynuowania treningu przez określony czas (w minutach)
def continue_training(minutes):
    start_time = time.time()
    end_time = start_time + minutes * 60  # Przekształcenie minut na sekundy

    while time.time() < end_time:
        loaded_model.fit(X, y, epochs=1, verbose=1)

    # Ponowne zapisanie modelu po treningu
    loaded_model.save_weights("model.weights.h5")

# Funkcja do testowania modelu
def test_model(expression):
    try:
        result = eval(expression)
    except:
        return "Incorrect operation (an error may have occurred)"
    
#    prediction = loaded_model.predict(np.array([result], dtype=np.float32))
#    predicted_class = "Parzysta" if prediction < 0.5 else "Nieparzysta"
    
    return f"Result: {result}"

# Strona główna z formularzami do testowania i kontynuowania treningu
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint do testowania modelu
@app.route('/test', methods=['POST'])
def test():
    expression = request.form['expression']
    result = test_model(expression)
    return jsonify(result=result)

# Endpoint do kontynuowania treningu
@app.route('/continue', methods=['POST'])
def continue_train():
    minutes = int(request.form['minutes'])
    continue_training(minutes)
    return jsonify(result=f"Model trained for {minutes} minutes.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)