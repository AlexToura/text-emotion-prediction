import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


def train_test(train_csv, test_csv, predictions_txt):
    # Start preprocessing timer
    start_preprocess = time.time()

    # Load and preprocess training data
    train_data = pd.read_csv(train_csv)
    train_texts = train_data['text']
    train_labels = train_data['emotion']

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_texts)
    X_train = tokenizer.texts_to_sequences(train_texts)
    X_train = pad_sequences(X_train, padding='post')

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_labels)
    y_train = to_categorical(y_train)

    # Preprocessing timer ends
    end_preprocess = time.time()

    # CNN Model
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=50, input_length=len(X_train[0])))
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(len(encoder.classes_), activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Start training timer
    start_train = time.time()

    # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=2)

    # Training timer ends
    end_train = time.time()

    # Load and preprocess test data
    test_data = pd.read_csv(test_csv)
    X_test = tokenizer.texts_to_sequences(test_data['text'])
    X_test = pad_sequences(X_test, padding='post', maxlen=len(X_train[0]))

    # Start testing timer
    start_test = time.time()

    # Predicting on the test set
    y_test_pred = model.predict(X_test)
    y_test_pred = np.argmax(y_test_pred, axis=1)
    y_test_pred = encoder.inverse_transform(y_test_pred)

    # Testing timer ends
    end_test = time.time()

    # Writing predictions to file
    with open(predictions_txt, 'w') as f:
        for item in y_test_pred:
            f.write("%s\n" % item)

    # Print time taken for each step
    print(f"Preprocessing Time: {end_preprocess - start_preprocess} seconds")
    print(f"Training Time: {end_train - start_train} seconds")
    print(f"Testing Time: {end_test - start_test} seconds")


# Example usage
train_csv = "C:/Users/kakis/Downloads/train_emotion.csv"
test_csv = "C:/Users/kakis/Downloads/test_emotion.csv"
train_test(train_csv, test_csv, 'predictions.txt')
