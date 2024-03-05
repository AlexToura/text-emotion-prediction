import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np

# Load the dataset
file_path = "C:/Users/kakis/Downloads/train_emotion.csv"
data = pd.read_csv(file_path)

# Splitting the dataset into training and validation sets
train_data, validation_data, train_labels, validation_labels = train_test_split(
    data['text'], data['emotion'], test_size=0.2, random_state=42)

# Tokenize text data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data)
X_train = tokenizer.texts_to_sequences(train_data)
X_val = tokenizer.texts_to_sequences(validation_data)

# Pad sequences to ensure uniform input size
maxlen = 100  # You can adjust this
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)

# Encode labels
encoder = LabelEncoder()
encoder.fit(train_labels)
y_train = encoder.transform(train_labels)
y_val = encoder.transform(validation_labels)
num_classes = len(encoder.classes_)
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)

# CNN architecture
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=50, input_length=maxlen))
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=4, batch_size=64, verbose=2, validation_data=(X_val, y_val))
# Predicting on the validation set
y_val_pred_probs = model.predict(X_val)
y_val_pred = np.argmax(y_val_pred_probs, axis=1)

# Transforming the encoded labels back to original labels
y_val_true = np.argmax(y_val, axis=1)

# Generate a classification report
report = classification_report(y_val_true, y_val_pred, target_names=encoder.classes_)
print(report)
