import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.src.utils.module_utils import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU is available and memory growth is set.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Running on CPU.")


# Function to load and resize images from folders named after each celebrity
def load_and_resize_images(folder, size=(200, 200)):
    images = []
    labels = []
    label_dict = {}
    label_index = 0
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            label_dict[label_index] = subfolder
            for filename in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, size)
                    images.append(img)
                    labels.append(label_index)
            label_index += 1
    return images, labels, label_dict


# Load labeled training data
labeled_images, labels, label_dict = load_and_resize_images("labeledData")

# Split data into training, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(labeled_images, labels, test_size=0.2, random_state=42,
                                                  stratify=labels)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp)

# Process training data
X_train = np.array(X_train) / 255.0
X_val = np.array(X_val) / 255.0
X_test = np.array(X_test) / 255.0

y_train = to_categorical(y_train, num_classes=len(label_dict))
y_val = to_categorical(y_val, num_classes=len(label_dict))
y_test = to_categorical(y_test, num_classes=len(label_dict))

# Define the improved model
model = Sequential([
    Conv2D(256, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(512, activation='relu'),
    BatchNormalization(),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(len(label_dict), activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define a checkpoint callback to save the best model
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Train the model
model.fit(X_train, y_train, batch_size=4, epochs=100, validation_data=(X_val, y_val), callbacks=[checkpoint])

# Load the best model before evaluating on the test set
model.load_weights('best_model.keras')

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')


# Function to generate augmented images
def generate_augmented_images(images, labels, num_augmented):
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2,
                                 horizontal_flip=True, fill_mode='nearest')

    augmented_images = []
    augmented_labels = []

    for img, label in zip(images, labels):
        img = np.expand_dims(img, axis=0)
        augmented_iter = datagen.flow(img, batch_size=1)
        for _ in range(num_augmented):
            augmented_img = next(augmented_iter)[0].astype(np.uint8)
            augmented_images.append(augmented_img)
            augmented_labels.append(label)

    return np.array(augmented_images), np.array(augmented_labels)


# Parameters
num_augmented = 5  # Number of augmented images per original image

# Generate augmented images
augmented_images, augmented_labels = generate_augmented_images(X_test, y_test, num_augmented)

# Combine the original and augmented test images
X_test_augmented = np.concatenate((X_test, augmented_images), axis=0)
y_test_augmented = np.concatenate((y_test, augmented_labels), axis=0)

# Evaluate the model on the augmented test set
test_loss_augmented, test_acc_augmented = model.evaluate(X_test_augmented, y_test_augmented)
print(f'Test accuracy on augmented images: {test_acc_augmented:.4f}')


# Main function for live face recognition
def main():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_img = frame[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, (200, 200))
                face_img = np.expand_dims(face_img, axis=0) / 255.0

                # Prediction
                prediction = model.predict(face_img)
                label_index = np.argmax(prediction)
                confidence = prediction[0][label_index]

                # Annotate the frame with the label
                label = label_dict[label_index]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f'{label} ({confidence:.2f})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 0, 0), 2)

            cv2.imshow('Camera', frame)
        key = cv2.waitKey(1)
        if key == 27:  # Press 'ESC' to exit
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
