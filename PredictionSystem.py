import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# Define the data generators for training, validation, and testing
train_dir = '../MajorProject/DataSet/train'
val_dir = '../MajorProject/DataSet/val'
test_dir = '../MajorProject/DataSet/test'
img_size = (224, 224)
batch_size = 32

def sample_image(image_path):
    img = load_img(image_path)
    print((img_to_array(img)).shape)
    plt.imshow(img)
    plt.show()

print("NORMAL X-ray")
image_path = "/Users/apple/Desktop/MajorProject/DataSet/val/NORMAL/NORMAL2-IM-1430-0001.jpeg"
sample_image(image_path)
print("COVID19 X-ray")
image_path = "/Users/apple/Desktop/MajorProject/DataSet/val/COVID19/COVID19(566).jpg"
sample_image(image_path)
print("Pneumonia X-ray")
image_path = "/Users/apple/Desktop/MajorProject/DataSet/val/PNEUMONIA/person1949_bacteria_4880.jpeg"
sample_image(image_path)
print("TUBERCULOSIS X-ray")
image_path = "/Users/apple/Desktop/MajorProject/DataSet/val/TURBERCULOSIS/Tuberculosis-651.png"
sample_image(image_path)
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=img_size,
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(val_dir,
                                                target_size=img_size,
                                                batch_size=batch_size,
                                                class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                   target_size=img_size,
                                                   batch_size=batch_size,
                                                   class_mode='categorical')

# Define the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])

# Train the model
epochs = 15
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // batch_size,
                    epochs=epochs,
                    validation_data=val_generator,
                    validation_steps=val_generator.samples // batch_size)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print('Test accuracy:', test_acc)

# Make predictions on new x-ray images
import numpy as np
from tensorflow.keras.preprocessing import image

img_path = '/Users/apple/Desktop/MajorProject/DataSet/val/NORMAL/NORMAL2-IM-1431-0001.jpeg'
img = image.load_img(img_path, target_size=img_size)
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

preds = model.predict(img_tensor)
print('Predictions:', preds)

import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


from sklearn.metrics import classification_report, confusion_matrix

# Predict on the test set
y_pred = model.predict(test_generator)

# Get the true labels
y_true = test_generator.classes

# Get the class labels
class_labels = list(test_generator.class_indices.keys())

# Generate and print classification report
report = classification_report(y_true, np.argmax(y_pred, axis=1), target_names=class_labels)
print(report)

# Generate and plot confusion matrix
conf_mat = confusion_matrix(y_true, np.argmax(y_pred, axis=1))
plt.matshow(conf_mat, cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.xticks(np.arange(len(class_labels)), class_labels)
plt.yticks(np.arange(len(class_labels)), class_labels)
plt.show()

from keras.models import load_model

model.save('model.h5')

print("Accuracy:", history.history['accuracy'][-1])
CNNAccuracy = test_acc
AccuracyCNN = history.history['accuracy'][-1]
print(CNNAccuracy)
print(AccuracyCNN)

import tensorflow as tf

# Load data
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    '/Users/apple/Desktop/MajorProject/DataSet/train',
    image_size=(224, 224),
    batch_size=32
)
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    '/Users/apple/Desktop/MajorProject/DataSet/test',
    image_size=(224, 224),
    batch_size=32
)

# Create model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# Train model
history = model.fit(train_data, epochs=10, validation_data=test_data)

# Print accuracy
print("Accuracy:", history.history['accuracy'][-1])

# Evaluate model
test_loss, test_acc = model.evaluate(test_data)
print("Testing Accuracy:", test_acc)

import matplotlib.pyplot as plt

# Define the data for the two models
FNN_train_acc = history.history['accuracy'][-1] 
FNN_test_acc = test_acc
CNN_train_acc = 0.9566253423690796
CNN_test_acc = 0.9166666865348816

print("FNN Training Accuracy :", FNN_train_acc * 100)
print("FNN Testing Accuracy :", FNN_test_acc * 100)
print("CNN Training Accuracy :", CNN_train_acc * 100)
print("FNN Testing Accuracy :", CNN_test_acc * 100)
# Define the labels and positions for the bars
model_labels = ['Model FNN', 'Model CNN']
y = []
bar_positions_train = [0, 1.5]
bar_positions_test = [0.5, 2]

# Define the heights of the bars
train_accs = [FNN_train_acc * 100, CNN_train_acc * 100]
test_accs = [FNN_test_acc * 100, CNN_test_acc * 100]

# Create the figure and axis objects
fig, ax = plt.subplots()

# Create the train accuracy bars
ax.bar(bar_positions_train, train_accs, width=0.5, color='blue', alpha=0.5, label='Train')

# Create the test accuracy bars
ax.bar(bar_positions_test, test_accs, width=0.5, color='green', alpha=0.5, label='Test')

# Add labels and titles to the graph
ax.set_xlabel('Models')
ax.set_ylabel('Accuracy')
ax.set_title('Comparison of Model Accuracies')
ax.set_xticks([0.25, 1.75])
ax.set_xticklabels(model_labels)
ax.legend()
# Display the graph
plt.show()
