from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Set the paths for the training and validation data directories
train_dir = '/Users/apple/Downloads/HeartDiseasePredicitonSystem-master/XrayDataSet/train'
val_dir = '/Users/apple/Downloads/HeartDiseasePredicitonSystem-master/XrayDataSet/validation'

# Set the input image dimensions
img_width, img_height = 224, 224

# Set the batch size for training and validation
batch_size = 32

# Set the number of training and validation samples
num_train_samples = 332
num_val_samples = 87

# Define the data generator for training data
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# Define the data generator for validation data
val_datagen = ImageDataGenerator(rescale=1./255)

# Set the training data
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

# Set the validation data
validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

# Train the model
history = model.fit_generator(
        train_generator,
        steps_per_epoch=num_train_samples // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=num_val_samples // batch_size)

# Save the model
model.save('xray_classifier.h5')
