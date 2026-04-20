import os
import tensorflow as tf
from tensorflow.keras import layers, models

def create_model():
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model.save('models/model.h5')
    print("Dummy model saved to models/model.h5")

if __name__ == '__main__':
    create_model()
