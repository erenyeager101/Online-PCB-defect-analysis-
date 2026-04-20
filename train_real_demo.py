import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

def train():
    # 1. Load data from static folder
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    files = glob.glob(os.path.join(static_dir, '*.*'))
    
    images = []
    labels = []
    
    for f in files:
        ext = f.lower().split('.')[-1]
        name = os.path.basename(f).lower()
        if ext not in ['jpg', 'jpeg', 'png']:
            continue
            
        if 'bad' in name:
            label = 1.0
        elif 'good' in name:
            label = 0.0
        else:
            continue
            
        try:
            img = load_img(f, target_size=(224, 224))
            arr = img_to_array(img) / 255.0
            images.append(arr)
            labels.append(label)
        except Exception as e:
            pass

    X = np.array(images)
    y = np.array(labels)
    print(f"Loaded {len(X)} images (Bad=1, Good=0)")

    # 2. Build MobileNetV2
    base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base.trainable = False
    
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # 3. Train
    print("Training quickly on available images...")
    model.fit(X, y, epochs=10, batch_size=16)

    # 4. Save
    model.save(r'D:\Linux_projects\pcb\Online-PCB-defect-analysis-\models\model.h5')
    print("Real demo model saved over models/model.h5")

if __name__ == '__main__':
    train()
