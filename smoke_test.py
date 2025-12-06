"""Quick smoke test: loads model.h5 and runs prediction on a sample image.
Usage: python smoke_test.py <path-to-image>
"""
import sys
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.h5')

if not os.path.exists(MODEL_PATH):
    print('models/model.h5 not found. Please place your trained model at models/model.h5')
    sys.exit(1)

if len(sys.argv) < 2:
    print('Usage: python smoke_test.py <path-to-image>')
    sys.exit(1)

img_path = sys.argv[1]
if not os.path.exists(img_path):
    print('Image not found:', img_path)
    sys.exit(1)

print('Loading model...')
model = load_model(MODEL_PATH)
print('Model loaded.')

img = image.load_img(img_path, target_size=(224, 224))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor = img_tensor / 255.0

pred = model.predict(img_tensor)
print('Raw prediction:', pred)
labels = np.array(pred)
labels[labels >= 0.6] = 1
labels[labels < 0.6] = 0
print('Thresholded:', labels)
if labels[0][0] == 1:
    print('Final: Bad')
else:
    print('Final: Good')
