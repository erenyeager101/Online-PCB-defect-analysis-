import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import app

def test_saliency_blur(img_path):
    app.get_model()
    model = app.model
    # Load image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x_in = np.expand_dims(x / 255.0, axis=0)
    x_tf = tf.convert_to_tensor(x_in)
    
    with tf.GradientTape() as tape:
        tape.watch(x_tf)
        preds = model(x_tf, training=False)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        # Use predicted class score
        score = preds[:, 0] if tf.rank(preds) == 2 else preds
    
    grads = tape.gradient(score, x_tf)
    sal = tf.math.reduce_max(tf.math.abs(grads), axis=-1)[0].numpy()
    
    # Blur heavily
    sal_blurred = cv2.GaussianBlur(sal, (21, 21), 0)
    sal_blurred = sal_blurred - np.min(sal_blurred)
    sal_blurred = sal_blurred / (np.max(sal_blurred) + 1e-8)
    
    # Colorize mapping
    if app._HAS_MPL:
        from matplotlib import cm
        color = cm.get_cmap('jet')(sal_blurred)[..., :3]
    else:
        color = np.stack([sal_blurred, np.zeros_like(sal_blurred), 1 - sal_blurred], axis=-1)
        
    alpha = 0.5
    base = (x_in[0]).copy()
    overlay = (1 - alpha) * base + alpha * color
    overlay = np.clip(overlay, 0, 1) * 255.0
    overlay = overlay.astype('uint8')
    
    from PIL import Image
    Image.fromarray(overlay).save('test_sal_blur.jpg')
    print("Saved test_sal_blur.jpg")

test_saliency_blur(r"D:\Linux_projects\pcb\Online-PCB-defect-analysis-\static\goodk.jpeg")
