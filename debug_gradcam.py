import os
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import app

p = r"D:\Linux_projects\pcb\Online-PCB-defect-analysis-\static\goodk.jpeg"
app.get_model()
print('model type:', type(app.model))
last = app._find_last_conv_layer(app.model)
print('last conv layer name:', last)
layer = app.model.get_layer(last)
print('layer type:', type(layer))
print('layer has output:', hasattr(layer, 'output'))
print('layer has outputs:', hasattr(layer, 'outputs'))
print('model has outputs:', getattr(app.model, 'outputs', None))

img = image.load_img(p, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)/255.0
x_tf = tf.convert_to_tensor(x, dtype=tf.float32)

try:
    grad_model = tf.keras.models.Model(inputs=app.model.inputs, outputs=[layer.output, app.model.outputs[0]])
    print('built grad_model')
    # Compute conv activations and predictions inside the same tape scope
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(x_tf)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if tf.rank(preds) == 2:
            score = preds[:, 0]
        else:
            score = preds
    grads = tape.gradient(score, conv_outputs)
    print('grads:', grads)
    if grads is not None:
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        print('pooled_grads shape:', getattr(pooled_grads, 'shape', None))
    else:
        print('grads is None')
except Exception as e:
    print('error building/using grad_model:', e)
