import os
import app

app.get_model()
print('model is', type(app.model))
print('has attr output:', hasattr(app.model, 'output'))
print('has attr outputs:', hasattr(app.model, 'outputs'))
print('model.outputs:', getattr(app.model, 'outputs', None))

last = app._find_last_conv_layer(app.model)
print('last conv layer name:', last)
try:
    layer = app.model.get_layer(last)
    print('target layer type:', type(layer))
    print('has attr output:', hasattr(layer, 'output'))
    print('has attr outputs:', hasattr(layer, 'outputs'))
    print('layer.outputs:', getattr(layer, 'outputs', None))
except Exception as e:
    print('get_layer error:', e)
