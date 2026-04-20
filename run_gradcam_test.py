import os
from app import grad_cam, predict_label_and_conf

p = r"D:\Linux_projects\pcb\Online-PCB-defect-analysis-\static\goodk.jpeg"
print('Predicting...')
label, confidence, prob = predict_label_and_conf(p)
print('Prediction:', label, confidence, prob)
print('Running grad_cam...')
cam = grad_cam(p)
print('CAM path:', cam)
