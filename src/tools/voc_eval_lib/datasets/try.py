import torch
import os
annopath1 = os.path.join('VOC2007', 'Annotations', '{}.xml')
filename = '0001'
print(annopath1.format(filename))
