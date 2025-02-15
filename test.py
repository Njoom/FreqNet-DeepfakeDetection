import sys
import time
import os
import csv
import torch
from util import Logger, printSet
from validate import validate
from networks.freqnet import freqnet
from options.test_options import TestOptions
import numpy as np
import random


DetectionTests = {
                'ForenSynths': { 'dataroot'   : '/opt/data/private/DeepfakeDetection/ForenSynths/',
                                 'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
                                 'no_crop'    : True,
                               },

           'GANGen-Detection': { 'dataroot'   : '/opt/data/private/DeepfakeDetection/GANGen-Detection/',
                                 'no_resize'  : True,
                                 'no_crop'    : True,
                               },

                 }


opt = TestOptions().parse(print_options=False)
vals = ['0_real', '1_fake']
#dataroot = '/content/drive/MyDrive/CelebA_Test_FreqNetPaper/test'
Testopt = TestOptions().parse(print_options=False)
Testdataroot = os.path.join(opt.dataroot, 'test')

print(f'Model_path {opt.model_path}')
model = freqnet(num_classes=1)

model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=True)
model.cuda()
model.eval()
accs = [];aps = []
print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
"""
#train.py copy

 for v_id, val in enumerate(vals):
            Testopt.dataroot = Testdataroot
            class_directory = os.path.join(Testopt.dataroot, val)  # Construct the full path for the class
            print("Class directory:", class_directory)  # Debugging line
            Testopt.classes = os.listdir(class_directory) 
#original copy
for v_id, val in enumerate(os.listdir(dataroot)):
        opt.dataroot = '{}/{}'.format(dataroot, val)
        opt.classes  = '' #os.listdir(opt.dataroot) if multiclass[v_id] else ['']
"""

for v_id, val in enumerate(vals):
  Testopt.dataroot = Testdataroot
  class_directory = os.path.join(Testopt.dataroot, val)  
  Testopt.classes  = os.listdir(class_directory)
  opt.no_resize = False
  opt.no_crop   = True
  acc, ap, _, _, _, _ = validate(model, opt)
  accs.append(acc);aps.append(ap)
  print("({} {:12}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 

