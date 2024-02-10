from utils import *
import glob,tifffile
from tqdm.notebook import tqdm
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
global_path = "/kaggle/input/solarpaneldetect-solafune/"
raw_image = []
mask_image = []
for i in range( 2066 ):
    data = tifffile.imread(f'{global_path}train/s2_image/train_s2_image_{i}.tif')
    mask = tifffile.imread(f'{global_path}train/mask/train_mask_{i}.tif')
    raw_image.append(data)
    mask_image.append(mask)
print(raw_image[2].shape, mask_image[2].shape)

import tensorflow as tf
print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE
from tensorflow.keras import  mixed_precision
tf.compat.v1.enable_eager_execution()
strategy=None
TPU=True
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print("Running on TPU:", tpu.master())

    # set half precision policy
    # Mixed precision is the use of both 16-bit and 32-bit floating-point types
    # in a model during training to make it run faster and use less memory.
    mixed_precision.set_global_policy('mixed_bfloat16' if TPU else 'float32')
    print(f'Compute dtype: {mixed_precision.global_policy().compute_dtype}')
    print(f'Variable dtype: {mixed_precision.global_policy().variable_dtype}')

    # enable XLA optmizations
    tf.config.optimizer.set_jit(True)
except ValueError:
    strategy = tf.distribute.MirroredStrategy()
    mixed_precision.set_global_policy('mixed_float16')

with strategy.scope():
    model = SolarPanelDetectionDeepLearn( ( raw_image, mask_image ), seed =11 )
    model.fit()

try:
    count = 1
    for x in model.models:
        x.compile( loss= tf.keras.losses.BinaryCrossentropy() )
        x.save( f'save_model_{count}.h5' )
        count +=1
except:
    count = 1
    for x in model.models:
        x.save_weights( f'save_model_weights_{count}.h5' )
        count +=1


import zipfile
import os
from tqdm import tqdm
paths = ["invert_Unet_FE_RI_m3.zip"]
paths.append( paths[0].replace( '.zip', '_low_bloom.zip' ) )
evalution = glob.glob( global_path + "evaluation/*.tif" )
count= 0
test = [ tifffile.imread(x) for x in evalution ]
    
pred_masks = model.predictV3( test )
# with tf.device('/CPU:0'):
for path, mask in zip(paths, pred_masks):
    
    with zipfile.ZipFile(path, mode='w') as file:
        index = 0
        for x in tqdm(evalution):
            output_path = 'evaluation_mask_'+ x.split('_')[-1].split('.')[0] + ".tif"
            if 'evaluation_s2_image_1167.tif' in x:
                tifffile.imwrite( output_path, np.zeros((test[index].shape[0], test[index].shape[1]), dtype = np.uint8))
            else:
                tifffile.imwrite( output_path, mask[index].squeeze().astype(np.uint8))
            file.write(output_path)
            os.remove( output_path )
            index +=1


plt.show()

