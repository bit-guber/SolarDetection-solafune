
import numpy as np

bands_index = { 
    'B01':0, 
    'B02':1, 'B03':2, 'B04':3, 
    'B05':4, 'B06':5, 'B07':6, 
    'B08':7, 
    'B8A':8, 
    'B09':9, 
    'B11':10, 'B12':11
 }
class BandsGather:
    def __init__(self, data):
        self.data = data
    def __getitem__(self, id):
        return self.data[ :, bands_index[ id ] ]

eps = 1e-7
def ProperDivide( numerator:np.ndarray, denominator:np.ndarray ):
    result = numerator / denominator
    if np.sum(denominator == 0)>0:
        result[denominator == 0] = 0
        return result
    else:
        return result
    
def get_NDVI( data:BandsGather ):
    numerator = data[ 'B08' ] - data[ 'B04' ]
    denominator = data['B08'] + data['B04']
    return ProperDivide( numerator, denominator )
    
def get_ARI( data:BandsGather ):
    numerator = 1 / data['B03']
    denominator =  1/ data['B05']
    return ProperDivide( numerator, denominator )

def get_mARI( data:BandsGather ):
    return ( ( ProperDivide( 1, data['B03'] ) - ProperDivide( 1, data['B05'] ) ) * data['B07'] )

def get_ARVI( data:BandsGather, y = 0.106 ):
    numerator = data['B8A'] - data['B04'] - y * ( data['B04'] - data['B02'] )
    denominator = data['B8A'] + data['B04'] - y * ( data['B04'] - data['B02'] )
    return ProperDivide( numerator, denominator )

def get_CHL_REDEDGE( data:BandsGather ):
    numerator = data['B07']
    denominator = data['B05']
    return ProperDivide(numerator, denominator) - 1  

def get_REPO( data:BandsGather ):
    numerator = ( ( data['B04'] + data['B07'] ) / 2 ) - data['B05'] 
    denominator = data['B06'] - data['B05']
    return 700 + 40 * ProperDivide( numerator, denominator )

def get_EVI( data:BandsGather ):
    numerator = data['B08'] - data['B04']
    denominator = data['B08'] + 6 * data['B06'] - 7.5 * data['B02']
    return 2.5 * ( ProperDivide( numerator, denominator ) +1 )

def get_MCARI( data:BandsGather ):
    numerator = data['B05']
    denominator = data['B04']
    return ( ( data['B05'] - data['B04'] ) - 0.2 * ( data['B05'] - data['B03'] ) ) * ProperDivide( numerator, denominator )

def get_MSI( data:BandsGather ):
    numerator = data['B11']
    denominator = data['B08']
    return ProperDivide( numerator, denominator )

def get_NDMI( data:BandsGather ):
    numerator = data['B08'] - data['B11']
    denominator = data['B08'] + data['B11']
    return ProperDivide( numerator, denominator )

def get_NDWI( data:BandsGather ):
    numerator = data['B03'] - data['B08']
    denominator = data['B03'] + data['B08']
    return ProperDivide( numerator, denominator )


def get_NBR( data:BandsGather ):
    numerator = data['B08'] - data['B12']
    denominator = data['B08'] + data['B12']
    return ProperDivide( numerator, denominator )

def get_NDCI( data:BandsGather ):
    numerator = data['B05'] - data['B04']
    denominator =  data['B05'] + data['B04']
    return ProperDivide( numerator, denominator )


def get_NDSI( data:BandsGather ):
    numerator = data['B03'] - data['B11']
    denominator = data['B03'] + data['B11']
    return ProperDivide( numerator, denominator )

def get_PSSRb( data:BandsGather ):
    return ProperDivide(data['B08'], data['B04'])

def get_SAVI( data:BandsGather, L = 0.428 ):
    numerator = data['B08'] - data['B04']
    denominator = data['B08'] + data['B04'] + L
    return ProperDivide( numerator, denominator) * (1+L)

def get_SIPI( data:BandsGather ):
    numerator = data['B08'] - data[ 'B01']
    denominator = data['B08'] - data['B04']
    return ProperDivide( numerator, denominator )

def get_PSRI( data:BandsGather ):
    numerator = data['B04'] - data['B02']
    denominator = data['B06']
    return ProperDivide( numerator, denominator )

def get_OSI( data:BandsGather ):
    numerator = data['B03'] + data['B04']
    denominator = data['B02']
    return ProperDivide( numerator, denominator )

def get_BSI( data:BandsGather ):
    numerator = ( data['B11'] + data['B04'] ) - ( data['B08'] + data['B02'] )
    denominator = ( data['B11'] + data['B04'] ) + ( data['B08']+data['B02'] )
    return ProperDivide( numerator, denominator )

def get_NDYI( data:BandsGather ):
    numerator = data['B03'] - data['B02']
    denominator = data['B03'] + data['B02']
    return ProperDivide( numerator, denominator )

# 
def get_BNDVI( data:BandsGather ):
    numerator = data['B08'] - data['B02']
    denominator = data['B08'] + data['B02']
    return ProperDivide( numerator, denominator )

def get_GBNDVI( data:BandsGather ):
    numerator = data['B08'] - ( data['B03'] + data['B02']  )
    denominator = data['B08'] + ( data['B03'] + data['B02']  )
    return ProperDivide( numerator, denominator )

def get_GRNDVI( data:BandsGather ):
    numerator = data['B08'] - ( data['B03'] + data['B04']  )
    denominator = data['B08'] + ( data['B03'] + data['B04']  )
    return ProperDivide( numerator, denominator )

def get_NDRE( data:BandsGather ):
    numerator = data['B08'] - data['B05']
    denominator = data['B08'] + data['B05']
    return ProperDivide( numerator, denominator )

def get_RBNDVI( data:BandsGather ):
    numerator = data['B08'] - ( data['B04'] + data['B02'] )
    denominator = data['B08'] + ( data['B04'] + data['B02'] )
    return ProperDivide( numerator, denominator )

def get_SWI( data:BandsGather ):
    numerator = data['B05'] - data['B12']
    denominator = data['B05'] + data['B12']
    return ProperDivide( numerator, denominator )


# geo-university satellite indexes
def get_AVI( data:BandsGather ):
    avi = data['B08'] * ( 1 - data['B04'] ) * ( data['B08'] - data['B04'] )
    # avi_n = np.ones_like(avi)
    # avi_n[avi>=0] = -1
    # return (-np.power( np.abs(avi) , 1/3)) * avi_n
    # return np.power(np.abs(avi), 1/3) 
    return np.power(avi, 1/3) 
def get_SI( data:BandsGather ):
    si = ( 1- data['B02'] ) * ( 1-data['B03'] ) * ( 1 - data['B04'] )
    # si_n = np.ones_like( si )
    # si_n[si>=0] = -1
    # return ( -np.power( np.abs(si) , 1/3) ) * si_n
    # return np.power(np.abs(si), 1/3)
    return np.power(si, 1/3)


def get_NPCRI( data:BandsGather ):
    numerator = data['B04'] - data['B02']
    denominator = data['B04'] + data['B02']
    return ProperDivide( numerator, denominator )

def get_specialEX( data:BandsGather ):
    
    norm1 = ProperDivide( data['B02'] - data['B03'] , data['B03']  + data['B02'] )
    norm2 = ProperDivide( data['B11'] - data['B12'], data['B12'] + data['B11']  )
    return (norm1 + norm2)/2


# feature_functions = [  get_NDVI, get_ARI, get_mARI, get_ARVI, get_CHL_REDEDGE, get_REPO, get_EVI, get_MCARI, 
#                 get_MSI, get_NDMI, get_NDWI, get_NBR, get_NDCI, get_NDSI, get_PSSRb, 
#                 get_SAVI, get_SIPI, get_PSRI, get_OSI, get_BSI, get_NDYI, get_BNDVI, get_GBNDVI, get_GRNDVI, get_NDRE, get_RBNDVI, get_SWI,
#                 get_AVI,  get_SI, get_NPCRI ]

feature_functions = [ get_MSI, get_NDMI, get_NDWI, get_OSI, get_BSI, get_NDYI, get_BNDVI, get_GBNDVI,
                    get_RBNDVI, get_NBR, get_PSRI, get_GRNDVI, get_ARI, get_NDRE, get_SAVI, get_NDVI, 
                    get_mARI, get_ARVI, get_PSSRb, get_MCARI,
                    get_AVI,  get_SI, get_NPCRI,
                    get_CHL_REDEDGE, get_REPO, get_EVI,
                    get_NDCI, get_NDSI,
                    get_SIPI, get_SWI ] 


def features_extraction( data ):
    data = BandsGather( data * 0.0001 )

    
    features = np.concatenate([ x(data).reshape( -1, 1 ) for x in feature_functions ], axis = 1)
    features = np.concatenate( [ data.data, features ], axis = 1 )
    return features

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import gc, glob, os
from tensorflow.keras import  mixed_precision
from sklearn.metrics import mean_absolute_error

AUTO = tf.data.experimental.AUTOTUNE


@tf.function
def f1_score_tf(y_true, y_pred, axis = (1, 2,3), dtype = tf.float32 ):
    
    TP = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
    TP = tf.cast(TP, dtype)
    TP = tf.reduce_sum(TP, axis= axis )

    FP = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
    FP = tf.cast(FP, dtype)
    FP = tf.reduce_sum(FP, axis= axis)

    FN = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
    FN = tf.cast(FN, dtype)
    FN = tf.reduce_sum(FN, axis= axis)

    denominator = TP + 0.5 * ( FP + FN )
    result = tf.math.divide_no_nan( TP, denominator )
    result = result + tf.cast(tf.equal( denominator, 0 ), dtype)
#     print(result.shape, result.dtype, result.device)
    return tf.math.reduce_mean(result)

@tf.function
def find_best_threshold(target, prediction, thresholds, metric = f1_score_tf, dtype = tf.float32):
    target = tf.cast(target, tf.bool)
    prediction = tf.cast(prediction, tf.float32)
    best_threshold = .5
    best_score = metric( target, tf.math.greater_equal(prediction, best_threshold), dtype = dtype )
    for i in thresholds:
        current = metric( target, tf.math.greater_equal(prediction, i ), dtype = tf.float32 )
        if current > best_score :
            best_score =current
            best_threshold = i
    return best_threshold, best_score

class F1_scoreV2(tf.keras.callbacks.Callback):
    def __init__(self, test_dataset):
        super().__init__()
        self.test_X, self.test_Y = test_dataset
        self.global_score = 0
    def on_epoch_end(self, epoch, logs=None): 
        test_logits = self.model.predict(self.test_X, batch_size = 32, verbose = 0  )[-1]
        self.test_Y = self.test_Y[:test_logits.shape[0]]
#         print( 'y_true', self.test_Y.shape, 'y_pred', test_logits.shape )
        thresholds =tf.constant([0., .1, .2, .3, .4, .6 ,.7, .8, .9], dtype = tf.float32)
        best_threshold, best_score = find_best_threshold(self.test_Y, test_logits, thresholds)
        logs['val_f1_score_BT'] = best_threshold
        logs['val_f1_score_BS'] = best_score
        low_bloom = (test_logits>best_threshold.numpy()).astype(np.int8).sum( axis =(1, 2, 3) ) < 5
        
        print(low_bloom.sum(), low_bloom.shape)
        rest_threshold = (1 - best_threshold) /2
        new_threshold = np.ones_like(test_logits, dtype = np.float32) * best_threshold.numpy()
        new_threshold[low_bloom] += rest_threshold
        experiment_bloom = test_logits>new_threshold
        
        logs['low_bloom_BS'] = f1_score_tf(self.test_Y, experiment_bloom)
        
        if self.global_score < best_score:
            self.global_score=best_score
            try:
                self.model.save( f'models/save_model_ep-{epoch}.h5' )
            except:
                self.model.save_weights( f'models/save_model_weights_ep-{epoch}.h5' )


@tf.function
def dice_loss( target, prediction, axis=(1, 2), smooth=1e-5):

    target = tf.cast( target, tf.float32 )
    prediction = tf.cast( prediction, tf.float32 )
    intersection = tf.reduce_sum(prediction * target, axis=axis)
    
    p = tf.reduce_sum(prediction, axis=axis)
    t = tf.reduce_sum(target, axis=axis)
    
    numerator = intersection + smooth
    denominator = t + p + smooth
    loss = ( 2.*numerator ) / denominator

    return 1 - loss

@tf.function
def combined_loss( y_true, y_pred ):
    return tf.keras.metrics.binary_focal_crossentropy( y_true, y_pred, gamma=2.0, axis = (1,2) ) + dice_loss( y_pred, y_true )

class NormalizerV3:
    def transform(self, data):
        data[:, 12] = data[:, 12] * .1 
        data[:, 15] = data[:, 15] * .1
        data[:, 24] = data[:, 24] * .1
        data[:, 28] = data[:, 28] * .1
        
        data[:, 30] = data[:, 30] / 20
        data[:, 35] = data[:, 35] / 5
        
        data[:, 36] = np.clip(data[:, 36], a_max=1000, a_min=500) * .001
        data[:, 37] = np.clip( data[:, 37], a_min=0, a_max= 5) / 5
        data[:, 40] = np.clip( data[:, 40], a_max=5, a_min=0 ) / 5
        
        data[~np.isfinite(data)] = 0
        return data
    def fit(self, data):
        pass
    def fit_tranform( self, data ):
        return self.transform(data)
    
def get_group(label, nGroups = 10):
    group = []
    for x in label:
        coverage = x.sum() / x.size
        if coverage == 0:
            group.append( 0 )
        else:
            group.append( int(coverage * nGroups) + 1 )
    return group

fig, ax = plt.subplots(2, 2)
class SolarPanelDetectionDeepLearn:
    def __init__( self, data, batch_size, n_split, seed = 11 ):
        
        self.models = []
        self.metric = f1_score
        self.normalize = NormalizerV3()#NormalizerScale()#RobustScaler()
        self.thresholds = []
        self.data, self.label = data
        self.seed = seed
        self.image_size = 32
        self.image_scale = 1
        self.batch_size = batch_size 
        self.n_features = self.data[0].shape[-1] + len(feature_functions)
        print( 'preparation' )
        gc.collect()
        self.folds = list(StratifiedKFold( n_splits=n_split, shuffle=True, random_state= self.seed ).split( np.arange(len(self.data)), get_group(self.label) ))
        
    def preprocess(self, image, mask = None):
        image = tf.image.resize( image, ( self.image_size, self.image_size), method='nearest' ).numpy()[None,...]
        if mask is not None:
            mask = tf.image.resize( mask, ( self.image_size, self.image_size), method='nearest' ).numpy()[None,...]
            return image, mask            
        return image
    
    def fit_normalize( self, data ):
        self.normalize.fit(  data  )
    
    def get_valid_set( self, indexes ):
        Xdata = []
        Ydata = []
        for i in indexes:
            X, Y = self.preprocess( self.data[i], np.expand_dims(self.label[i], axis = -1) )

            Xdata.append( X )
            Ydata.append( Y )
        Xdata, Ydata = np.concatenate( Xdata ), np.concatenate( Ydata )
        
        nfeature = Xdata.shape[-1]
        Xdata = Xdata.reshape( -1, nfeature ) 
        Xdata = features_extraction(Xdata)
        Xdata = self.normalize.transform( Xdata )
        
        Xdata = Xdata.reshape( -1, self.image_size, self.image_size, self.n_features  )
        
        if np.isnan(Ydata).sum()>0:
            print('validation data contains nan', np.isnan(Ydata).sum())
        return Xdata, Ydata
    
    def DataAugmentation( self, indexes ):
        Xdata =[]
        Ydata =[]
        for i in indexes:
            X, Y = self.preprocess( self.data[i], np.expand_dims( self.label[i], axis = -1 ) )
            for _ in range(1, 3):
                Xdata.append(np.flip(X, _ ))
                Ydata.append(np.flip(Y, _ ))
            for _ in range(4):
                Xdata.append( np.rot90( X, k=_, axes = (1, 2) ).astype(np.float32) ) 
                Ydata.append( np.rot90( Y, k=_, axes = (1, 2) ).astype(np.float32)) 
            Xdata.append(np.transpose(X, (0, 2, 1, 3) ))
            Ydata.append(np.transpose(Y, (0, 2, 1, 3) ))
            
            Xdata.append(np.rot90(np.transpose(X, (0, 2, 1, 3) ), k =2 ))
            Ydata.append(np.rot90(np.transpose(Y, (0, 2, 1, 3) ), k =2 ))
            
        Xdata, Ydata= np.concatenate( Xdata ), np.concatenate( Ydata )
        
        nfeature = Xdata.shape[-1]
        Xdata = Xdata.reshape( -1, nfeature )
        Xdata = features_extraction(Xdata)
        Xdata = self.normalize.transform( Xdata )
        Xdata = Xdata.reshape( -1, self.image_size, self.image_size, self.n_features )
        
        if np.isnan(Ydata).sum()>0:
            print('tuting data contains nan', np.isnan(Ydata).sum())
        return Xdata, Ydata

    
    def get_model(self):
        n_filters = 64
        window_size = 3

        def convBlock( value, n_filters, activation = 'relu'):
            x = tf.keras.layers.Conv2D( n_filters, window_size, padding='same', activation=activation )( value )
            x = tf.keras.layers.Dense( n_filters, use_bias = False, activation=activation )( x )    
            return x
        
        def LastDenseBlock( x, name, activation = 'relu' ):
            x = tf.keras.layers.Dense( 512, use_bias = True, activation=activation )(x)
            x = tf.keras.layers.Dropout( .8, noise_shape=( 1, 1, 512 ) )(x)
            x = tf.keras.layers.Dense( 1, use_bias = False, activation = 'sigmoid', dtype = 'float32', name = name )(x)
            return x 
        def downscale(x, factor = 1):
            skip = convBlock(x, n_filters // factor)
            x = tf.keras.layers.AveragePooling2D( 2 )(skip)
            return skip, x
        def upscale(x, skip, factor = 1 ):
            x = tf.keras.layers.Conv2DTranspose( x.shape[-1], 2, 2, padding='same', kernel_initializer='ones', trainable = False, data_format = 'channels_last' )(x)
            x = tf.keras.layers.concatenate([x, skip])
            x = convBlock(x, n_filters// factor)
            return x
         # inputs
        inputs = tf.keras.layers.Input(shape=(32, 32, self.n_features )) # 20x20
        
        x = tf.keras.layers.Dense( n_filters, use_bias = False )(inputs)
        x = tf.keras.layers.BatchNormalization(synchronized=True)(x)
        
        s_1, d_1 = downscale( inputs)
        s_2, d_2 = downscale( d_1, 2 )
        s_3, d_3 = downscale( d_2, 4 )
        
        bottleneck = convBlock(d_3, n_filters//8)
    
        u_1 = upscale( bottleneck, s_3, 4 )
        u_2 = upscale( u_1, s_2, 2 )
        u_3 = upscale( u_2, s_1)
        
        last_1 = LastDenseBlock( u_1, 'last_1' )
        last_2 = LastDenseBlock( u_2, 'last_2' )
        last_3 = LastDenseBlock( u_3, 'last_3' )

        model = tf.keras.Model( inputs= inputs, outputs =[last_1, last_2, last_3]  )

        model.compile( optimizer=mixed_precision.LossScaleOptimizer(tf.keras.optimizers.AdamW( 0.0005, weight_decay = 0.0 )), loss = combined_loss,
                      metrics=None, steps_per_execution = 16 )
        return model

    def get_callbacks(self):

        
        earlyStop = tf.keras.callbacks.EarlyStopping(
            monitor='val_last_3_loss',
            patience=12,
            verbose=0,
            mode='min',
            restore_best_weights=False,
            start_from_epoch=0
        )
        lr_schedule = tf.keras.callbacks.ReduceLROnPlateau( monitor='val_loss', factor=0.9, patience=3, mode = 'min', min_lr = 1e-5)

        return [ earlyStop, lr_schedule ]
    
    def restore_best_weight(self, model):
        path = "models/"
        model_paths = glob.glob( f'{path}*.h5' )
        model_paths.sort( key = lambda x : os.path.getctime(x), reverse=True )
        try:
            print( 'best weight', model_paths[0] )
            model.load_weights(model_paths[0], by_name = True)

            for x in model_paths:
                os.remove( x )
        
        except IndexError:
            print( "can't get models or no best weights" )

        return model
    def set_scoreToIndexes(self, indexes, scores):
        indexes = indexes.tolist()
        for i in range( len(indexes) ):
            for _ in range( 4 ):
                try:
                    self.indexScore[indexes[i]].append( scores[(i*4) + _] )
                except:
                    self.indexScore[indexes[i]] = [ scores[(i*4) + _] ]
    def fit( self ):
        tf.keras.utils.set_random_seed(self.seed)
        tf.config.experimental.enable_op_determinism()
#         tf.compat.v1.disable_eager_execution()

        self.avg_score = 0
        self.fit_normalize( features_extraction(np.concatenate([ x.reshape(-1, 12) for x in self.data])) )
        def add_resize( image, label ):
            label = { 
                'last_1': tf.image.resize( label, ( self.image_size//4, self.image_size//4), method='nearest' ), 
                'last_2': tf.image.resize( label, ( self.image_size//2, self.image_size//2), method='nearest' ),
                'last_3': label }
            
            return image, label

        for train, valid  in self.folds :
            train_data, train_label = self.DataAugmentation(train)

            gc.collect()
            print( train_data.shape )
            train_batches = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices( train_data ), tf.data.Dataset.from_tensor_slices( train_label )))
            train_batches = train_batches.shuffle(  train_batches.cardinality(), seed=self.seed, reshuffle_each_iteration=True).map( add_resize, num_parallel_calls=AUTO, deterministic=True ).batch( self.batch_size, drop_remainder = True, num_parallel_calls=AUTO, deterministic=True ).prefetch(16)
            gc.collect()
            valid_data, valid_label = self.get_valid_set( valid )
            valid_batches = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices( valid_data ), tf.data.Dataset.from_tensor_slices( valid_label) )).map( add_resize, num_parallel_calls=AUTO, deterministic=True  ).batch( self.batch_size, drop_remainder = True, num_parallel_calls=AUTO, deterministic=True ).prefetch(16)

            gc.collect()

        
            model = self.get_model()
            model.fit( train_batches, batch_size=self.batch_size, epochs=900, validation_data=valid_batches, 
                verbose= 2, callbacks=[*self.get_callbacks(), F1_scoreV2((valid_data, valid_label))] , workers=-1, use_multiprocessing=True )
        
            model = self.restore_best_weight(model)
            pred_y = model.predict( valid_data, verbose = 0, batch_size=self.batch_size)[-1]
            
            limit = 100
            best_threshold , _ = find_best_threshold(valid_label, pred_y, np.arange( 1, limit , dtype = np.float32) / limit )
            best_threshold = best_threshold.numpy()
            valid_label = valid_label[:pred_y.shape[0]]
            score = []
            x = []
            y = []
            for pred, true in zip(pred_y, valid_label):
                if true.astype(np.int8).sum() == 0 and (pred>best_threshold).astype(np.int8).sum() == 0:
                    score.append(1)
                else:
                    score.append(self.metric( (true>best_threshold).astype(np.int8).reshape(-1), (pred>best_threshold).astype(np.int8).reshape(-1) ))
                y.append( mean_absolute_error( true.reshape(-1), pred.reshape(-1) ) )
                x.append( true.sum() / true.size )
            sns.lineplot( x = x, y= score, ax = ax[1][1] ) 
            sns.lineplot(x =  x, y = y, ax = ax[0][1]) 
            del x, y, score
            gc.collect()
            
            self.models.append( model )
            self.thresholds.append( best_threshold )
            print( pred_y.sum(), valid_label.sum() )
            print( 'current validation metric', _.numpy(), 'threshold', best_threshold )
            del pred_y, valid_label, valid_data
            gc.collect()

            self.avg_score += _.numpy()
            pred_train = model.predict( train_data, verbose = 0,batch_size=self.batch_size )[-1]
            train_label = train_label[:pred_train.shape[0]]
            score = []
            x = []
            y = []
            for pred, true in zip(pred_train, train_label):
                
                if true.astype(np.int8).sum() == 0 and (pred>best_threshold).astype(np.int8).sum() == 0:
                    score.append(1)
                else:
                    score.append(self.metric( (true>best_threshold).astype(np.int8).reshape(-1), (pred>best_threshold).astype(np.int8).reshape(-1) ))
                
                y.append( mean_absolute_error( true.reshape(-1), pred.reshape(-1) ) )
                x.append( true.sum() / true.size )
             
            sns.lineplot( x = x, y= score, ax = ax[1][0] )
            sns.lineplot(x =  x, y = y, ax = ax[0][0])
            del x, y, score
            gc.collect()
            tf.keras.backend.clear_session()
        self.avg_score /= len(self.folds)
        print( "for best mean of f1 ", self.avg_score )
    
    def predictV3(self, test, blooms = 5):
        def preload( x ):
            height, width, _ = x.shape
            x = x.reshape( -1, _ )  
            x = self.normalize.transform( features_extraction(x) )
            x = x.reshape( height, width, -1 )
            x = self.preprocess(x)
            return x
        image_resolution = [ x.shape[:2] for x in test ]
        test = [ preload(x) for x in test ]
        test = np.concatenate(test)

        
        pred_y = np.zeros( (test.shape[0], self.image_size, self.image_size, 1), dtype = np.float32 )

        for model in self.models: 
            pred_y += model.predict( test, verbose = 0, batch_size=self.batch_size )[-1].astype(np.float32) #[-1]
        pred_y /= len(self.models)
        threshold = sum(self.thresholds)/ len(self.thresholds)
        
        pred_mask = ( pred_y> threshold ).astype(np.uint8)
        low_blooms = pred_mask.sum(axis = (1, 2, 3)) < blooms
        print(pred_mask.sum(), threshold)
        pred_mask = [ tf.image.resize(x, size, method='nearest' ).numpy() for x, size in zip(pred_mask, image_resolution) ]
        
        
        new_threshold = np.ones_like(pred_y, dtype = np.float32) * threshold
        
        new_threshold[low_blooms] += (1 - threshold)/2
        
        low_blooms = (pred_y>new_threshold).astype(np.uint8)
        low_blooms = [ tf.image.resize(x, size, method='nearest' ).numpy() for x, size in zip(low_blooms, image_resolution) ]
        return pred_mask, low_blooms
