from utils import *
import tifffile
evalution = glob.glob( "evaluation/*.tif" )
evalution = [tifffile.imread(x).reshape(-1, 12) for x in evalution ]

class SolarPanelDetectionDeepLearn:
    def __init__( self, batch_size = 64):
        
        self.models = []
        self.normalize = NormalizerV3()#NormalizerScale()#RobustScaler()
        self.thresholds = np.array([ .48, .48, .43, .48, .51])
        self.image_size = 32
        self.image_scale = 1
        self.batch_size = batch_size 
        self.n_features = 42
        print( 'preparation' )
        gc.collect()
    def preprocess(self, image, mask = None):
        image = tf.image.resize( image, ( self.image_size, self.image_size), method='nearest' ).numpy()[None,...]
        if mask is not None:
            mask = tf.image.resize( mask, ( self.image_size, self.image_size), method='nearest' ).numpy()[None,...]
            return image, mask            
        return image
    
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
            pred_y += model.predict( test, verbose = 1, batch_size=self.batch_size )[-1].astype(np.float32) #[-1]
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
Model = SolarPanelDetectionDeepLearn(  )
Model.models = [ tf.keras.models.load_model(f'./SolarDetect-solafune-github/save_model_{i}.h5', compile = False) for i in range(1, 6) ]
pred_mask = Model.predictV3( evalution )