import numpy as np
import cv2
import math
import tensorflow as tf

class BeeProcessor():
    'Process bee related images'
    def __init__(self, image, vertices, blurriness_threshold=300,
                 model_weights="Models/FCN_v6_weights-672-0.3838.hdf5",
                 lower_thresh=0.3, upper_thresh=1, min_size=6):
        'Initialization'
        
        # Error information
        self.errorFlag = False
        self.errorMsg = ''
        
        # Image to be processed
        self.input = image
        self.input_dim = image.shape
        
        # Vertices of the Region of Interest
        self.vertices = vertices
        self.vx_min = vertices[0, :, 0].min()
        self.vx_max = vertices[0, :, 0].max()
        self.vy_min = vertices[0, :, 1].min()
        self.vy_max = vertices[0, :, 1].max()
        
        # Region of Interest
        self.roi, self.strict_roi = self.region_of_interest()
        
        # Blurriness check
        self.blurriness_threshold = blurriness_threshold
        self.check_blurriness()
        
        # Variables related to breaking up image to segments
        self.side = 200
        self.halfside = self.side / 2
        self.a = math.floor(self.strict_roi.shape[0] / 200)
        self.b = math.floor(self.strict_roi.shape[1] / 200)
        self.segment_nbr = self.a * self.b
        self.grid = np.mgrid[0:self.a,0:self.b]*self.side + self.halfside
        self.segments = self.make_segments_for_NN()
        self.check_segments()
        
        # NN prediction
        self.model = 0
        self.model_weights = model_weights
        self.create_model()
        
        # Bee position estimation
        self.lower_thresh = lower_thresh
        self.upper_thresh = upper_thresh
        self.erosion_kernel = np.ones((3,3),np.uint8)
        self.min_size = min_size
        self.scale = 10

    def region_of_interest(self):
        """
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        `vertices` should be a numpy array of integer points.
        """
        # Defining a blank mask to start with
        mask = np.zeros_like(self.input)

        # Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(self.input_dim) > 2:
            channel_count = self.input_dim[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # Filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, self.vertices, ignore_mask_color)

        # Returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(self.input, mask)

        # Decrease image size
        masked_cropped_image = masked_image[self.vy_min:self.vy_max, self.vx_min:self.vx_max, :]
        
        return masked_image, masked_cropped_image

    def check_blurriness(self):
        """
        Calculate an indicator which measure blurriness.
        Sets error flag if indicator above preset threshold.
        Good practice: Call this function on region of interest.
        """
        # Apply laplacian kernel and calculate variance of pixels
        gray = cv2.cvtColor(self.input, cv2.COLOR_BGR2GRAY)
        indicator = cv2.Laplacian(gray, cv2.CV_64F).var()
    
        # Small amount of variance in edge detected image = blurry image
        if (indicator > self.blurriness_threshold):
            self.errorFlag = True
            self.errorMsg += 'Blurry image! '
            return True, indicator
        else:
            return False, indicator
            
    def make_segments_for_NN(self):        
        'Breaking up RoI to smaller segments for Neural Network.'
        output = []
        for i in range(self.a):
            for j in range(self.b):
                segment = self.strict_roi[int(self.grid[0,i,j]-self.halfside) :
                                          int(self.grid[0,i,j]+self.halfside) ,
                                          int(self.grid[1,i,j]-self.halfside) :
                                          int(self.grid[1,i,j]+self.halfside)]
                output.append(segment)
        
        return output
    
    def check_segments(self):
        """
        The enclosing rectangle of the Region of Interest will be
        split into AxA segments for Neural Network. The length and
        width of the rectangle should be integer times A.
        """
        dx = self.vx_max - self.vx_min
        dy = self.vy_max - self.vy_min
        
        if (dx%self.side != 0) or (dy%self.side != 0):
            self.errorFlag = True
            self.errorMsg += 'Neural Network will not process the whole RoI! '
            
    def save_segments(self, folderpath='Saved_segments/', prefix=''):
        'Saving segments for labeling'
        index = 0
        for segment in self.segments:
            cv2.imwrite(folderpath + prefix + str(index) + '.jpg', segment)
            index += 1
    
    def create_model(self):
        'Model structure created and trained weights loaded'
        model_name="FCN_v6"
        sample_segment_dim = (self.side, self.side, self.input_dim[2])
        max_norm_rate_c = tf.keras.constraints.MaxNorm(3)
        spatial_dropout_rate = 0.5

        self.model = tf.keras.Sequential(name=model_name)

        self.model.add(tf.keras.layers.Convolution2D(16, (5,5), strides=(2, 2), activation="relu", input_shape=sample_segment_dim))
        self.model.add(tf.keras.layers.SpatialDropout2D(spatial_dropout_rate))
        self.model.add(tf.keras.layers.GaussianNoise(0.4))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))

        self.model.add(tf.keras.layers.Convolution2D(24, (5,5), strides=(2, 2), activation="relu", kernel_constraint=max_norm_rate_c, input_shape=sample_segment_dim))
        self.model.add(tf.keras.layers.SpatialDropout2D(spatial_dropout_rate))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid'))

        self.model.add(tf.keras.layers.Convolution2D(32, (3,3), activation="relu", kernel_constraint=max_norm_rate_c))
        self.model.add(tf.keras.layers.SpatialDropout2D(spatial_dropout_rate))

        self.model.add(tf.keras.layers.Convolution2D(64, (3,3), activation="relu", kernel_constraint=max_norm_rate_c))
        self.model.add(tf.keras.layers.SpatialDropout2D(spatial_dropout_rate))

        self.model.add(tf.keras.layers.Convolution2D(96, (3,3), activation="relu", kernel_constraint=max_norm_rate_c))
        self.model.add(tf.keras.layers.SpatialDropout2D(spatial_dropout_rate))

        self.model.add(tf.keras.layers.Conv2DTranspose(8, (4,4), activation="relu"))

        self.model.add(tf.keras.layers.Conv2DTranspose(6, (4,4), activation="relu"))

        self.model.add(tf.keras.layers.Conv2DTranspose(4, (4,4), activation="relu"))

        self.model.add(tf.keras.layers.Conv2DTranspose(2, (4,4), activation="relu"))

        self.model.add(tf.keras.layers.Conv2DTranspose(2, (4,4), activation="softmax"))

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

        self.model.load_weights(self.model_weights)
    
    def predict_class(self):
        'Return prediction'
        if self.errorFlag:
            print('Prediction might not be accurate due to settings issues!')
        
        # Set up generator
        input_data = np.asarray(self.segments)
        test_ID_list = np.arange(input_data.shape[0])
        params = {'dim': (self.side, self.side, self.input_dim[2]),
                  'batch_size': 1,
                  'n_classes': 2}
        prediction_generator = DataGenerator(input_data, test_ID_list, **params)
        
        # Run prediction on segments
        output = self.model.predict_generator(prediction_generator, verbose=0)
        
        return output
    
    def estimate_bee_positions(self):
        'Return prediction'
        if self.errorFlag:
            print('Estimation will be less accurate due to settings issues!')
        
        # Run prediction
        prediction = self.predict_class()
        
        # Concat prediction segments
        # OpenCV functions prefer background (detected mite make a "hole" in bee)
        concated_predictions = np.zeros((self.a*20, self.b*20))
        for i in range(self.a):
            for j in range(self.b):
                concated_predictions[i*20:i*20+20, j*20:j*20+20] = prediction[i*self.b+j][:, :, 0]
        
        # Finding sure foreground area
        ret, sure_fg = cv2.threshold(concated_predictions, self.lower_thresh, self.upper_thresh, cv2.THRESH_BINARY_INV)
        
        # Erosion
        erosion = cv2.erode(sure_fg, self.erosion_kernel, iterations = 1)
        erosion_uint8 = erosion.astype(np.uint8)
        
        # Find connected components (blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(erosion_uint8, connectivity=8)

        # Remove background which is also considered a component
        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        # Keep only which is above min_size
        centers = []
        for i in range(0, nb_components):
            if sizes[i] >= self.min_size:
                centers.append((int(self.scale*centroids[i+1, 0]), int(self.scale*centroids[i+1, 1])))
                
        return centers
    
    def read_errors(self):
        """
        Return error messages generated by sanity checks.
        Good practice: Call this function after input processed and
        before new input is given.
        """
        return self.errorFlag, self.errorMsg
        
    def new_input(self, input):
        """
        New image to be processed. Same dimensions expected
        """
        # Error information
        self.errorFlag = False
        self.errorMsg = ''
        
        # Preprocess image
        self.input = input
        self.roi, self.strict_roi = self.region_of_interest()
        self.check_blurriness()
        self.segments = self.make_segments_for_NN()
        self.check_segments()

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, list_IDs, batch_size=32,
                 dim=(32,32,32), n_classes=10):
        'Initialization'
        self.data = data
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.dim = dim
        self.n_classes = n_classes
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        x = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, 20, 20), dtype=int)

        # Load batch data
        for i, ID in enumerate(list_IDs_temp):
            # Load image segment and normalize it
            x[i,] = (self.data[ID, :, :, :]/255) - 0.5
            y[i] = 0

        return x, y