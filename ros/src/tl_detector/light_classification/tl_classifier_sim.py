from styx_msgs.msg import TrafficLight
import cv2
import os
import fnmatch
import numpy as np
import tensorflow as tf
import random
from tensorflow.contrib.layers import flatten

BATCH_SIZE = 1 # Training batch size

EPOCH_NUM = 20 # Number of traininig epochs

LIGHT_THRESHOLD = 50 # How many pixels with certain color has to be detected to report light

CLASSIFICATION = np.array(['red', 'yellow', 'other'])
class TLClassifierSim(object):
    def __init__(self):
        self.filePath = os.path.dirname(os.path.abspath(__file__)) # tl_classifier path
        print("file Path: ", self.filePath)

        self.sessSim = tf.Session()
        # Load the model
        self.simSaver = tf.train.import_meta_graph(self.filePath + '/trainedModel/simulation.ckpt.meta')
        # Initialize model with training values
        self.simSaver.restore(self.sessSim, self.filePath + '/trainedModel/simulation.ckpt')

        #Load model
        graph = tf.get_default_graph()
        self.xSim  = graph.get_tensor_by_name('xSim:0')
        self.logitsSim = graph.get_tensor_by_name('logitSim:0')

    def get_label(self, file_path):
        # convert the path to a list of path components
        #parts = tf.strings.split(file_path, '/')
        path, folder = os.path.split(file_path)
        path, folder = os.path.split(path)
        # The second to last is the class-directory
        #return  tf.convert_to_tensor((folder == CLASSIFICATION).astype(np.int32))
        return (folder == CLASSIFICATION).astype(np.int32)

    def list_png_paths(self, patchToData):
        matches = []
        for root, dirnames, filenames in os.walk(patchToData):
            for filename in fnmatch.filter(filenames, '*.png'):
                matches.append(os.path.join(root, filename))
        return matches

    def load_img_and_label(self, path):
        label = self.get_label(path)
        cvImg = cv2.imread(path)
        cvImg = self.preprocess_input_sim(cvImg)
        return cvImg, label


    def LeNet_sim(self, x):
        # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
        mu = 0
        sigma = 0.1

        # Layer 1: Convolutional. Input = 100x72x3 . Output = 96x68x6
        layer1Weights = tf.Variable(tf.truncated_normal([5,5,3,6], mu, sigma))
        layer1Bias = tf.Variable(tf.zeros(6))

        # stride for each dimension (batch_size, height, width, depth)
        strides = [1, 1, 1, 1]
        padding = 'VALID'
        # https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#conv2d
        # `tf.nn.conv2d` does not include the bias computation so we have to add it ourselves after.
        layer1Conv = tf.nn.conv2d(x, layer1Weights, strides, padding) + layer1Bias
        # Activation relu.
        layer1 = tf.nn.relu(layer1Conv)

        # Pooling. Input = 96x68x6. Output = 48x34x6.
        filter_shape = [1, 2, 2, 1]
        strides = [1, 2, 2, 1]
        padding = 'VALID'
        layer1 = tf.nn.max_pool(layer1, filter_shape, strides, padding)

        # Layer 2: Convolutional. Input = 48x34x6.  Output = 40x26x16.
        layer2Weights = tf.Variable(tf.truncated_normal([9,9,6,16], mu, sigma))
        layer2Bias = tf.Variable(tf.zeros(16))
        # stride for each dimension (batch_size, height, width, depth)
        strides = [1, 1, 1, 1]
        padding = 'VALID'
        layer2Conv = tf.nn.conv2d(layer1, layer2Weights, strides, padding) + layer2Bias
        # Activation relu.
        layer2 = tf.nn.relu(layer2Conv)
        # Pooling. Input = 40x26x16. Output = 20x13x16.
        filter_shape = [1, 2, 2, 1]
        strides = [1, 2, 2, 1]
        padding = 'VALID'
        layer2 = tf.nn.max_pool(layer2,filter_shape, strides, padding)
        # Flatten. Input = 20x13x16. Output = 4160.
        layer2 = flatten(layer2)
        # Layer 3: Fully Connected. Input = 4160. Output = 120.
        layer3Weights = tf.Variable(tf.truncated_normal([4160, 120], mu, sigma))
        layer3Bias = tf.Variable(tf.zeros(120))
        layer3 = tf.add(tf.matmul(layer2, layer3Weights), layer3Bias)
        # Activation relu.
        layer3 = tf.nn.relu(layer3)

        # Layer 4: Fully Connected. Input = 120. Output = 84.
        layer4Weights = tf.Variable(tf.truncated_normal([120, 84], mu, sigma))
        layer4Bias = tf.Variable(tf.zeros(84))
        layer4 = tf.add(tf.matmul(layer3, layer4Weights), layer4Bias)
        # Activation, sigmoid.
        #layer4 = tf.nn.relu(layer4)
        layer4 = tf.nn.sigmoid(layer4)
        # Layer 5: Fully Connected. Input = 84. Output = 3.
        layer5Weights = tf.Variable(tf.truncated_normal([84, 3], mu, sigma))
        layer5Bias = tf.Variable(tf.zeros(3))
        logits = tf.add(tf.matmul(layer4, layer5Weights), layer5Bias, name='logitSim')
        return logits

    def preprocess_input_sim(self, image):
        #crop image
        image = image[0:550, 100:500]
        #reduce pixel countrandom.shuffle
        #cv2.imshow('full', image)
        image = cv2.resize(image, (72,100))

        #cv2.imshow('resize', image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        image = image.astype(np.float32)
        image = image/255.0
        return image

    def train_model_sim(self):
        #load training data
        xTrain = []
        yTrain = []
        trainingList = self.list_png_paths(self.filePath + '/Training/simImg/')

        for path in trainingList:
            (tf_img, label) = self.load_img_and_label(path)
            xTrain.append(tf_img)
            yTrain.append(label)

        #load validation data
        xVal = []
        yVal = []
        trainingList = self.list_png_paths(self.filePath + '/Validation/simImg/')

        for path in trainingList:
            (tf_img, label) = self.load_img_and_label(path)
            xVal.append(tf_img)
            yVal.append(label)
        #load the model
        x = tf.placeholder(tf.float32, (None,100, 72, 3), name='xSim') #input size
        y = tf.placeholder(tf.int32, (None, 3))
        learningRate = tf.placeholder(tf.float32, shape=[])
        #one_hot_y = tf.one_hot(y, 3)

        rate = 0.001

        logits = self.LeNet_sim(x)
        crossEntropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        lossOperation = tf.reduce_mean(crossEntropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
        trainingOperation = optimizer.minimize(lossOperation)

        correctPrediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracyOperation = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            numTrain = len(xTrain)
            numValid = len(xVal)
            print("num train:", numTrain)
            print("num valid:", numValid)
            print("Training...")
            print()
            prev_valid_acc = 0.0
            for i in range(EPOCH_NUM):
                # Shuffle training data
                combined = list(zip(xTrain, yTrain))
                random.shuffle(combined)
                xTrain[:], yTrain[:] = zip(*combined)
                # Train
                for offset in range(0, numTrain, BATCH_SIZE):
                    end = offset + BATCH_SIZE
                    batch_x, batch_y = xTrain[offset:end], yTrain[offset:end]
                    sess.run(trainingOperation, feed_dict={x: batch_x, y: batch_y, learningRate: rate})
                #Validate
                total_accuracy = 0
                sess = tf.get_default_session()
                for offset in range(0, numValid, BATCH_SIZE):
                    batch_x, batch_y = xVal[offset:offset+BATCH_SIZE], yVal[offset:offset+BATCH_SIZE]
                    accuracy = sess.run(accuracyOperation, feed_dict={x: batch_x, y: batch_y})
                    total_accuracy += (accuracy * len(batch_x))
                validation_accuracy = total_accuracy / numValid
                #if validation_accuracy < prev_valid_acc:
                #    rate/=2
                #    print("Learning rate dropped to {}".format(rate))
                print("EPOCH {} ...".format(i+1))
                print("Validation Accuracy = {:.3f}".format(validation_accuracy))
                print()
                prev_valid_acc = validation_accuracy
            save_path = saver.save(sess, self.filePath + '/trainedModel/simulation.ckpt')
            print("Model saved in path: %s" % save_path)


    def get_classification_sim(self, image):
        image = self.preprocess_input_sim(image)
        images = [image]
        #detection =  np.argmax(self.sessSim.run(y_pred, feed_dict={x: image}), axis=1)
        classificationOutput = self.sessSim.run(self.logitsSim, feed_dict={self.xSim: images})
        detection =  np.argmax(classificationOutput, axis=1)
        print(classificationOutput)
        print(detection)
        print(CLASSIFICATION[detection])
        if CLASSIFICATION[detection] == 'red':
            return TrafficLight.RED
        elif CLASSIFICATION[detection] == 'yellow':
            return TrafficLight.YELLOW
        else: TrafficLight.UNKNOWN


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # converting from BGR to HSV color space
        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        #Crop the image to select only middle traffic signal
        hsv = hsv[50:550, 320:500]
        # Detect Red light
        # Range for lower red
        lowerRed = np.array([0,120,70])
        upperRed = np.array([10,255,255])
        mask1 = cv2.inRange(hsv, lowerRed, upperRed)
        # Range for upper range
        lowerRed = np.array([170,120,70])
        upperRed = np.array([180,255,255])
        mask2 = cv2.inRange(hsv,lowerRed,upperRed)
        # Generating the final mask to detect red color
        mask1 = mask1+mask2
        #cv2.imshow('red' ,mask1)
        nzCount = cv2.countNonZero(mask1)
        if nzCount > LIGHT_THRESHOLD:
            return TrafficLight.RED
        # Detect Yellow light
        lowerYellow = np.array([25,180,100])
        upperYellow = np.array([35,255,255])
        mask1 = cv2.inRange(hsv, lowerYellow, upperYellow)
        #cv2.imshow('yellow' ,mask1)
        nzCount = cv2.countNonZero(mask1)
        if nzCount > LIGHT_THRESHOLD:
            return TrafficLight.YELLOW
        return TrafficLight.UNKNOWN
