from styx_msgs.msg import TrafficLight
import cv2
import os
import fnmatch
import numpy as np
import tensorflow as tf
import random
from tensorflow.contrib.layers import flatten

BATCH_SIZE = 1

LIGHT_THRESHOLD = 50 # How many pixels with certain color has to be detected to report light

CLASSIFICATION = np.array(['red', 'yellow', 'other'])
class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        pass
    def LeNet(self, x):
        global layer1Conv
        global layer2Conv
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

        # Layer 2: Convolutional. Output = 44x30x16.
        layer2Weights = tf.Variable(tf.truncated_normal([5,5,6,16], mu, sigma))
        layer2Bias = tf.Variable(tf.zeros(16))
        # stride for each dimension (batch_size, height, width, depth)
        strides = [1, 1, 1, 1]
        padding = 'VALID'
        layer2Conv = tf.nn.conv2d(layer1, layer2Weights, strides, padding) + layer2Bias
        # Activation relu.
        layer2 = tf.nn.relu(layer2Conv)
 
        # Pooling. Input = 44x30x16. Output = 22x15x16.
        filter_shape = [1, 2, 2, 1]
        strides = [1, 2, 2, 1]
        padding = 'VALID'
        layer2 = tf.nn.max_pool(layer2,filter_shape, strides, padding)
        # Flatten. Input = 5x5x16,14x14x6. Output = 1576.
        #layer2Flat = flatten(layer2)
        #layer1Flat = flatten(layer1)
        #layer2 = tf.concat(1,[layer1Flat,layer2Flat])
        # Flatten. Input = 22x15x16. Output = 5280.
        layer2 = flatten(layer2)
        # Layer 3: Fully Connected. Input = 5280. Output = 120.
        layer3Weights = tf.Variable(tf.truncated_normal([5280, 120], mu, sigma))
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
        logits = tf.add(tf.matmul(layer4, layer5Weights), layer5Bias)
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

        #convert to tensor flow input
        # convert image to a 3D uint8 tensor
        #image = tf.convert_to_tensor(image)
        # Convert to float in the [0,1] range.
        #image = tf.image.convert_image_dtype(image, tf.float32)
        image = image.astype(np.float32)
        image = image/255.0
        return image

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

    def train_model_sim(self):
        #load training data
        X_train = []
        y_train = []
        trainingList = self.list_png_paths('Training/simImg/')

        for path in trainingList:
            (tf_img, label) = self.load_img_and_label(path)
            X_train.append(tf_img)
            y_train.append(label)

        #load validation data
        X_val = []
        y_val = []
        trainingList = self.list_png_paths('Validation/simImg/')

        for path in trainingList:
            (tf_img, label) = self.load_img_and_label(path)
            X_val.append(tf_img)
            y_val.append(label)
        #load the model
        x = tf.placeholder(tf.float32, (None,100, 72, 3)) #input size
        y = tf.placeholder(tf.int32, (None, 3))
        learning_rate = tf.placeholder(tf.float32, shape=[])
        #one_hot_y = tf.one_hot(y, 3)

        rate = 0.001

        logits = self.LeNet(x)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss_operation = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_operation = optimizer.minimize(loss_operation)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            num_train = len(X_train)
            num_valid = len(X_val)
            print("num train:", num_train)
            print("num valid:", num_valid)
            print("Training...")
            print()
            prev_valid_acc = 0.0
            for i in range(10):
                # Shuffle training data
                combined = list(zip(X_train, y_train))
                random.shuffle(combined)
                X_train[:], y_train[:] = zip(*combined)
                # Train
                for offset in range(0, num_train, BATCH_SIZE):
                    end = offset + BATCH_SIZE
                    batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                    sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, learning_rate: rate})
                #Validate   
                total_accuracy = 0
                sess = tf.get_default_session()
                for offset in range(0, num_valid, BATCH_SIZE):
                    batch_x, batch_y = X_val[offset:offset+BATCH_SIZE], y_val[offset:offset+BATCH_SIZE]
                    accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
                    total_accuracy += (accuracy * len(batch_x))
                validation_accuracy = total_accuracy / num_valid 
                #if validation_accuracy < prev_valid_acc:
                #    rate/=2
                #    print("Learning rate dropped to {}".format(rate))
                print("EPOCH {} ...".format(i+1))
                print("Validation Accuracy = {:.3f}".format(validation_accuracy))
                print()
                prev_valid_acc = validation_accuracy
                
            saver.save(sess, './signs')
            print("Model saved")


    def load_img_and_label(self, path):
        label = self.get_label(path)
        cv_img = cv2.imread(path)
        tf_img = self.preprocess_input_sim(cv_img)
        #sess = tf.InteractiveSession()
        #a = tf.Print(tf_img, [tf_img], message="This is tf_img: ")
        #a.eval()
        return tf_img, label


    def get_classification_sim(self, image):
        pass

    def get_classification_simple(self, image):
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
        lower_red = np.array([0,120,70])
        upper_red = np.array([10,255,255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        # Range for upper range
        lower_red = np.array([170,120,70])
        upper_red = np.array([180,255,255])
        mask2 = cv2.inRange(hsv,lower_red,upper_red)
        # Generating the final mask to detect red color
        mask1 = mask1+mask2
        #cv2.imshow('red' ,mask1)
        nzCount = cv2.countNonZero(mask1)
        if nzCount > LIGHT_THRESHOLD:
            return TrafficLight.RED
        # Detect Yellow light
        lower_yellow = np.array([25,180,100])
        upper_yellow = np.array([35,255,255])
        mask1 = cv2.inRange(hsv, lower_yellow, upper_yellow)
        #cv2.imshow('yellow' ,mask1)
        nzCount = cv2.countNonZero(mask1)
        if nzCount > LIGHT_THRESHOLD:
            return TrafficLight.YELLOW
        return TrafficLight.UNKNOWN
