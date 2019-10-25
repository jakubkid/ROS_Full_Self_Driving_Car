from styx_msgs.msg import TrafficLight
import cv2
import os
import fnmatch
import numpy as np
import tensorflow as tf

BATCH_SIZE = 16

LIGHT_THRESHOLD = 50 # How many pixels with certain color has to be detected to report light

CLASSIFICATION = np.array(['red', 'yellow', 'other'])
class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        pass

    def preprocess_input_sim(self, image):
        #crop image
        image = image[0:550, 100:500]
        #reduce pixel count
        cv2.imshow('full', image)
        image = cv2.resize(image, (130,180))

        cv2.imshow('resize', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #convert to tensor flow input
        # convert image to a 3D uint8 tensor
        image = tf.convert_to_tensor(image)
        # Convert to float in the [0,1] range.
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image

    def get_label(self, file_path):
        # convert the path to a list of path components
        #parts = tf.strings.split(file_path, '/')
        path, folder = os.path.split(file_path)
        path, folder = os.path.split(path)
        # The second to last is the class-directory
        return  folder == CLASSIFICATION

    def list_png_paths(self, patchToData):
        matches = []
        for root, dirnames, filenames in os.walk(patchToData):
            for filename in fnmatch.filter(filenames, '*.png'):
                matches.append(os.path.join(root, filename))
        return matches

    def train_model_sim(self):
        #load training data
        trainingList = self.list_png_paths('Training/simImg/')
        for path in trainingList:
            label = self.get_label(path)
            #TODO remove
            if label[1]:
                cv_img = cv2.imread(path)
                self.preprocess_input_sim(cv_img)


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
