from styx_msgs.msg import TrafficLight
import cv2
import numpy as np

LIGHT_THRESHOLD = 50 # How many pixels with certain color has to be detected to report light
class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        pass

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
        lower_red = np.array([0,120,70])
        upper_red = np.array([10,255,255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)
        # Range for upper range
        lower_red = np.array([170,120,70])
        upper_red = np.array([180,255,255])
        mask2 = cv2.inRange(hsv,lower_red,upper_red)
        # Generating the final mask to detect red color
        mask1 = mask1+mask2
        nzCount = cv2.countNonZero(mask1)
        if nzCount > LIGHT_THRESHOLD:
            return TrafficLight.RED
        # Detect Yellow light
        lower_yellow = np.array([25,180,100])
        upper_yellow = np.array([35,255,255])
        mask1 = cv2.inRange(hsv, lower_yellow, upper_yellow)
        nzCount = cv2.countNonZero(mask1)
        if nzCount > LIGHT_THRESHOLD:
            return TrafficLight.YELLOW
        return TrafficLight.UNKNOWN
