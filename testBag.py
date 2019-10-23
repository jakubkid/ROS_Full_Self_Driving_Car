import rosbag
from cv_bridge import CvBridge
import cv2
import time
# add path to tl classifier
import sys
sys.path.append('/ros/src/tl_detector/light_classification/')
from tl_classifier import TLClassifier


bag = rosbag.Bag('../traffic_light_bag_file/traffic_light_training.bag')
bridge = CvBridge()

for topic, image, t in bag.read_messages(topics=['/image_color']):
    cv_image = bridge.imgmsg_to_cv2(image, "bgr8")
    cv2.imshow('time '+str(t) ,cv_image)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
bag.close()
