import rosbag
from cv_bridge import CvBridge
import cv2
import time

from tl_classifier import TLClassifier


light_classifier = TLClassifier()
light_classifier.train_model_sim()

