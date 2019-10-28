import rosbag
from cv_bridge import CvBridge
import cv2
import time

from tl_classifier_sim import TLClassifierSim


light_classifier = TLClassifierSim()
light_classifier.train_model_sim()

