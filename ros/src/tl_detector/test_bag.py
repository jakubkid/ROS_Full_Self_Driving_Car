import rosbag
from cv_bridge import CvBridge
import cv2
import time
# add path to tl classifier
#import sys
#sys.path.append('/ros/src/tl_detector/')
from light_classification.tl_classifier import TLClassifier


bag = rosbag.Bag('../../../../traffic_light_bag_file/traffic_light_training.bag')
bridge = CvBridge()
light_classifier = TLClassifier()
prevTtag = 0
startSec = None
for topic, image, t in bag.read_messages(topics=['/image_color']):
    if startSec == None:
        startSec = t.secs
    ttagMs = (t.secs-startSec)*1000 + t.nsecs/1000000
    if ttagMs - prevTtag < 100:
	continue
    prevTtag = ttagMs
    cv_image = bridge.imgmsg_to_cv2(image, "bgr8")
    light_classifier.get_classification(cv_image)
    #cv2.imshow('time ms ' + str(ttagMs), cv_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #print('realImg/img'+ str(ttagMs)+ '.png')
    status = cv2.imwrite('realImg/img'+ str(ttagMs)+ '.png', cv_image)
    #print("Image written to file-system : ", status)
bag.close()
