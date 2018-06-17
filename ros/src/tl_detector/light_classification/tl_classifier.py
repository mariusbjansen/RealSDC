from styx_msgs.msg import TrafficLight
import numpy as np
import cv2

#import ptvsd
#ptvsd.enable_attach("my_secret", address = ('127.0.0.1', 3000))
#ptvsd.wait_for_attach()

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        pass

    def get_classification(self, image, light):
        """Creates training data from the simulator and the given ground truth

        Will place images in folder with name 0,1 or 2 for RED, YELLOW, GREEN
        Manual steps necessary: 
            - Review if there are false positives in any folder
            - Move images with no traffic light visible in the folder 3 (UNKNOWN)
                - For example do it with an empty recycle bin like this
                    - Install gthumb
                    - Open gthumb, click until you find a candidate you want to remove
                    - Delete by pressing delete key
                    - Find all your candidates in recycle bin
                    - Move all those candidates to folder 3 (UNKNOWN)
                    - done

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            Not necessary in this context
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        root = "/home/student/train/"
        filename = root+str(light.state)+"/"+str(self.cnt)+".png"
        cv2.imwrite(filename,image)
        self.cnt+=1
        
        # in this context UNKNOWN is OK. purpose is to create training data
        # please see code above
        return TrafficLight.UNKNOWN


    cnt = long(0)
