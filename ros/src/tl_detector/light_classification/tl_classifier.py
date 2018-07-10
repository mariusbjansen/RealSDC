from styx_msgs.msg import TrafficLight
import rospy, rospkg
import tensorflow as tf
import numpy as np
import cv2
import os

from datetime import datetime

# path setup
rp = rospkg.RosPack()
# get path of this package
tl_path = rp.get_path("tl_detector")
# get specific subfolder
lc_path = 'light_classification'
# frozen model name
## Selection of newer ssd lite mobilenet v2 net pretrained with coco
## see analysis in doc folder
#pb_name = 'ssd_mobilenet_v1_coco_2017_11_17.pb'
#pb_name = 'ssdlite_mobilenet_v2_coco.pb'
pb_name = 'ssd_mobilenet_v2_coco_2018_03_29.pb'
pb_path = os.path.join(tl_path, lc_path, pb_name)


### PARAMETER SECTION BEGIN ###
## traffic light detection ##
# class of traffic light (no need to change)
CLASS_TRAFFIC_LIGHT = 10
# confidence threshold when to accept/reject traffic light match
THRESHOLD_SCORE = 0.10

## color classification
# reminder HSV (Hue, Saturation, Value)
# amazing website http://mkweb.bcgsc.ca/color-summarizer/ helped me to find the values
# Note the H values are from 0-360 on the website and 0-180 in opencv
# also S and V are from 0-100 on the website and 0-255 in opencv
# minumum Saturation of HSV color space
SAT_LOW = 160
# minimum Value of HSV color space
VAL_LOW = 140
# minimum hue red area low
HUE_RED_MIN1 = 0
# maximum hue red area low
HUE_RED_MAX1 = 10
# minimum hue red area high
HUE_RED_MIN2 = 170
# maximum hue red area high
HUE_RED_MAX2 = 179
# minimum hue yellow area
HUE_YELLOW_MIN = 20
# maximum hue yellow area
HUE_YELLOW_MAX = 40
# minimum hue green area
HUE_GREEN_MIN = 50
# maximum hue green area
HUE_GREEN_MAX = 70

## ground truth for training
TRAINING = True # Set to True if you want examples to train
TRAIN_ROOT = os.path.join(tl_path, '../../../../train')
TLC_DEBUG = True

# Parameters for Color CNN NET
COLOR_CNN_NET_NAME = 'ColorTrafficSign_Net'
COLOR_CNN_RUN_MODE = 'simulator' # simulator / real
COLOR_CNN_FULL_NAME = os.path.join(tl_path, lc_path, COLOR_CNN_NET_NAME)

### PARAMETER SECTION END ###


class TLClassifier(object):
    def __init__(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(pb_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        self.sess = None
        self.setup = False
        self.tensor_dict = None
        self.image_tensor = None

        # Initialize Color CNN
        self.color_cnn_graph = None
        self.color_cnn_sess = None
        self.color_cnn_sess, self.color_cnn_graph = self.restore_my_model(COLOR_CNN_FULL_NAME, COLOR_CNN_RUN_MODE)

        # user info: you may delete the train folder itself but not individual folders inside!
        if TRAINING:
            self.training_path_root = TRAIN_ROOT
            self.training_path_orig = os.path.join(TRAIN_ROOT, 'orig')
            self.training_path_boxed = os.path.join(TRAIN_ROOT, 'boxed')
            self.training_path_tl = os.path.join(TRAIN_ROOT, 'tl')
            self.training_tl_state_folders = {
                TrafficLight.RED: '0 - RED',
                TrafficLight.YELLOW: '1 - YELLOW',
                TrafficLight.GREEN: '2 - GREEN',
                TrafficLight.UNKNOWN: '4 - UNKNOWN'}

            try:
                os.mkdir(self.training_path_root)
            except:
                pass

            try:
                os.mkdir(self.training_path_orig)
            except:
                pass

            try:
                os.mkdir(self.training_path_boxed)
            except:
                pass

            try:
                os.mkdir(self.training_path_tl)
            except:
                pass

            for state_name in self.training_tl_state_folders.values():
                try:
                    os.mkdir(os.path.join(self.training_path_orig, state_name))
                except:
                    pass

                try:
                    os.mkdir(os.path.join(self.training_path_tl, state_name))
                except:
                    pass

    def get_classification(self, image, light):
        timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]

        # Attempts to combine several scales in one image
        #image_2 = np.zeros_like(image)
        #img_small = cv2.resize(image, (image.shape[1] // 3, image.shape[0] // 3), interpolation = cv2.INTER_CUBIC)
        #image_2[:img_small.shape[0], :img_small.shape[1]] = img_small

        #img_small = cv2.resize(image, (image.shape[1] - image.shape[1] // 3, image.shape[0] - image.shape[0] // 3), interpolation = cv2.INTER_CUBIC)
        #image_2[(image_2.shape[0] - img_small.shape[0]):, (image_2.shape[1] - img_small.shape[1]):] = img_small

        #img_small = cv2.resize(image, (image.shape[1] // 3, image.shape[0] - image.shape[0] // 3), interpolation = cv2.INTER_CUBIC)
        #image_2[(image_2.shape[0] - img_small.shape[0]):, :img_small.shape[1]] = img_small

        #img_small = cv2.resize(image, (image.shape[1] - image.shape[1] // 3, image.shape[0] // 3), interpolation = cv2.INTER_CUBIC)
        #image_2[:img_small.shape[0], (image_2.shape[1] - img_small.shape[1]):] = img_small
        
        #image = image_2

        if TRAINING:
            path = os.path.join(self.training_path_orig, self.training_tl_state_folders[light.state], '{}.png'.format(timestamp))
            cv2.imwrite(path, image)

        cvimage = image[...,::-1]
        (im_height, im_width) = cvimage.shape[:2]
        npimage = np.array(cvimage.reshape(im_height, im_width, 3)).astype(np.uint8)

        # detection call
        output = self.run_inference_for_single_image(npimage)

        boxes = output['detection_boxes']
        classes =  output['detection_classes']
        scores = output['detection_scores']

        height, width = image.shape[:2]
        idxTL = np.where(classes == CLASS_TRAFFIC_LIGHT)  
        bestThresh = THRESHOLD_SCORE
        match = None

        if TRAINING:
            img_boxed = image.copy()

        for i in idxTL[0].tolist():
            if scores[i] >= THRESHOLD_SCORE:
                if TRAINING:
                    cv2.rectangle(img_boxed, (int(boxes[i][1] * width), int(boxes[i][0] * height)), (int(boxes[i][3] * width), int(boxes[i][2] * height)), (255, 255, 255), 3)

                if scores[i] > bestThresh:
                    match = i
                    bestThresh = scores[i]

        if TRAINING:
            if match is not None:
                cv2.rectangle(img_boxed, (int(boxes[match][1] * width), int(boxes[match][0] * height)), (int(boxes[match][3] * width), int(boxes[match][2] * height)), (0, 0, 255), 3)

            path = os.path.join(self.training_path_boxed, '{}.png'.format(timestamp))
            cv2.imwrite(path, img_boxed)

        if match is not None:
            # extract/crop region of interest and plot
            left_y = int(boxes[match][0]*height)
            left_x = int(boxes[match][1]*width)
            right_y = int(boxes[match][2]*height)
            right_x = int(boxes[match][3]*width)

            roi = image[left_y:right_y, left_x:right_x]

            if (self.color_cnn_sess is not None):
                # Calculate 32x32 image
                my_img_w = right_x - left_x
                my_img_h = right_y - left_y

                my_max_dim = max(my_img_w, my_img_h)
                my_sub_img = image[max(0, left_y + (my_img_h // 2) - (my_max_dim // 2)):max(0, left_y + (my_img_h // 2) - (my_max_dim // 2)) + my_max_dim, max(0, left_x + (my_img_w // 2) - (my_max_dim // 2)):max(0, left_x + (my_img_w // 2) - (my_max_dim // 2)) + my_max_dim]

                #if (my_sub_img.shape[0] != my_max_dim) or (my_sub_img.shape[1] != my_max_dim):
                my_sub_img_new = np.zeros((my_max_dim, my_max_dim, my_sub_img.shape[2]), dtype=np.uint8)
                my_sub_img_new[max(0, (my_max_dim - my_sub_img.shape[0]) // 2):max(0, (my_max_dim - my_sub_img.shape[0]) // 2) + my_sub_img.shape[0], max(0, (my_max_dim - my_sub_img.shape[1]) // 2):max(0, (my_max_dim - my_sub_img.shape[1]) // 2) + my_sub_img.shape[1]] = my_sub_img
                my_sub_img = my_sub_img_new

                my_sub_img = cv2.resize(my_sub_img, (32, 32), interpolation = cv2.INTER_CUBIC)
                #
                batch_1_img32x32 = np.empty((1,32,32,3))
                batch_1_img32x32[0,:,:,:] = my_sub_img
                #rospy.loginfo("Type of input image: {}".format(batch_1_img32x32[0,0,0,0]))

                batch_1_img32x32_normalized = self.normalize_data(batch_1_img32x32, -0.5, 0.5)

                _, _, _, _, y_list_predicted = self.predict_saved(self.color_cnn_sess, self.color_cnn_graph, batch_1_img32x32_normalized)
                y_predicted = y_list_predicted[0]
                cur_state = TrafficLight.RED if y_predicted==0 else TrafficLight.YELLOW if y_predicted==1 else TrafficLight.GREEN if y_predicted==2 else TrafficLight.UNKNOWN
            else:
                cur_state = TrafficLight.UNKNOWN
            
            cur_state = self.red_green_yellow(roi)

            if TLC_DEBUG:
                state_name = "RED" if cur_state == TrafficLight.RED else "GREEN" if cur_state == TrafficLight.GREEN else "YELLOW" if cur_state == TrafficLight.YELLOW else "UNKNOWN"
                rospy.loginfo("Traffic light status: {}".format(state_name))

            if TRAINING:
                img_w = right_x - left_x
                img_h = right_y - left_y

                max_dim = max(img_w, img_h)
                sub_img = image[max(0, left_y + (img_h // 2) - (max_dim // 2)):max(0, left_y + (img_h // 2) - (max_dim // 2)) + max_dim, max(0, left_x + (img_w // 2) - (max_dim // 2)):max(0, left_x + (img_w // 2) - (max_dim // 2)) + max_dim]

                #if (sub_img.shape[0] != max_dim) or (sub_img.shape[1] != max_dim):
                sub_img_new = np.zeros((max_dim, max_dim, sub_img.shape[2]), dtype=np.uint8)
                sub_img_new[max(0, (max_dim - sub_img.shape[0]) // 2):max(0, (max_dim - sub_img.shape[0]) // 2) + sub_img.shape[0], max(0, (max_dim - sub_img.shape[1]) // 2):max(0, (max_dim - sub_img.shape[1]) // 2) + sub_img.shape[1]] = sub_img
                sub_img = sub_img_new

                sub_img = cv2.resize(sub_img, (32, 32), interpolation = cv2.INTER_CUBIC)

                path = os.path.join(self.training_path_tl, self.training_tl_state_folders[cur_state], '{}.png'.format(timestamp))
                cv2.imwrite(path, sub_img)

            return cur_state
        else:
            rospy.loginfo("No traffic light detected")

        return TrafficLight.UNKNOWN
    
    def run_inference_for_single_image(self, image):
      with self.detection_graph.as_default():
        if (self.setup == False):
            with tf.Session() as self.sess:
              # Get handles to input and output tensors
              ops = tf.get_default_graph().get_operations()
              all_tensor_names = {output.name for op in ops for output in op.outputs}
              self.tensor_dict = {}
              for key in [
                  'num_detections', 'detection_boxes', 'detection_scores',
                  'detection_classes', 'detection_masks'
              ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                  self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                      tensor_name)
              if 'detection_masks' in self.tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(self.tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(self.tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(self.tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                self.tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
              self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
              self.setup = True
    
        if (self.setup == True):
            # Run inference
            output_dict = self.sess.run(self.tensor_dict,feed_dict={self.image_tensor: np.expand_dims(image, 0)})
        
            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
                return output_dict
        
        return output_dict

    def findNonZero(self, image_in):
      rows, cols, _ = image_in.shape
      counter = 0
    
      for row in range(rows):
        for col in range(cols):
          pixel = image_in[row, col]
          if sum(pixel) != 0:
            counter = counter + 1
    
      return counter

    # Taken and adapted from: https://github.com/thedch/traffic-light-classifier?files=1
    def red_green_yellow(self, image_in):
      hsv = cv2.cvtColor(image_in, cv2.COLOR_BGR2HSV)

      # Red
      lower_red = np.array([HUE_RED_MIN1,SAT_LOW,VAL_LOW])
      upper_red = np.array([HUE_RED_MAX1,255,255])
      red_mask = cv2.inRange(hsv, lower_red, upper_red)
      red_result = cv2.bitwise_and(image_in, image_in, mask = red_mask)
      sum_red = self.findNonZero(red_result)  
      lower_red = np.array([HUE_RED_MIN2,SAT_LOW,VAL_LOW])
      upper_red = np.array([HUE_RED_MAX2,255,255])
      red_mask = cv2.inRange(hsv, lower_red, upper_red)
      red_result = cv2.bitwise_and(image_in, image_in, mask = red_mask)  
      sum_red += self.findNonZero(red_result)
    
      # Yellow
      lower_yellow = np.array([HUE_YELLOW_MIN,SAT_LOW,VAL_LOW])
      upper_yellow = np.array([HUE_YELLOW_MAX,255,255])
      yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
      yellow_result = cv2.bitwise_and(image_in, image_in, mask = yellow_mask)
      sum_yellow = self.findNonZero(yellow_result)
      
      # Green
      lower_green = np.array([HUE_GREEN_MIN,SAT_LOW,VAL_LOW])
      upper_green = np.array([HUE_GREEN_MAX,255,255])
      green_mask = cv2.inRange(hsv, lower_green, upper_green)
      green_result = cv2.bitwise_and(image_in, image_in, mask = green_mask)
      sum_green = self.findNonZero(green_result)
    
      if (sum_red >= sum_yellow) and (sum_red >= sum_green):
        return TrafficLight.RED
      if sum_yellow >= sum_green:
        return TrafficLight.YELLOW
      return TrafficLight.GREEN

    def restore_my_model(self, my_net_name, run_mode):
      # RESET GRAPH 
      # reset any previous variables
      #tf.reset_default_graph()
    
      # Create a new Graph
      my_new_graph = tf.Graph()
      # Create a new session
      my_new_session = tf.Session(graph = my_new_graph)
    
      with my_new_graph.as_default():
        # INITIALIZE VARIABLES (to be overwrite later on when restore happens)
        # Initialize variables that will be used by our operations to be calculated
        # These weights and bias will be overwrite with saved values when saver.restore is executed below.
        # Note: IMPORTANT !!! we have to include variable but we SHOULDN'T include placeholders
        #                   placeholders will be get from get_tensor_by_name and populated with new data (see predict_saved)

        # Useful comments
        # Its a standard practice to feed all the training data and hyperparameters using placeholders. 
        # Note that when the network is saved, placeholders definitions are saved by values of the placeholders are not saved.

        NUMBER_OF_CLASSES = 4

        # Hyperparameters
        mu = 0
        sigma = 0.1
        w_conv1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 3,  6), mean = mu, stddev = sigma), name='w_conv1'),
        w_conv2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma), name='w_conv2'),
        w_Full1 = tf.Variable(tf.truncated_normal(shape=(5*5*16, 130), mean = mu, stddev = sigma), name='w_Full1'),
        w_Full2 = tf.Variable(tf.truncated_normal(shape=(130, 100), mean = mu, stddev = sigma), name='w_Full2'),
        w_out   = tf.Variable(tf.truncated_normal(shape=(100, NUMBER_OF_CLASSES), mean = mu, stddev = sigma), name='w_out')

        b_conv1 = tf.Variable(tf.zeros(6), name='b_conv1'),
        b_conv2 = tf.Variable(tf.zeros(16), name='b_conv2'),
        b_Full1 = tf.Variable(tf.zeros(130), name='b_Full1'),
        b_Full2 = tf.Variable(tf.zeros(100), name='b_Full2'),
        b_out   = tf.Variable(tf.zeros(NUMBER_OF_CLASSES), name='b_out')

        # Define an operation in order to initialize our variables
        init_op = tf.global_variables_initializer()

        # It will initialize our variables with random values by now
        my_new_session.run(init_op)

        #First let's load meta graph (actual graph with all operations, but not variables)
        saver_restored = tf.train.import_meta_graph(my_net_name + '_' + run_mode + '.ckpt.meta')

        #Second restore variables (weights are variables as well)
        saver_restored.restore(my_new_session, my_net_name + "_" + run_mode + ".ckpt")
        
        #print(my_net_name + "_" + run_mode + ".ckpt")

      return my_new_session, my_new_graph

    def predict_saved(self, sess, graph, X_data):

      # Another way to get graph is using sess.graph
      #graph = sess.graph
    
      # Now, let's access and create placeholders variables 
      x_pred = graph.get_tensor_by_name("x:0")
      #y_pred = graph.get_tensor_by_name("y:0")    
    
      #Now, access the operations that you want to run. 
      my_logits             = graph.get_tensor_by_name("my_logits:0")
      my_logits_top2        = graph.get_tensor_by_name('my_logits_top2:0')
      my_probabilities      = graph.get_tensor_by_name('my_probabilities:0')
      my_probabilities_top2 = graph.get_tensor_by_name('my_probabilities_top2:0')
      my_predictions        = graph.get_tensor_by_name('my_predictions:0')

      logs           = sess.run(my_logits, feed_dict={x_pred: X_data})
      logs_top2      = sess.run(my_logits_top2, feed_dict={x_pred: X_data})
      probab         = sess.run(my_probabilities, feed_dict={x_pred: X_data})
      probab_top2    = sess.run(my_probabilities_top2, feed_dict={x_pred: X_data})
      predicts       = sess.run(my_predictions, feed_dict={x_pred: X_data})
      
      return logs, logs_top2, probab, probab_top2, predicts

    # Normalizing data
    def normalize_data(self, data, new_min, new_max):
      x_max = np.amax(data)
      x_min = np.amin(data)
      data_normalized = new_min + ( ((data - x_min) * (new_max - new_min)) / (x_max - x_min) )
      return data_normalized
      