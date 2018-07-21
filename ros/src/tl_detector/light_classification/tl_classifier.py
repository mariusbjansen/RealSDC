from styx_msgs.msg import TrafficLight
import rospy, rospkg
import tensorflow as tf
import numpy as np
import cv2
import math
import os
from datetime import datetime

from DeepModelEngine import DeepModelEngineV3

tl_path = os.path.join(rospkg.RosPack().get_path("tl_detector"), 'light_classification')

# frozen model name alternatives
pb_name = 'ssd_mobilenet_v1_coco_2017_11_17.pb'
pb_path = os.path.join(tl_path, pb_name)

# CNN model
model_dir = os.path.join(tl_path, 'deep_model')
model_data_shape = (32, 32, 3)
model_class_num = 4
model_depth = 2

### PARAMETER SECTION BEGIN ###
## traffic light detection ##
# class of traffic light (no need to change)
CLASS_TRAFFIC_LIGHT = 10
# confidence threshold when to accept/reject traffic light match
THRESHOLD_SCORE = 0.025

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
TLC_DEBUG = True
TRAINING = False # Set to True if you want examples to train
TRAIN_ROOT = os.path.join(tl_path, '../../../../../train')


class TLClassifier(object):
    def __init__(self):
        self.sess = None
        self.setup = False
        self.tensor_dict = None
        self.image_tensor = None

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
                    os.mkdir(os.path.join(self.training_path_tl, state_name))
                except:
                    pass

        self.hist_clahe = cv2.createCLAHE(clipLimit = 1.0, tileGridSize = (48, 48))

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(pb_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        self.cnn_model = DeepModelEngineV3(
            storage_dir = model_dir,
            data_shape = model_data_shape,
            class_num = model_class_num,
            model_depth = model_depth)
        self.cnn_session = self.cnn_model.load_model()

        # Do first fake classification to initialize TF
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.get_classification(image)

    def get_classification(self, image):
        timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]

        if TRAINING:
            path = os.path.join(self.training_path_orig, '{}.jpg'.format(timestamp))
            cv2.imwrite(path, image)

        # Image color correction
        #image = self.apply_clahe(image)
        image = self.color_correction(image, 1.0)
        image = self.gamma_correction(image)
        image = cv2.GaussianBlur(image, (7, 7), 0)

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if TRAINING:
            img_boxed = image.copy()

        # TL position detection call
        output = self.run_inference_for_single_image(img_rgb)

        boxes = output['detection_boxes']
        classes =  output['detection_classes']
        scores = output['detection_scores']

        height, width = image.shape[:2]

        boxes_filter = np.multiply(classes == CLASS_TRAFFIC_LIGHT, scores >= THRESHOLD_SCORE)
        scores = scores[boxes_filter]
        boxes = boxes[boxes_filter]
        boxes_num = len(boxes)
        match = None
        match_color = TrafficLight.UNKNOWN
        state_name = "UNKNOWN"

        if boxes_num > 0:
            x_data = np.zeros((boxes_num, model_data_shape[0], model_data_shape[1], model_data_shape[2]))

            for idx in range(boxes_num):
                cur_box = boxes[idx]
                left_y = int(cur_box[0] * height)
                left_x = int(cur_box[1] * width)
                right_y = int(cur_box[2] * height)
                right_x = int(cur_box[3] * width)

                if TRAINING:
                    cv2.rectangle(img_boxed, (left_x, left_y), (right_x, right_y), (255, 255, 255), 3)

                sub_img = self.get_sub_image(img_rgb, cur_box, model_data_shape)
                sub_img = (sub_img - 128.0) / 255.0

                x_data[idx] = sub_img

            model_predictions = self.cnn_model.model_prediction(self.cnn_session, x_data)

            if TRAINING:
                for idx in range(boxes_num):
                    model_prediction = model_predictions[idx]
                    pred_val = model_prediction[0]
                    pred_color = TrafficLight.RED if pred_val == 0 else TrafficLight.YELLOW if pred_val == 1 else TrafficLight.GREEN if pred_val == 2 else TrafficLight.UNKNOWN

                    cur_box = boxes[idx]
                    left_y = int(cur_box[0] * height)
                    left_x = int(cur_box[1] * width)
                    right_y = int(cur_box[2] * height)
                    right_x = int(cur_box[3] * width)

                    sub_img = self.get_sub_image(image, cur_box, model_data_shape)

                    path = os.path.join(self.training_path_tl, self.training_tl_state_folders[pred_color], '{}_{}.jpg'.format(timestamp, idx))
                    cv2.imwrite(path, sub_img)

            best_prob = -1
            best_val = 3
            for idx in range(boxes_num):
                model_prediction = model_predictions[idx]
                pred_val = model_prediction[0]
                pred_prob = model_prediction[1] * scores[idx]
                if (pred_val <= 2) and (pred_prob > best_prob):
                    match = idx
                    best_val = pred_val
                    best_prob = pred_prob

            match_color = TrafficLight.RED if best_val == 0 else TrafficLight.YELLOW if best_val == 1 else TrafficLight.GREEN if best_val == 2 else TrafficLight.UNKNOWN
            state_name = "RED" if match_color == TrafficLight.RED else "GREEN" if match_color == TrafficLight.GREEN else "YELLOW" if match_color == TrafficLight.YELLOW else "UNKNOWN"

            if TLC_DEBUG:
                rospy.loginfo("Traffic light status: {}".format(state_name))
        else:
            rospy.loginfo("No traffic light detected")

        if TRAINING:
            if match is not None:
                cur_box = boxes[match]
                left_y = int(cur_box[0] * height)
                left_x = int(cur_box[1] * width)
                right_y = int(cur_box[2] * height)
                right_x = int(cur_box[3] * width)

                cv2.rectangle(img_boxed, (left_x, left_y), (right_x, right_y), (0, 0, 255), 3)
                cv2.putText(
                    img_boxed,
                    state_name,
                    (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 255),
                    thickness = 2,
                    lineType = cv2.LINE_AA)

            path = os.path.join(self.training_path_boxed, '{}.jpg'.format(timestamp))
            cv2.imwrite(path, img_boxed)

        return match_color

    def get_sub_image(self, image, box, shape, interpolation = cv2.INTER_CUBIC):
        height, width = image.shape[:2]

        left_y = int(box[0] * height)
        left_x = int(box[1] * width)
        right_y = int(box[2] * height)
        right_x = int(box[3] * width)

        img_w = right_x - left_x
        img_h = right_y - left_y

        max_dim = max(img_w, img_h)

        img_w2 = img_w // 2
        img_h2 = img_h // 2
        max_dim2 = max_dim // 2

        sub_img_y = max(0, left_y + img_h2 - max_dim2)
        sub_img_x = max(0, left_x + img_w2 - max_dim2)

        sub_img = image[sub_img_y:(sub_img_y + max_dim), sub_img_x:(sub_img_x + max_dim)]
        sub_img_height, sub_img_width = sub_img.shape[:2]

        if (sub_img_height != max_dim) or (sub_img_width != max_dim) or True:
            sub_img_new = np.zeros((max_dim, max_dim, sub_img.shape[2]), dtype=np.uint8)

            sub_img_new_y = max(0, (max_dim - sub_img_height) // 2)
            sub_img_new_x = max(0, (max_dim - sub_img_width) // 2)
            sub_img_new[sub_img_new_y:(sub_img_new_y + sub_img_height), sub_img_new_x:(sub_img_new_x + sub_img_width)] = sub_img
            sub_img = sub_img_new
        
        sub_img = cv2.resize(sub_img, (shape[1], shape[0]), interpolation = interpolation)

        return sub_img
    
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

    def color_correction(self, image, percent, channel_idx = -1):
        half_percent = percent / 200.0
        height, width, channels_num = image.shape[:3]
        vec_size = width * height

        image_res = image.copy()

        for idx in range(channels_num):
            if (channel_idx < 0) or (idx == channel_idx):
                channel = image_res[:,:,idx]
                intens_sorted = np.sort(channel.reshape(vec_size))
                intens_num = intens_sorted.shape[0] - 1

                low_val  = intens_sorted[int(math.floor(intens_num * half_percent))]
                high_val = intens_sorted[int(math.ceil(intens_num * (1.0 - half_percent)))]

                channel[channel < low_val] = low_val
                channel[channel > high_val] = high_val

                channel = cv2.normalize(channel, channel.copy(), alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)

                image_res[:,:,idx] = channel

        return image_res

    def gamma_correction(self, image):
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        intens = image_lab[:,:,0] / 255.0
        lum_avg = np.average(intens)

        try:
            if (lum_avg > 0) and (lum_avg < 1):
                gamma = -1.0 / np.log2(lum_avg)
            else:
                gamma = 1.0
        except:
            gamma = 1.0

        image_lab[:,:,0] = np.power(intens, gamma) * 255.0

        image_res = cv2.cvtColor(image_lab, cv2.COLOR_LAB2BGR)

        return image_res

    def apply_clahe(self, image):
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        image_lab[:,:,0] = self.hist_clahe.apply(image_lab[:,:,0])
        image_res = cv2.cvtColor(image_lab, cv2.COLOR_LAB2BGR)

        return image_res
