import numpy as np
import os
import rospkg
from DeepDataEngine import DeepDataEngine
from DeepModelEngine import DeepModelEngineV3

tl_path = os.path.join(rospkg.RosPack().get_path("tl_detector"), 'light_classification')
images_path = os.path.join(tl_path, 'train_images')
descriptions_file = os.path.join(tl_path, 'lightsname.csv')
storage_dir = os.path.join(tl_path, 'deep_storage')
model_dir = os.path.join(tl_path, 'deep_model')

data_test = DeepDataEngine('test', storage_dir = storage_dir)
data_test.loadDescriptionsFromFile(descriptions_file)
data_test.initStorage(override = False)

model = DeepModelEngineV3(
    storage_dir = model_dir,
    data_shape = (32, 32, 3),
    class_num = 4,
    model_depth = 2)
session = model.load_model()

data_test.initRead()

if data_test.canReadMore():
    x_data, y_data = data_test.readNext()
    weights = np.ones_like(y_data)

    predictions = model.model_prediction(session, x_data)
    for pred in predictions:
        pred_val = pred[0]
        pred_prob = pred[1]
        print("{} - {:.2f}".format(pred_val, pred_prob))

    pred_weighted = model.model_prediction_weighted(session, x_data, weights)
    print("Best: {}".format(pred_weighted))
