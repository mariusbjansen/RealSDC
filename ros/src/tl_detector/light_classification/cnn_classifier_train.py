import numpy as np
import os
import rospkg
from DeepDataEngine import DeepDataEngine
from DeepModelEngine import DeepModelEngineV3

tl_path = os.path.join(rospkg.RosPack().get_path("tl_detector"), 'light_classification')
train_images_path = os.path.join(tl_path, 'train_images')
valid_images_path = os.path.join(tl_path, 'valid_images')
test_images_path = os.path.join(tl_path, 'test_images')
descriptions_file = os.path.join(tl_path, 'lightsname.csv')
storage_dir = os.path.join(tl_path, 'deep_storage')
model_dir = os.path.join(tl_path, 'deep_model')
valid_set_percent = 30.0
test_set_percent = 10.0
augmented_class_size = 3000

def load_base_data():
    """
    Load pickled base data
    """
    
    data_train = DeepDataEngine('train', storage_dir = storage_dir)
    data_train.loadDataFromImageSet(train_images_path)
    data_train.loadDescriptionsFromFile(descriptions_file)

    data_valid = DeepDataEngine('valid', storage_dir = storage_dir)
    data_valid.loadDataFromImageSet(valid_images_path)
    data_valid.loadDescriptionsFromFile(descriptions_file)

    data_test = DeepDataEngine('test', storage_dir = storage_dir)
    data_test.loadDataFromImageSet(test_images_path)
    data_test.loadDescriptionsFromFile(descriptions_file)
    
    # features_num = len(data_train.features)
    # permutation = np.random.permutation(features_num)
    # data_train.features = data_train.features[permutation]
    # data_train.labels = data_train.labels[permutation]
    
    # valid_set_num = int((valid_set_percent / 100.0) * features_num)
    # test_set_num =  int(((valid_set_percent + test_set_percent) / 100.0) * features_num) - valid_set_num
    
    # data_valid = DeepDataEngine('valid', storage_dir = storage_dir)
    # data_valid.features = data_train.features[:valid_set_num]
    # data_valid.labels = data_train.labels[:valid_set_num]
    # data_valid.loadDescriptionsFromFile(descriptions_file)
    
    # data_test = DeepDataEngine('test', storage_dir = storage_dir)
    # data_test.features = data_train.features[valid_set_num:(valid_set_num + test_set_num)]
    # data_test.labels = data_train.labels[valid_set_num:(valid_set_num + test_set_num)]
    # data_test.loadDescriptionsFromFile(descriptions_file)
    
    # data_train.features = data_train.features[(valid_set_num + test_set_num):]
    # data_train.labels = data_train.labels[(valid_set_num + test_set_num):]

    return data_train, data_valid, data_test

def print_data_information(data_train, data_valid, data_test):
    n_train = data_train.getDataSize()
    n_valid = data_valid.getDataSize()
    n_test = data_test.getDataSize()
    
    image_shape = data_train.getImageShape()
    n_classes = data_train.getClassesNum()
    
    
    print("Number of training examples =", n_train)
    print("Number of validation examples =", n_valid)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

data_train, data_valid, data_test = load_base_data()
print_data_information(data_train, data_valid, data_test)

data_train.initStorage(override = True, class_samples = augmented_class_size)
data_valid.initStorage(override = True, class_samples = int(augmented_class_size * (valid_set_percent / 100.0)))
data_test.initStorage(override = True, class_samples = int(augmented_class_size * (test_set_percent / 100.0)))

print('Base data set was pre-processed and augmented storage was created.')

data_shape = data_train.getDataShape()
class_num = data_train.getClassesNum()

model = DeepModelEngineV3(
    storage_dir = model_dir,
    data_shape = data_shape,
    class_num = class_num,
    model_depth = 2)

model.train_model(
    data_train, data_valid,
    learn_rate_from = 0.003, learn_rate_to = 0.0005,
    keep_prob_from = 0.5, keep_prob_to = 0.5,
    reg_factor = 0.0001,
    epochs = 75,
    train_rounds = 6,
    continue_training = False,
    verbose = True)

print("Model is trained.")
print("Training data accuracy: {:.2f}%".format(model.validate_model(data_train) * 100))
print("Validation data accuracy: {:.2f}%".format(model.validate_model(data_valid) * 100))
print("Test data accuracy: {:.2f}%".format(model.validate_model(data_test) * 100))