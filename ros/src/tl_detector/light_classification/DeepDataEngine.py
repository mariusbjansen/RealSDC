import numpy as np
import cv2
import pickle
import csv
import os
import matplotlib.image as mpimg

class DeepDataEngine:
    """
    Data engine.
    Main purpose - work with augmented data amounts of any size, create it and feed it to leaning and validation process
    """

    def __init__(
        self,
        set_name,
        storage_dir = './deep_storage',
        mem_size = 64 * 1024 * 1024,
        batch_size = 128):

        self.set_name = set_name
        self.storage_dir = storage_dir
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.features = None
        self.labels = None
        self.descriptions = None
        self.storage_files = []
        self.storage_file_active = -1
        self.storage_buf_x = None
        self.storage_buf_y = None
        self.storage_buf_pos = 0
        self.data_depth = 3 # Depends on storage pre-processing algorithm

    def isVirtual(self):
        return False

    def loadDescriptionsFromFile(self, file_path):
        self.descriptions = {}

        n_classes = -1
        with open(file_path) as csvfile:
            reader = csv.DictReader(csvfile)
        
            for row in reader:
                curClass = int(row['ClassId'])
                n_classes = max(n_classes, curClass)
                self.descriptions[curClass] = row['Description']

        n_classes += 1
        for curClass in range(n_classes):
            if not(curClass in self.descriptions):
                self.descriptions[curClass] = 'Class {}'.format(curClass)

    def _unpickleFromFile(self, file_path):
        with open(file_path, mode='rb') as f:
            data_set = pickle.load(f)
    
        X_data, y_data = data_set['features'], data_set['labels']

        assert(X_data.shape[0] == y_data.shape[0])

        return X_data, y_data

    def _pickleToFile(self, file_path, X_data, y_data):
        with open(file_path, mode='wb') as f:
            data_set = {'features' : X_data, 'labels' : y_data}
            pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)

    def loadDataFromFile(self, file_path):
        self.features, self.labels = self._unpickleFromFile(file_path)

    def loadDataFromImageSet(self, dir_path, img_width = 32, img_height = 32):
        images_list = os.listdir(dir_path)

        x_data = []
        y_data = []

        for image_name in images_list:
            idx = image_name.find('_')
            if idx > 0:
                img_class = int(image_name[:idx])

                image_path = dir_path + '/' + image_name
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                image = cv2.resize(image, (img_height, img_width), interpolation = cv2.INTER_AREA)
                
                x_data += [image]
                y_data += [img_class]

        self.features = np.reshape(x_data, (-1, 32, 32, 3))
        self.labels = np.reshape(y_data, (-1))

        assert(self.features.shape[0] == self.labels.shape[0])

    def saveDataToImages(self, dir_path, img_format = 'png'):
        try:
            os.makedirs(dir_path)
        except:
            pass

        cnt = self.features.shape[0]
        for idx in range(cnt):
            mpimg.imsave('{}/{:0>3}_{:0>6}.{}'.format(dir_path, self.labels[idx], idx, img_format), self.features[idx], format = img_format)

    def getData(self):
        return self.features, self.labels

    def getDataSize(self):
        return self.features.shape[0]

    def getImageShape(self):
        return self.features.shape[1::]

    def getDataShape(self):
        return (self.features.shape[1], self.features.shape[2], self.data_depth)

    def getClassesNum(self):
        return np.max(self.labels) + 1

    def getDescriptions(self):
        return self.descriptions

    def getDataStatistic(self, samples = 5):
        permutation = np.random.permutation(self.features.shape[0])
        x_data = self.features[permutation]
        y_data = self.labels[permutation]

        n_classes = np.max(y_data) + 1
        stat_classes = np.zeros(n_classes, dtype = int)
        samples_dict = {idx : [] for idx in range(n_classes)}
        for idx in range(x_data.shape[0]):
            curClass = y_data[idx]
            stat_classes[curClass] += 1
            
            dict_smpl_list = samples_dict[curClass]
            
            if len(dict_smpl_list) < samples:
                dict_smpl_list += [x_data[idx]]

        return stat_classes, samples_dict

    def clearData(self):
        self.features = None
        self.labels = None

    def _loadStorage(self):
        self.storage_files = []
        self.storage_file_active = -1

        set_file_base_name = self.set_name + '_';

        try:
            os.makedirs(self.storage_dir)
        except:
            pass

        try:
            for file_name in os.listdir(self.storage_dir):
                file_path = self.storage_dir + '/' + file_name
                if (os.path.exists(file_path) and
                    os.path.isfile(file_path) and
                    (str(os.path.splitext(file_path)[1]).upper() in ('.DAT')) and
                    (str(file_name[:len(set_file_base_name)]).upper() == str(set_file_base_name).upper())):
                    
                    self.storage_files += [file_path]

        except:
            pass

    def _delete_storage(self):
        for file_name in self.storage_files:
            try:
                os.remove(file_name)
            except:
                pass

        self.storage_files = []

    def _pre_transform_image(self, img):
        data_width = img.shape[1]
        data_height = img.shape[0]

        center_col = np.random.uniform(float(data_width) / 3.0, 2.0 * (float(data_width) / 3.0))
        center_row = np.random.uniform(float(data_height) / 3.0, 2.0 * (float(data_height) / 3.0))
        rot_angle = np.random.uniform(-30.0, 30.0)
        scale_factor = np.random.uniform(-0.3, 0.3)
        affineM = cv2.getRotationMatrix2D((center_col, center_row), rot_angle, 1.0 + scale_factor)

        img_proc = cv2.warpAffine(img, affineM, (data_width, data_height), borderMode = cv2.BORDER_REFLECT)
        
        return img_proc

    def _pre_process_image(self, img):
        img_norm = np.zeros_like(img, dtype = float)
        img_norm[:] = img[:]
        img_norm = (img_norm - 128.0) / 255.0

        return img_norm

    def _create_storage(self, class_samples):
        try:
            os.makedirs(self.storage_dir)
        except:
            pass

        data_shape = self.features.shape
        n_classes = np.max(self.labels) + 1

        data_size = data_shape[0]
        data_height = data_shape[1]
        data_width = data_shape[2]

        buf_size = int(self.mem_size / (data_height * data_width * self.data_depth * 8))

        x_buf = np.zeros((buf_size, data_height, data_width, self.data_depth))
        y_buf = np.zeros(buf_size, dtype = self.labels.dtype)

        samples_cnt = np.zeros(n_classes, dtype = int)
        samples_idx = {idx : [] for idx in range(n_classes)}
        for idx in range(data_size):
            curClass = self.labels[idx]
            samples_cnt[curClass] += 1
            class_idx = samples_idx[curClass]
            class_idx += [idx]

        class_size = max(np.max(samples_cnt), class_samples)

        generation_plan = []
        for curClass in range(n_classes):
            if samples_cnt[curClass] > 0:
                class_idx = samples_idx[curClass]

                cnt = 0
                isFirstPass = True
                while isFirstPass or (cnt < class_size):
                    np.random.shuffle(class_idx)

                    for idx in class_idx:
                        generation_plan += [(idx, isFirstPass)]
                        
                        cnt += 1

                        if (not isFirstPass) and (cnt >= class_size):
                            break

                    isFirstPass = False

                    if class_samples < 0:
                        break

        np.random.shuffle(generation_plan)
        
        file_idx = 0
        buf_pos = 0

        for idx, isFirstPass in generation_plan:
            if isFirstPass:
                img_proc = self.features[idx]
            else:
                img_proc = self._pre_transform_image(self.features[idx])

            img_proc = self._pre_process_image(img_proc)
            
            for img_depth_idx in range(min(self.data_depth, len(img_proc))):
                x_buf[buf_pos, :, :, img_depth_idx] = img_proc[:, :, img_depth_idx]

            y_buf[buf_pos] = self.labels[idx]
                        
            buf_pos += 1

            if buf_pos >= buf_size:
                self._pickleToFile('{}/{}_{:0>6}.dat'.format(self.storage_dir, self.set_name, file_idx), x_buf, y_buf)
                file_idx += 1
                buf_pos = 0

        if buf_pos > 0:
            x_buf = x_buf[:buf_pos]
            y_buf = y_buf[:buf_pos]
            self._pickleToFile('{}/{}_{:0>6}.dat'.format(self.storage_dir, self.set_name, file_idx), x_buf, y_buf)

    def initStorage(self, override = False, base_file_path = '', class_samples = -1):
        self._loadStorage()

        if override or (len(self.storage_files) <= 0):
            if len(base_file_path) > 0:
                self.loadDataFromFile(base_file_path)

            assert(len(self.features) > 0)
            assert(len(self.labels) > 0)

            self._delete_storage()

            self._create_storage(class_samples)

            self._loadStorage()

    def _readNextStorageFile(self):
        self.storage_buf_x, self.storage_buf_y = self._unpickleFromFile(self.storage_files[self.storage_file_active])
        permutation = np.random.permutation(self.storage_buf_x.shape[0])
        self.storage_buf_x = self.storage_buf_x[permutation]
        self.storage_buf_y = self.storage_buf_y[permutation]
        self.storage_buf_pos = 0

    def initRead(self):
        if (len(self.storage_files) == 1) and (self.storage_file_active == 0):
            permutation = np.random.permutation(self.storage_buf_x.shape[0])
            self.storage_buf_x = self.storage_buf_x[permutation]
            self.storage_buf_y = self.storage_buf_y[permutation]
            self.storage_buf_pos = 0
        else:
            np.random.shuffle(self.storage_files)
            self.storage_file_active = 0
            self._readNextStorageFile()

            while self.storage_buf_pos >= self.storage_buf_x.shape[0]:
                if (self.storage_file_active + 1) < len(self.storage_files):
                    self.storage_file_active += 1
                    self._readNextStorageFile()
                else:
                    break

    def canReadMore(self):
        if self.storage_buf_pos < self.storage_buf_x.shape[0]:
            return True

        return False

    def readNext(self):
        x_data = self.storage_buf_x[self.storage_buf_pos:self.storage_buf_pos + self.batch_size]
        y_data = self.storage_buf_y[self.storage_buf_pos:self.storage_buf_pos + self.batch_size]

        self.storage_buf_pos += len(x_data)

        try_read_next = True

        while try_read_next:
            try_read_next = False

            if self.storage_buf_pos >= self.storage_buf_x.shape[0]:
                if (self.storage_file_active + 1) < len(self.storage_files):
                    self.storage_file_active += 1
                    self._readNextStorageFile()

                    if self.storage_buf_pos < self.storage_buf_x.shape[0]:
                        if len(x_data) <= 0:
                            x_data = self.storage_buf_x[self.storage_buf_pos:self.storage_buf_pos + self.batch_size]
                            y_data = self.storage_buf_y[self.storage_buf_pos:self.storage_buf_pos + self.batch_size]

                            self.storage_buf_pos += len(x_data)
                        elif len(x_data) < self.batch_size:
                            size_orig = len(x_data)
                            batch_remain = self.batch_size - size_orig
                            x_data = np.append(x_data, self.storage_buf_x[self.storage_buf_pos:self.storage_buf_pos + batch_remain], axis = 0)
                            y_data = np.append(y_data, self.storage_buf_y[self.storage_buf_pos:self.storage_buf_pos + batch_remain], axis = 0)

                            self.storage_buf_pos += len(x_data) - size_orig

                    if self.storage_buf_pos >= self.storage_buf_x.shape[0]:
                        try_read_next = True

        return x_data, y_data

    def saveStorageToImages(self, dir_path, plane = -1, img_format = 'png'):
        try:
            os.makedirs(dir_path)
        except:
            pass

        self.initRead()

        total_idx = 0
        while (self.canReadMore()):
            x_data, y_data = self.readNext()
            samples = x_data.shape[0]
            planes = x_data.shape[3]
            for idx in range(samples):
                for plane_idx in range(planes):
                    if (plane < 0) or (plane == plane_idx):
                        mpimg.imsave('{}/{:0>3}_{:0>6}_{}.{}'.format(dir_path, y_data[idx], total_idx, plane_idx, img_format), x_data[idx, :, :, plane_idx], cmap='Greys_r', format = img_format)

                total_idx += 1
