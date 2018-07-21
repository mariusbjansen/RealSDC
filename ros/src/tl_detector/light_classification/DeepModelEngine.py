import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten

class DeepModelEngine:
    def __init__(
        self,
        data_shape,
        class_num,
        model_depth = 1,
        storage_dir = './deep_model',
        storage_file_name = 'deep_model',
        normal_mean = 0.0,
        normal_sigma = 0.1):
        
        self.data_shape = data_shape
        self.class_num = class_num
        self.model_depth = model_depth
        self.storage_dir = storage_dir
        self.storage_file_name = storage_file_name
        self.normal_mean = normal_mean
        self.normal_sigma = normal_sigma

        self._init_model_process_graph()

    def _get_model_structure(self):
        raise NotImplementedError

    def _init_model_process_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.data_x = tf.placeholder(tf.float32, (None, self.data_shape[0], self.data_shape[1], self.data_shape[2]), name = 'data_x')
            self.data_y = tf.placeholder(tf.int32, (None), name = 'data_y')
            self.tf_keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name = 'tf_keep_prob')
            self.tf_decay_learn_rate = tf.placeholder(dtype=tf.float32, shape=(), name = 'tf_decay_learn_rate')
            self.tf_reg_factor = tf.placeholder(dtype=tf.float32, shape=(), name = 'tf_reg_factor')
            self.one_hot_y = tf.one_hot(self.data_y, self.class_num, name = 'one_hot_y')

            self.logits, self.model_weights_reg, self.model_weights_noreg, self.model_conv_layers = self._get_model_structure()
            self.soft_max = tf.nn.softmax(logits = self.logits, name = 'soft_max')
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = self.one_hot_y, logits = self.logits, name = 'cross_entropy')
            self.loss_operation = tf.reduce_mean(self.cross_entropy, name = 'loss_operation')

            reg_idx = 0
            for l_weight in self.model_weights_reg:
                self.loss_operation = tf.add(self.loss_operation, tf.scalar_mul(self.tf_reg_factor, tf.nn.l2_loss(l_weight)), name = 'loss_operation_reg_{}'.format(reg_idx))

                reg_idx += 1
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate = self.tf_decay_learn_rate, name = 'adam_optimizer')
            self.training_operation = self.optimizer.minimize(self.loss_operation, name = 'training_operation')

            self.logits_prediction = tf.argmax(self.logits, 1, name = 'logits_prediction')
            self.correct_value = tf.argmax(self.one_hot_y, 1, name = 'correct_value')
            self.correct_prediction = tf.equal(self.logits_prediction, self.correct_value, name = 'correct_prediction')
            self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name = 'accuracy_operation')

    def _model_evaluate(self, data_valid, session):
        num_examples = 0
        total_accuracy = 0
        total_loss = 0

        data_valid.initRead()
        while data_valid.canReadMore():
            x_data, y_data = data_valid.readNext()
            accuracy = session.run(self.accuracy_operation, feed_dict={self.data_x: x_data, self.data_y: y_data, self.tf_keep_prob : 1.0})
            loss = session.run(self.cross_entropy, feed_dict={self.data_x: x_data, self.data_y: y_data, self.tf_keep_prob : 1.0})
                    
            num_examples += len(x_data)
            total_accuracy += accuracy * len(x_data)
            total_loss += sum(loss) * len(x_data)
                    
        if num_examples <= 0:
            total_accuracy = 0
            total_loss = 0
            num_examples = 1

        return (float(total_accuracy) / float(num_examples), float(total_loss) / float(num_examples))

    def _save_model(self, session):
        saver = tf.train.Saver(save_relative_paths = True)
        saver.save(session, self.storage_dir + '/' + self.storage_file_name)

    def _load_model(self, session):
        saver = tf.train.Saver()
        saver.restore(session, tf.train.latest_checkpoint(self.storage_dir))

    def train_model(
        self,
        data_train, data_valid,
        learn_rate_from = 0.003, learn_rate_to = 0.0005,
        keep_prob_from = 0.5, keep_prob_to = 0.5,
        reg_factor = 0.0001,
        epochs = 75,
        train_rounds = 3,
        continue_training = False,
        verbose = True):

        reg_factor_val = reg_factor * (1.0 / (self.model_depth ** 2))

        load_saved_model = continue_training

        for train_rnd_num in range(train_rounds):
            with self.graph.as_default():
                with tf.Session() as session:
                    if load_saved_model:
                        self._load_model(session)
                        best_accuracy, best_loss = self._model_evaluate(data_valid, session)
                    else:
                        session.run(tf.global_variables_initializer())
                        best_accuracy = 0
                        best_loss = 0

                    if verbose:
                        print("Training ...")
                        print()

                    for i in range(epochs):
                        decay_learn_rate = learn_rate_from * ((float(learn_rate_to) / float(learn_rate_from)) ** (float(i) / float(epochs)))
                        decay_keep_prob = keep_prob_from * ((float(keep_prob_to) / float(keep_prob_from)) ** (float(i) / float(epochs)))

                        if verbose:
                            print("ROUND {}:".format(train_rnd_num + 1))
                            print("EPOCH {}:".format(i + 1))
                            print("Learn rate = {}:".format(decay_learn_rate))
                            print("Keep prob = {}:".format(decay_keep_prob))

                        data_train.initRead()
                        while data_train.canReadMore():
                            x_data, y_data = data_train.readNext()

                            session.run(
                                self.training_operation,
                                feed_dict = {
                                    self.data_x: x_data,
                                    self.data_y: y_data,
                                    self.tf_keep_prob: decay_keep_prob,
                                    self.tf_decay_learn_rate: decay_learn_rate,
                                    self.tf_reg_factor: reg_factor_val})
            
                        validation_accuracy, validation_loss = self._model_evaluate(data_valid, session)
                    
                        if verbose:
                            print("    Validation Accuracy = {:.2f}%".format(validation_accuracy * 100))
                            print("    Validation Loss = {:.5f}".format(validation_loss))

                        if (validation_accuracy > best_accuracy) or ((validation_accuracy == best_accuracy) and (validation_loss < best_loss)):
                            best_accuracy = validation_accuracy
                            best_loss = validation_loss
                            self._save_model(session)

                            if verbose:
                                print("    Accuracy is improved. Model is saved.")

                        if verbose:
                            print()
            
            load_saved_model = True

        if verbose:
            print("Best accuracy = {:.2f}%".format(best_accuracy * 100))
            print()

        return best_accuracy

    def validate_model(self, data_valid, verbose = False):
        with self.graph.as_default():
            with tf.Session() as session:
                self._load_model(session)

                validation_accuracy, validation_loss = self._model_evaluate(data_valid, session)

                if verbose:
                    print("Accuracy = {:.2f}%".format(validation_accuracy * 100))
                    print()

        return validation_accuracy

    def repack_model(self):
        with self.graph.as_default():
            with tf.Session() as session:
                self._load_model(session)
                self._save_model(session)

    def model_precision_recall(self, data_valid):
        precision_recall_dict = {idx : (0, 0, 0) for idx in range(self.class_num)}
        
        with self.graph.as_default():
            with tf.Session() as session:
                self._load_model(session)

                data_valid.initRead()
                while data_valid.canReadMore():
                    x_data, y_data = data_valid.readNext()

                    logits_prediction_val = session.run(self.logits_prediction, feed_dict={self.data_x: x_data, self.tf_keep_prob : 1.0})

                    data_cnt = len(x_data)
                    for i in range(data_cnt):
                        pos, neg, rel = precision_recall_dict[y_data[i]]
                        rel += 1

                        if y_data[i] == logits_prediction_val[i]:
                            pos += 1
                        else:
                            pos_n, neg_n, rel_n = precision_recall_dict[logits_prediction_val[i]]
                            neg_n += 1

                            precision_recall_dict[logits_prediction_val[i]] = (pos_n, neg_n, rel_n)

                        precision_recall_dict[y_data[i]] = (pos, neg, rel)

        precision_recall_dict_ext = {}
        for idx in range(self.class_num):
            pos, neg, rel = precision_recall_dict[idx]
            if (pos + neg) > 0:
                prec = float(pos) / float(pos + neg)
            else:
                prec = 0.0

            if rel > 0:
                rec = float(pos) / float(rel)
            else:
                rec = 0.0

            precision_recall_dict_ext[idx] = (pos, neg, rel, prec, rec)

        return precision_recall_dict_ext

    def get_batch_prediction(self, data_valid, init_read = True):
        prediction_data = []
        
        with self.graph.as_default():
            with tf.Session() as session:
                self._load_model(session)

                if init_read:
                    data_valid.initRead()

                if data_valid.canReadMore():
                    x_data, y_data = data_valid.readNext()

                    logits_prediction_val = session.run(self.logits_prediction, feed_dict={self.data_x: x_data, self.tf_keep_prob : 1.0})

                    data_cnt = len(x_data)
                    for i in range(data_cnt):
                        prediction_data += [(x_data[i], y_data[i], logits_prediction_val[i])]

        return prediction_data

    def get_batch_softmax(self, data_valid, top_values = 5, init_read = True):
        softmax_top_data = []
        
        with self.graph.as_default():
            with tf.Session() as session:
                self._load_model(session)

                if init_read:
                    data_valid.initRead()

                if data_valid.canReadMore():
                    x_data, y_data = data_valid.readNext()

                    soft_max_top = tf.nn.top_k(self.soft_max, top_values)
                    softmax_top_val = session.run(soft_max_top, feed_dict={self.data_x: x_data, self.tf_keep_prob : 1.0})

                    val, ind = (softmax_top_val.values, softmax_top_val.indices)
                    data_cnt = len(x_data)
                    for i in range(data_cnt):
                        softmax_top_data += [(x_data[i], y_data[i], val[i], ind[i])]

        return softmax_top_data

    def get_batch_conv_activations(self, data_valid, init_read = True):
        activations_data = []
        
        with self.graph.as_default():
            with tf.Session() as session:
                self._load_model(session)

                if init_read:
                    data_valid.initRead()

                if data_valid.canReadMore():
                    x_data, y_data = data_valid.readNext()

                    logits_prediction_val = session.run(self.logits_prediction, feed_dict={self.data_x: x_data, self.tf_keep_prob : 1.0})

                    layer_act = []
                    for l_conv_layer in self.model_conv_layers:
                        layer_act += [session.run(l_conv_layer, feed_dict={self.data_x: x_data, self.tf_keep_prob : 1.0})]

                    data_cnt = len(x_data)
                    for i in range(data_cnt):
                        curLayerAct = []
                        for layer_act_w in layer_act:
                            curLayerAct += [layer_act_w[i]]

                        activations_data += [(x_data[i], y_data[i], logits_prediction_val[i], curLayerAct)]

        return activations_data

    def load_model(self):
        with self.graph.as_default():
            session = tf.Session()
            self._load_model(session)
        return session

    def model_prediction(self, session, x_data):
        predictions = []
        with self.graph.as_default():
            pred_probs = session.run(self.soft_max, feed_dict={self.data_x: x_data, self.tf_keep_prob : 1.0})
            for pred_prob in pred_probs:
                prediction = np.argmax(pred_prob)
                probability = pred_prob[prediction]
                predictions.append([prediction, probability])

        return predictions

    def model_prediction_weighted(self, session, x_data, weights):
        best_pred = 0
        best_prob = 0

        with self.graph.as_default():
            pred_probs = session.run(self.soft_max, feed_dict={self.data_x: x_data, self.tf_keep_prob : 1.0})
            for idx in range(len(pred_probs)):
                pred_prob = pred_probs[idx]
                prediction = np.argmax(pred_prob)
                probability = pred_prob[prediction] * weights[idx]
                if probability > best_prob:
                    best_pred = prediction
                    best_prob = probability

        return best_pred

class DeepModelEngineV3(DeepModelEngine):
    def __init__(
        self,
        data_shape,
        class_num,
        model_depth = 1,
        storage_dir = './deep_model',
        storage_file_name = 'deep_model',
        normal_mean = 0.0,
        normal_sigma = 0.1):

        DeepModelEngine.__init__(
            self,
            data_shape,
            class_num,
            model_depth = model_depth,
            storage_dir = storage_dir,
            storage_file_name = storage_file_name,
            normal_mean = normal_mean,
            normal_sigma = normal_sigma)

    def _get_model_structure(self):
        # Layer 1: Convolutional. Input = 32x32ximg_depth. Output = 30x30x16*model_depth.
        self.mdl_conv1_weights = tf.Variable(tf.truncated_normal([3, 3, self.data_shape[2], 16 * self.model_depth], mean = self.normal_mean, stddev = self.normal_sigma, dtype = tf.float32, name = 'conv1_weights_norm'), name = 'conv1_weights')
        self.mdl_conv1_biases = tf.Variable(tf.zeros(16 * self.model_depth, dtype = tf.float32, name = 'conv1_biases_zoros'), name = 'conv1_biases')
        self.mdl_conv1 = tf.nn.bias_add(tf.nn.conv2d(self.data_x, self.mdl_conv1_weights, [1, 1, 1, 1], padding = "VALID", name = 'conv1_conv2d'), self.mdl_conv1_biases, name = 'conv1_conv2d_bias')
    
        # Activation.
        self.mdl_l1_act = tf.nn.relu(self.mdl_conv1, name = 'l1_act_relu')
    
        # Pooling. Input = 30x30x16 * model_depth. Output = 15x15x16 * model_depth.
        self.mdl_l1_pool = tf.nn.avg_pool(self.mdl_l1_act, [1, 2, 2, 1], [1, 2, 2, 1], padding = "VALID", name = 'l1_pool')

        # Layer 2: Convolutional. Output = 12x12x24 * model_depth.
        self.mdl_conv2_weights = tf.Variable(tf.truncated_normal([4, 4, 16 * self.model_depth, 24 * self.model_depth], mean = self.normal_mean, stddev = self.normal_sigma, dtype = tf.float32, name = 'conv2_weights_norm'), name = 'conv2_weights')
        self.mdl_conv2_biases = tf.Variable(tf.zeros(24 * self.model_depth, dtype = tf.float32, name = 'conv2_biases_zero'), name = 'conv2_biases')
        self.mdl_conv2 = tf.nn.bias_add(tf.nn.conv2d(self.mdl_l1_pool, self.mdl_conv2_weights, [1, 1, 1, 1], padding = "VALID", name = 'conv2_conv2d'), self.mdl_conv2_biases, name = 'conv2_conv2d_bias')
    
        # Activation.
        self.mdl_l2_act = tf.nn.relu(self.mdl_conv2, name = 'l2_act_relu')

        # Pooling. Input = 12x12x24 * model_depth. Output = 6x6x24 * model_depth.
        self.mdl_l2_pool = tf.nn.avg_pool(self.mdl_l2_act, [1, 2, 2, 1], [1, 2, 2, 1], padding = "VALID", name = 'l2_pool')

        # Layer 3: Convolutional. Output = 4x4x32 * model_depth.
        self.mdl_conv3_weights = tf.Variable(tf.truncated_normal([3, 3, 24 * self.model_depth, 32 * self.model_depth], mean = self.normal_mean, stddev = self.normal_sigma, dtype = tf.float32, name = 'conv3_weights_norm'), name = 'conv3_weights')
        self.mdl_conv3_biases = tf.Variable(tf.zeros(32 * self.model_depth, dtype = tf.float32, name = 'conv3_biases_zero'), name = 'conv3_biases')
        self.mdl_conv3 = tf.nn.bias_add(tf.nn.conv2d(self.mdl_l2_pool, self.mdl_conv3_weights, [1, 1, 1, 1], padding = "VALID", name = 'conv3_conv2d'), self.mdl_conv3_biases, name = 'conv3_conv2d_bias')
    
        # Activation.
        self.mdl_l3_act = tf.nn.dropout(tf.nn.relu(self.mdl_conv3, name = 'l3_act_relu'), self.tf_keep_prob, name = 'l3_act_relu_dt')

        # Flatten. Input = 4x4x32 * model_depth. Output = 512 * model_depth.
        self.mdl_l3_flat = flatten(self.mdl_l3_act)
    
        # Layer 4: Fully Connected. Input = 512 * model_depth. Output = 150 * model_depth.
        self.mdl_l4_weights = tf.Variable(tf.truncated_normal([512 * self.model_depth, 150 * self.model_depth], mean = self.normal_mean, stddev = self.normal_sigma, dtype = tf.float32, name = 'l4_weights_norm'), name = 'l4_weights')
        self.mdl_l4_biases = tf.Variable(tf.zeros(150 * self.model_depth, dtype = tf.float32, name = 'l4_biases_zero'), name = 'l4_biases')
        self.mdl_l4 = tf.nn.bias_add(tf.matmul(self.mdl_l3_flat, self.mdl_l4_weights, name = 'l4_matmul'), self.mdl_l4_biases, name = 'l4_matmul_bias')
    
        # Activation.
        self.mdl_l4_act = tf.nn.dropout(tf.nn.relu(self.mdl_l4, name = 'l4_act_relu'), self.tf_keep_prob, name = 'l4_act_relu_dt')

        # Layer 5: Fully Connected. Input = 150 * model_depth. Output = 100 * model_depth.
        self.mdl_l5_weights = tf.Variable(tf.truncated_normal([150 * self.model_depth, 100 * self.model_depth], mean = self.normal_mean, stddev = self.normal_sigma, dtype = tf.float32, name = 'l5_weights_norm'), name = 'l5_weights')
        self.mdl_l5_biases = tf.Variable(tf.zeros(100 * self.model_depth, dtype = tf.float32, name = 'l5_biases_zero'), name = 'l5_biases')
        self.mdl_l5 = tf.nn.bias_add(tf.matmul(self.mdl_l4_act, self.mdl_l5_weights, name = 'l5_matmul'), self.mdl_l5_biases, name = 'l5_matmul_bias')
    
        # Activation.
        self.mdl_l5_act = tf.nn.dropout(tf.nn.relu(self.mdl_l5, name = 'l5_act_relu'), self.tf_keep_prob, name = 'l5_act_relu_dt')

        # Layer 6: Fully Connected. Input = 100 * model_depth. Output = class_num.
        self.mdl_l6_weights = tf.Variable(tf.truncated_normal([100 * self.model_depth, self.class_num], mean = self.normal_mean, stddev = self.normal_sigma, dtype = tf.float32, name = 'l6_weights_norm'), name = 'l6_weights')
        self.mdl_l6_biases = tf.Variable(tf.zeros(self.class_num, dtype = tf.float32, name = 'l6_biases_zero'), name = 'l6_biases')
        logits = tf.nn.bias_add(tf.matmul(self.mdl_l5_act, self.mdl_l6_weights, name = 'l6_matmul'), self.mdl_l6_biases, name = 'l6_matmul_bias')

        model_weights_reg = [self.mdl_l4_weights, self.mdl_l5_weights, self.mdl_l6_weights]
        model_weights_noreg = [self.mdl_conv1_weights, self.mdl_conv2_weights, self.mdl_conv3_weights]
        model_conv_layers = [self.mdl_conv1, self.mdl_conv2, self.mdl_conv3]
    
        return logits, model_weights_reg, model_weights_noreg, model_conv_layers
