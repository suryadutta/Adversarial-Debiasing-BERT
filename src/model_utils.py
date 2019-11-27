import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tensorflow_hub as hub

from tensorflow.keras import backend as K
from tensorflow.keras.backend import sparse_categorical_crossentropy
from tensorflow.keras.layers import Dense, TimeDistributed

import numpy as np
import pandas as pd
from scipy import spatial
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)

def custom_acc_orig_tokens(y_true, y_pred):
    """
    calculate loss dfunction filtering out also the newly inserted labels
    
    y_true: Shape: (batch x (max_length) )
    y_pred: predictions. Shape: (batch x x (max_length + 1) x num_distinct_ner_tokens ) 
    
    returns: accuracy
    """

    #get labels and predictions
    
    y_label = tf.reshape(tf.layers.Flatten()(tf.cast(y_true, tf.int64)),[-1])
    
    mask = (y_label < 6)
    y_label_masked = tf.boolean_mask(y_label, mask)
    
    y_predicted = tf.math.argmax(input = tf.reshape(tf.layers.Flatten()(tf.cast(y_pred, tf.float64)), [-1, 10]), axis=1)
    
    y_predicted_masked = tf.boolean_mask(y_predicted, mask)

    return tf.reduce_mean(tf.cast(tf.equal(y_predicted_masked,y_label_masked) , dtype=tf.float64))

def custom_acc_orig_non_other_tokens(y_true, y_pred):
    """
    calculate loss dfunction explicitly filtering out also the 'Other'- labels
    
    y_true: Shape: (batch x (max_length) )
    y_pred: predictions. Shape: (batch x x (max_length + 1) x num_distinct_ner_tokens ) 
    
    returns: accuracy
    """

    #get labels and predictions
    
    y_label = tf.reshape(tf.layers.Flatten()(tf.cast(y_true, tf.int64)),[-1])
    
    mask = (y_label < 5)
    y_label_masked = tf.boolean_mask(y_label, mask)
    
    y_predicted = tf.math.argmax(input = tf.reshape(tf.layers.Flatten()(tf.cast(y_pred, tf.float64)),\
                                                    [-1, 10]), axis=1)
    
    y_predicted_masked = tf.boolean_mask(y_predicted, mask)

    return tf.reduce_mean(tf.cast(tf.equal(y_predicted_masked,y_label_masked) , dtype=tf.float64))

def custom_acc_protected(y_true, y_pred):
    """
    calculate loss dfunction filtering out also the newly inserted labels
    
    y_true: Shape: (batch x (max_length) )
    y_pred: predictions. Shape: (batch x x (max_length + 1) x num_distinct_ner_tokens ) 
    
    returns: accuracy
    """

    #get labels and predictions
    
    y_label = tf.reshape(tf.layers.Flatten()(tf.cast(y_true, tf.int64)),[-1])
    
    mask = (y_label < 2)
    y_label_masked = tf.boolean_mask(y_label, mask)
    
    y_predicted = tf.math.argmax(input = tf.reshape(tf.layers.Flatten()(tf.cast(y_pred, tf.float64)), [-1, 6]), axis=1)
    
    y_predicted_masked = tf.boolean_mask(y_predicted, mask)

    return tf.reduce_mean(tf.cast(tf.equal(y_predicted_masked,y_label_masked) , dtype=tf.float64))

def custom_loss(y_true, y_pred):
    """
    calculate loss function explicitly, filtering out 'extra inserted labels'
    
    y_true: Shape: (batch x (max_length + 1) )
    y_pred: predictions. Shape: (batch x x (max_length + 1) x num_distinct_ner_tokens ) 
    
    returns:  cost
    """

    #get labels and predictions
    
    y_label = tf.reshape(tf.layers.Flatten()(tf.cast(y_true, tf.int32)),[-1])
    
    mask = (y_label < 6)   # This mask is used to remove all tokens that do not correspond to the original base text.

    y_label_masked = tf.boolean_mask(y_label, mask)  # mask the labels
    
    y_flat_pred = tf.reshape(tf.layers.Flatten()(tf.cast(y_pred, tf.float32)),[-1, 10])
    
    y_flat_pred_masked = tf.boolean_mask(y_flat_pred, mask) # mask the predictions
    
    return tf.reduce_mean(sparse_categorical_crossentropy(y_label_masked, y_flat_pred_masked,from_logits=False ))

def custom_loss_protected(y_true, y_pred):
    """
    calculate loss function explicitly, filtering out 'extra inserted labels'
    
    y_true: Shape: (batch x (max_length + 1) )
    y_pred: predictions. Shape: (batch x x (max_length + 1) x num_distinct_ner_tokens ) 
    
    returns:  cost
    """

    #get labels and predictions
    
    y_label = tf.reshape(tf.layers.Flatten()(tf.cast(y_true, tf.int32)),[-1])
    
    mask = (y_label < 2)   # This mask is used to remove all tokens that do not correspond to the original base text.

    y_label_masked = tf.boolean_mask(y_label, mask)  # mask the labels
    
    y_flat_pred = tf.reshape(tf.layers.Flatten()(tf.cast(y_pred, tf.float32)),[-1, 6])
    
    y_flat_pred_masked = tf.boolean_mask(y_flat_pred, mask) # mask the predictions

    y_label_masked_inv = tf.where(
        tf.equal(1,y_label_masked), 
        tf.zeros_like(y_label_masked), 
        tf.ones_like(y_label_masked)
    )

    #tf.math.abs(tf.math.subtract(y_flat_pred,1))
    
    return tf.reduce_mean(
        sparse_categorical_crossentropy( y_label_masked, y_flat_pred_masked, from_logits=False ) + \
        sparse_categorical_crossentropy( y_label_masked_inv, y_flat_pred_masked, from_logits=False )
        )

class BertLayer(tf.keras.layers.Layer):
    """
    Create BERT layer, following https://towardsdatascience.com/bert-in-keras-with-tensorflow-hub-76bcbc9417b
    init:  initialize layer. Specify various parameters regarding output types and dimensions. Very important is
           to set the number of trainable layers.
    build: build the layer based on parameters
    call:  call the BERT layer within a model
    """
    
    def __init__(
        self,
        n_fine_tune_layers=10,
        pooling="sequence",
        bert_url="https://tfhub.dev/google/bert_cased_L-12_H-768_A-12/1",
        **kwargs,
    ):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        self.pooling = pooling
        self.bert_url = bert_url

        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            self.bert_url, trainable=self.trainable, name=f"{self.name}_module"
        )

        # Remove unused layers
        trainable_vars = self.bert.variables
        trainable_vars = [
                var
                for var in trainable_vars
                if not "/cls/" in var.name and not "/pooler/" in var.name
            ]
        trainable_layers = []


        # Select how many layers to fine tune
        for i in range(self.n_fine_tune_layers):
            trainable_layers.append(f"encoder/layer_{str(11 - i)}")

        # Update trainable vars to contain only the specified layers
        trainable_vars = [
            var
            for var in trainable_vars
            if any([l in var.name for l in trainable_layers])
        ]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "sequence_output"
            ]

        mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)

        return result

    
    def compute_output_shape(self, input_shape):
        return (None, 128, 768)


class NER():

    def __init__(self, max_input_length, filename=None):

        self.max_input_length = max_input_length

        if filename:
            filename = "../models/"+filename
            self.model = tf.keras.load(filename)

    def generate(self, bert_train_layers):
    
        in_id = tf.keras.layers.Input(shape=(self.max_input_length,), name="input_ids")
        in_mask = tf.keras.layers.Input(shape=(self.max_input_length,), name="input_masks")
        in_segment = tf.keras.layers.Input(shape=(self.max_input_length,), name="segment_ids")
        in_nerLabels = tf.keras.layers.Input(shape=(self.max_input_length, 10), name="ner_labels_true")

        bert_sequence = BertLayer(n_fine_tune_layers=bert_train_layers)([in_id, in_mask, in_segment])

        dense = tf.keras.layers.Dense(256, activation='relu', name='pred_dense')(bert_sequence)

        dense = tf.keras.layers.Dropout(rate=0.1)(dense)

        pred = tf.keras.layers.Dense(10, activation='softmax', name='ner')(dense)

        reshape = tf.keras.layers.Reshape((self.max_input_length, 10))(pred)

        concatenate = tf.keras.layers.Concatenate(axis=-1)([in_nerLabels, reshape])
        
        genderPred = tf.keras.layers.Dense(6, activation='softmax', name='gender')(concatenate)

        racePred = tf.keras.layers.Dense(6, activation='softmax', name='race')(concatenate)
        
        self.model = tf.keras.models.Model(inputs=[in_id, in_mask, in_segment, in_nerLabels], outputs={
            "ner": pred,
            "race": racePred,
            "gender": genderPred
        })
        
        self.model.summary()
        
    def fit(self, sess, train_data, val_data, epochs, batch_size, debias, 
    gender_loss_weight = 0.1, race_loss_weight = 0.1, pred_learning_rate =  2**-16, protect_learning_rate = 2**-16):

        num_train_samples = len(train_data["nerLabels"])

        ids_ph = tf.placeholder(tf.float32, shape=[batch_size, self.max_input_length])
        masks_ph = tf.placeholder(tf.float32, shape=[batch_size, self.max_input_length])
        sentenceIds_ph = tf.placeholder(tf.float32, shape=[batch_size, self.max_input_length])

        ner_ph = tf.placeholder(tf.float32, shape=[batch_size, self.max_input_length])
        gender_ph = tf.placeholder(tf.float32, shape=[batch_size, self.max_input_length])
        race_ph = tf.placeholder(tf.float32, shape=[batch_size, self.max_input_length])
        ner_onehot_ph = tf.placeholder(tf.float32, shape=[batch_size, self.max_input_length, 10])

        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.001
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)

        gender_vars = [var for var in tf.trainable_variables() if 'gender' in var.name]
        race_vars = [var for var in tf.trainable_variables() if 'race' in var.name]
        ner_vars = self.model.layers[3]._trainable_weights + [var for var in tf.trainable_variables() if any(x in var.name for x in ["pred_dense","ner"])]

        y_pred = self.model([ids_ph, masks_ph, sentenceIds_ph, ner_onehot_ph], training=True)

        ner_loss = custom_loss(ner_ph, y_pred["ner"])
        gender_loss = custom_loss_protected(gender_ph, y_pred["gender"])
        race_loss = custom_loss_protected(race_ph, y_pred["race"])

        ner_opt = tf.train.AdamOptimizer(pred_learning_rate)
        gender_opt = tf.train.AdamOptimizer(protect_learning_rate)
        race_opt = tf.train.AdamOptimizer(protect_learning_rate)

        gender_grads = {var: grad for (grad, var) in ner_opt.compute_gradients(
            gender_loss,
            var_list=ner_vars
        )}

        race_grads = {var: grad for (grad, var) in ner_opt.compute_gradients(
            race_loss,
            var_list=ner_vars
        )}

        ner_grads = []

        tf_normalize = lambda x: x / (tf.norm(x) + np.finfo(np.float32).tiny)

        for (grad, var) in ner_opt.compute_gradients(ner_loss, var_list=ner_vars):

            if debias:

                gender_unit_protect = tf_normalize(gender_grads[var])
                race_unit_protect = tf_normalize(race_grads[var])

                grad -= tf.reduce_sum(grad * gender_unit_protect) * gender_unit_protect
                grad -= tf.math.scalar_mul(gender_loss_weight, gender_grads[var])

                grad -= tf.reduce_sum(grad * race_unit_protect) * race_unit_protect
                grad -= tf.math.scalar_mul(race_loss_weight, race_grads[var])

            ner_grads.append((grad, var))

        ner_min = ner_opt.apply_gradients(ner_grads, global_step=global_step)

        gender_min = gender_opt.minimize(gender_loss, var_list=[gender_vars], global_step=global_step)

        race_min = race_opt.minimize(race_loss, var_list=[race_vars], global_step=global_step)

        initialize_vars(sess)

        epoch_pb = tqdm(range(1, epochs+1))

        for epoch in epoch_pb:

            epoch_pb.set_description("Epoch %s" % epoch)

            shuffled_ids = np.random.choice(num_train_samples, num_train_samples)

            run_pb = tqdm(range(num_train_samples//batch_size))

            for i in run_pb:

                batch_ids = shuffled_ids[batch_size*i: batch_size*(i+1)]

                batch_feed_dict = {ids_ph: train_data["inputs"][0][batch_ids], 
                                masks_ph: train_data["inputs"][1][batch_ids],
                                sentenceIds_ph: train_data["inputs"][2][batch_ids],
                                ner_onehot_ph: np.array([np.eye(10)[i.reshape(-1)] for i in train_data["nerLabels"][batch_ids]]),
                                gender_ph: train_data["genderLabels"][batch_ids],
                                race_ph: train_data["raceLabels"][batch_ids],
                                ner_ph: train_data["nerLabels"][batch_ids]}

                _, _, _, ner_loss_value, gender_loss_value, race_loss_value  = sess.run([
                    ner_min,
                    gender_min,
                    race_min,
                    ner_loss,
                    gender_loss,
                    race_loss
                ], feed_dict=batch_feed_dict)

                run_pb.set_description("nl: %.2f; gl: %.2f;  rl: %.2f" % \
                        (ner_loss_value, gender_loss_value, race_loss_value))

            inputs = val_data["inputs"] 
            
            inputs.append(np.array([np.eye(10)[i.reshape(-1)] for i in train_data["nerLabels"]]))

            val_y_pred = self.model.predict(inputs, batch_size=32)

            ner_pred = val_y_pred[1]
            ner_true = val_data["nerLabels"]

            acc_orig_tokens = custom_acc_orig_tokens(ner_true, ner_pred).eval(session=sess)
            acc_orig_non_other_tokens = custom_acc_orig_non_other_tokens(ner_true, ner_pred).eval(session=sess)

            gender_pred = val_y_pred[0]
            gender_true = val_data["genderLabels"]

            acc_gender = custom_acc_protected(gender_true, gender_pred).eval(session=sess)

            race_pred = val_y_pred[2]
            race_true = val_data["raceLabels"]

            acc_race = custom_acc_protected(race_true, race_pred).eval(session=sess)            
            print("acc_ner: %.2f; acc_ner_non_other: %.2f;  acc_gender: %.2f; acc_race: %.2f" % (acc_orig_tokens, acc_orig_non_other_tokens, acc_gender, acc_race))

    def score(self, data, batch_size=32):

        y_pred = self.model.predict(data["inputs"], batch_size=batch_size)[1]

        y_true = data["nerLabels"]

        predictions_flat = [pred for preds in np.argmax(y_pred, axis=2) for pred in preds]
        labels_flat = [label for labels in y_true for label in labels]

        clean_preds = []
        clean_labels = []

        for pred, label in zip(predictions_flat, labels_flat):
            if label < 6:
                clean_preds.append(pred)
                clean_labels.append(label)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.InteractiveSession(config=config)

        cm = tf.math.confusion_matrix(
            clean_labels,
            clean_preds,
            num_classes=None,
            dtype=tf.dtypes.int32,
            name=None,
            weights=None
        ).eval()

        plt.imshow(cm[:-1,:-1], cmap='Greens')

        sess.close()

        return cm
            
    def save(self, filename):
        filename = "../models/"+filename
        self.model.save(filename)
        print("Saved model to " + filename)

    def yieldBertEmbeddings(self, inputs):

        embeddingsModel = tf.keras.models.Model(
            inputs = [layer.input for layer in self.model.layers[0:3]], 
            outputs = self.model.layers[3].output
        )

        for input_index in range(len(inputs[0])):

            bert_input = [
                [inputs[0][input_index]], 
                [inputs[1][input_index]],
                [inputs[2][input_index]]
            ]

            yield embeddingsModel.predict(bert_input)[0]

    def getCosineDistances(self, inputs, name_masks):

        distances = []

        for input_index, embeddings in enumerate(tqdm(self.yieldBertEmbeddings(inputs))):
            
            name_vector = np.zeros(768)
            ros_vector = np.zeros(768)

            name_mask = name_masks[input_index]
            for token_index, embedding in enumerate(embeddings):
                if name_mask[token_index] == 1:
                    name_vector = np.add(name_vector, embedding)
                else:
                    ros_vector = np.add(ros_vector, embedding)

            distances.append(spatial.distance.cosine(name_vector, ros_vector))

        return np.array(distances)

    def getBiasedPValues(self, data, num_iterations=10000):

        distances = self.getCosineDistances(data["inputs"], data["nameMasks"])

        names = []
        races = []
        genders = []
        sentiment_values = []
        sentiment_confidences = []

        for sent in data["sentences"]:
            names.append(sent.new_name)
            races.append(sent.race)
            genders.append(sent.gender)
            sentiment_values.append(sent.sentiment["value"])
            sentiment_confidences.append(sent.sentiment["confidence"])

        df = pd.DataFrame(list(zip(names, races, genders, sentiment_values, sentiment_confidences, distances)), 
                  columns =['name', 'race', 'gender', 'sentiment', 'confidence', 'distance']) 

        def getScoreForNames(data, names):
    
            score_sum = 0

            for name in names:

                data_subset = data[data["name"]==name]
                
                grouped = data_subset.groupby('sentiment')
                get_weighted_avg = lambda g: np.average(g['distance'], weights=g['confidence'])
                polarity_groupby = grouped.apply(get_weighted_avg)

                score_sum += (polarity_groupby["POSITIVE"] - polarity_groupby["NEGATIVE"])

            return score_sum

        afam_names = df[df["race"]=="AFRICAN-AMERICAN"]["name"].unique()

        european_names = df[df["race"]=="EUROPEAN"]["name"].unique()

        female_names = df[df["gender"]=="FEMALE"]["name"].unique()

        male_names = df[df["gender"]=="MALE"]["name"].unique()

        def getTestStatisticForRace(data):
            return getScoreForNames(data, afam_names) - getScoreForNames(data, european_names)

        race_test_statistic = getTestStatisticForRace(df)


        def getTestStatisticForGender(data):
            return getScoreForNames(data, female_names) - getScoreForNames(data, male_names)

        gender_test_statistic = getTestStatisticForGender(df)

        race_permutation_test_statistics = []
        gender_permutation_test_statistics = []
        shuffled_data = df.copy()

        for i in tqdm(range(num_iterations)):
            shuffled_data["distance"] = shuffled_data["distance"].sample(frac=1).reset_index().drop("index", axis=1)
            
            new_race_test_statistic = getTestStatisticForRace(shuffled_data)
            race_permutation_test_statistics.append(new_race_test_statistic)
            
            new_gender_test_statistic = getTestStatisticForGender(shuffled_data)
            gender_permutation_test_statistics.append(new_gender_test_statistic)

        
        race_p = np.sum(np.abs(race_test_statistic) < np.abs(race_permutation_test_statistics))/(2*len(race_permutation_test_statistics))

        gender_p = np.sum(np.abs(gender_test_statistic) < np.abs(gender_permutation_test_statistics))/(2*len(gender_permutation_test_statistics))

        return {
            "race": race_p, 
            "gender": gender_p,
            "race_test_statistic": race_test_statistic,
            "gender_test_statistic": gender_test_statistic,
            "race_std": np.std(race_permutation_test_statistics),
            "gender_std": np.std(gender_permutation_test_statistics)
        }