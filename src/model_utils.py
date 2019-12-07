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

    return tf.reduce_mean(sparse_categorical_crossentropy(y_label_masked, y_flat_pred_masked,from_logits=False ))


class BertLayer(tf.keras.layers.Layer):

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
        return (input_shape[0], self.output_size)

class NER():

    def __init__(self, max_input_length):

        self.max_input_length = max_input_length

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.initialized = False

    def initialize_vars(self):
        if not self.initialized:
            self.sess.run(tf.local_variables_initializer())
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.tables_initializer())
            K.set_session(self.sess)
            self.initialized = True

    def generate(self, bert_train_layers):
    
        in_id = tf.keras.layers.Input(shape=(self.max_input_length,), name="input_ids")
        in_mask = tf.keras.layers.Input(shape=(self.max_input_length,), name="input_masks")
        in_segment = tf.keras.layers.Input(shape=(self.max_input_length,), name="segment_ids")
        in_nerLabels = tf.keras.layers.Input(shape=(self.max_input_length, 10), name="ner_labels_true")

        bert_sequence = BertLayer(n_fine_tune_layers=bert_train_layers)([in_id, in_mask, in_segment])

        pred = tf.keras.layers.Dense(10, activation='softmax', name='ner')(bert_sequence)

        reshape = tf.keras.layers.Reshape((self.max_input_length, 10))(pred)

        concatenate = tf.keras.layers.Concatenate(axis=-1)([in_nerLabels, reshape])

        genderDense1 = tf.keras.layers.Dense(30, activation='relu', name='genderDense1')(concatenate)
        genderDense1 = tf.keras.layers.Dropout(rate=0.1)(genderDense1)

        genderDense2 = tf.keras.layers.Dense(30, activation='relu', name='genderDense2')(genderDense1)
        genderDense2 = tf.keras.layers.Dropout(rate=0.1)(genderDense2)

        genderPred = tf.keras.layers.Dense(6, activation='softmax', name='genderPred')(genderDense2)

        raceDense1 = tf.keras.layers.Dense(30, activation='relu', name='raceDense1')(concatenate)
        raceDense1 = tf.keras.layers.Dropout(rate=0.1)(raceDense1)

        raceDense2 = tf.keras.layers.Dense(30, activation='relu', name='raceDense2')(raceDense1)
        raceDense2 = tf.keras.layers.Dropout(rate=0.1)(raceDense2)

        racePred = tf.keras.layers.Dense(6, activation='softmax', name='racePred')(raceDense2)
        
        self.model = tf.keras.models.Model(inputs=[in_id, in_mask, in_segment, in_nerLabels], outputs={
            "ner": pred,
            "race": racePred,
            "gender": genderPred
        })
        
        self.model.summary()
        
    def fit(self, train_data, val_data, epochs, batch_size, debias, 
            pred_learning_rate = 2**-17, protect_learning_rate = 1**-3,
            alpha=0.1, beta=0.1):

        num_train_samples = len(train_data["nerLabels"])

        ids_ph = tf.placeholder(tf.float32, shape=[batch_size, self.max_input_length])
        masks_ph = tf.placeholder(tf.float32, shape=[batch_size, self.max_input_length])
        sentenceIds_ph = tf.placeholder(tf.float32, shape=[batch_size, self.max_input_length])

        ner_ph = tf.placeholder(tf.float32, shape=[batch_size, self.max_input_length])
        gender_ph = tf.placeholder(tf.float32, shape=[batch_size, self.max_input_length])
        race_ph = tf.placeholder(tf.float32, shape=[batch_size, self.max_input_length])
        ner_onehot_ph = tf.placeholder(tf.float32, shape=[batch_size, self.max_input_length, 10])

        global_step = tf.Variable(0, trainable=False)

        gender_vars = [var for var in tf.trainable_variables() if 'gender' in var.name]
        race_vars = [var for var in tf.trainable_variables() if 'race' in var.name]
        ner_vars = self.model.layers[3]._trainable_weights + [var for var in tf.trainable_variables() if "ner" in var.name]

        y_pred = self.model([ids_ph, masks_ph, sentenceIds_ph, ner_onehot_ph], training=True)

        ner_loss = custom_loss(ner_ph, y_pred["ner"])
        gender_loss = custom_loss_protected(gender_ph, y_pred["gender"])
        race_loss = custom_loss_protected(race_ph, y_pred["race"])

        ner_opt = tf.train.AdamOptimizer(pred_learning_rate)
        gender_opt = tf.train.AdamOptimizer(protect_learning_rate)
        race_opt = tf.train.AdamOptimizer(protect_learning_rate)

        if debias:

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

                gender_unit_protect = tf_normalize(gender_grads[var])
                race_unit_protect = tf_normalize(race_grads[var])

                grad -= tf.reduce_sum(grad * gender_unit_protect) * gender_unit_protect
                grad -= tf.math.scalar_mul(alpha, gender_grads[var])

                grad -= tf.reduce_sum(grad * race_unit_protect) * race_unit_protect
                grad -= tf.math.scalar_mul(beta, race_grads[var])

                ner_grads.append((grad, var))

            ner_min = ner_opt.apply_gradients(ner_grads, global_step=global_step)

        else:

            ner_min = ner_opt.minimize(ner_loss, var_list=ner_vars, global_step=global_step)

        gender_min = gender_opt.minimize(gender_loss, var_list=gender_vars, global_step=global_step)

        race_min = race_opt.minimize(race_loss, var_list=race_vars, global_step=global_step)

        self.initialize_vars()

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
                                ner_onehot_ph: train_data["inputs"][3][batch_ids],
                                gender_ph: train_data["genderLabels"][batch_ids],
                                race_ph: train_data["raceLabels"][batch_ids],
                                ner_ph: train_data["nerLabels"][batch_ids]}

                _, _, _, ner_loss_value, gender_loss_value, race_loss_value  = self.sess.run([
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
            
            val_y_pred = self.model.predict(inputs, batch_size=32)

            ner_pred = val_y_pred[1]
            ner_true = val_data["nerLabels"]

            acc_orig_tokens = custom_acc_orig_tokens(ner_true, ner_pred).eval(session=self.sess)
            acc_orig_non_other_tokens = custom_acc_orig_non_other_tokens(ner_true, ner_pred).eval(session=self.sess)

            gender_pred = val_y_pred[0]
            gender_true = val_data["genderLabels"]

            acc_gender = custom_acc_protected(gender_true, gender_pred).eval(session=self.sess)

            race_pred = val_y_pred[2]
            race_true = val_data["raceLabels"]

            acc_race = custom_acc_protected(race_true, race_pred).eval(session=self.sess)            
            print("val_acc_ner: %.2f; val_acc_ner_non_other: %.2f;  val_acc_gender: %.2f; val_acc_race: %.2f" % (acc_orig_tokens, acc_orig_non_other_tokens, acc_gender, acc_race))

    def score(self, data, batch_size=32):

        self.initialize_vars()

        y_pred = self.model.predict(data["inputs"], batch_size=batch_size)[1]

        y_true = data["nerLabels"]

        print("acc_ner: %.2f; acc_ner_non_other: %.2f" % (custom_acc_orig_tokens(y_true, y_pred).eval(session=self.sess), (custom_acc_orig_non_other_tokens(y_true, y_pred).eval(session=self.sess))))

        predictions_flat = [pred for preds in np.argmax(y_pred, axis=2) for pred in preds]
        labels_flat = [label for labels in y_true for label in labels]

        clean_preds = []
        clean_labels = []

        for pred, label in zip(predictions_flat, labels_flat):
            if label < 6:
                clean_preds.append(pred)
                clean_labels.append(label)

        cm = tf.math.confusion_matrix(
            clean_labels,
            clean_preds,
            num_classes=None,
            dtype=tf.dtypes.int32,
            name=None,
            weights=None
        ).eval(session=self.sess)

        plt.imshow(cm[:-1,:-1], cmap='Greens')

        return cm
            
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

        self.initialize_vars()

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


        def getAddendPerName(data, name):

            data_subset = data[data["name"]==name]
            grouped = data_subset.groupby('sentiment')
            get_weighted_avg = lambda g: np.average(g['distance'], weights=g['confidence'])
            polarity_groupby = grouped.apply(get_weighted_avg)

            return (polarity_groupby["POSITIVE"] - polarity_groupby["NEGATIVE"])

        afam_names = df[df["race"]=="AFRICAN-AMERICAN"]["name"].unique()

        european_names = df[df["race"]=="EUROPEAN"]["name"].unique()

        female_names = df[df["gender"]=="FEMALE"]["name"].unique()

        male_names = df[df["gender"]=="MALE"]["name"].unique()

        def getStatsForRace(data):

            ## returns test statistic and effect size

            afam_addends = [getAddendPerName(data, name) for name in afam_names]
            european_addends = [getAddendPerName(data, name) for name in european_names]

            test_statistic =  np.sum(afam_addends) - np.sum(european_addends)
            effect_size = (np.mean(afam_addends) - np.mean(european_addends)) / np.std(afam_addends + european_addends)

            return test_statistic, effect_size

        race_test_statistic, race_effect_size = getStatsForRace(df)

        def getStatsForGender(data):

            female_addends = [getAddendPerName(data, name) for name in female_names]
            male_addends = [getAddendPerName(data, name) for name in male_names]

            test_statistic =  np.sum(female_addends) - np.sum(male_addends)
            effect_size = (np.mean(female_addends) - np.mean(male_addends)) / np.std(female_addends + male_addends)

            return test_statistic, effect_size

        gender_test_statistic, gender_effect_size = getStatsForGender(df)

        race_permutation_test_statistics = []
        gender_permutation_test_statistics = []
        shuffled_data = df.copy()

        for i in tqdm(range(num_iterations)):
            shuffled_data["distance"] = shuffled_data["distance"].sample(frac=1).reset_index().drop("index", axis=1)
            
            new_race_test_statistic, new_race_effect_size = getStatsForRace(shuffled_data)
            race_permutation_test_statistics.append(new_race_test_statistic)
            
            new_gender_test_statistic, new_gender_effect_size = getStatsForGender(shuffled_data)
            gender_permutation_test_statistics.append(new_gender_test_statistic)
        
        race_p = np.sum(np.abs(race_test_statistic) < np.abs(race_permutation_test_statistics))/(2*len(race_permutation_test_statistics))
        gender_p = np.sum(np.abs(gender_test_statistic) < np.abs(gender_permutation_test_statistics))/(2*len(gender_permutation_test_statistics))

        return {
            "race": race_p, 
            "gender": gender_p,
            "race_test_statistic": race_test_statistic,
            "gender_test_statistic": gender_test_statistic,
            "race_std": np.std(race_permutation_test_statistics),
            "gender_std": np.std(gender_permutation_test_statistics),
            "race_effect_size": race_effect_size,
            "gender_effect_size": gender_effect_size
        }