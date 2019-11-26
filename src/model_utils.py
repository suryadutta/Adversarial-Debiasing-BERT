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

from tqdm import tqdm

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
    
    y_predicted = tf.math.argmax(input = tf.reshape(tf.layers.Flatten()(tf.cast(y_pred, tf.float64)),\
                                                    [-1, 10]), axis=1)
    
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
        return (input_shape[0], self.output_size)

class NER():

    def __init__(self, filename=None):
        if filename:
            filename = "../models/"+filename
            self.model = tf.keras.load(filename)

    def generate(self, max_input_length, train_layers, optimizer, debias, debiasWeight=0.5):
    
        in_id = tf.keras.layers.Input(shape=(max_input_length,), name="input_ids")
        in_mask = tf.keras.layers.Input(shape=(max_input_length,), name="input_masks")
        in_segment = tf.keras.layers.Input(shape=(max_input_length,), name="segment_ids")
        
        bert_inputs = [in_id, in_mask, in_segment]
        
        bert_sequence = BertLayer(n_fine_tune_layers=train_layers)(bert_inputs)
            
        dense = tf.keras.layers.Dense(256, activation='relu', name='dense')(bert_sequence)
        
        dense = tf.keras.layers.Dropout(rate=0.1)(dense)
        
        pred = tf.keras.layers.Dense(10, activation='softmax', name='ner')(dense)
        
        if(debias):

            genderPred = tf.keras.layers.Dense(6, activation='softmax', name='gender')(pred)

            racePred = tf.keras.layers.Dense(6, activation='softmax', name='race')(pred)

            losses = {
                "ner": custom_loss,
                "race": custom_loss_protected,
                "gender": custom_loss_protected
            }

            lossWeights = {
                "ner": 1.0-debiasWeight,
                "race": debiasWeight/2.0,
                "gender": debiasWeight/2.0
            }

            self.model = tf.keras.models.Model(inputs=bert_inputs, outputs={
                "ner": pred,
                "race": racePred,
                "gender": genderPred
            })

            self.model.compile(
                loss=losses,
                loss_weights=lossWeights,
                optimizer=optimizer, 
                metrics={"ner": [custom_acc_orig_tokens,custom_acc_orig_non_other_tokens]})

        else:
            
            self.model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)

            self.model.compile(loss=custom_loss, optimizer=optimizer, metrics=[custom_acc_orig_tokens, 
                                                                custom_acc_orig_non_other_tokens])
            
        self.model.summary()
        
    def fit(self, train_data, val_data, epochs, batch_size):

        self.model.fit(
            train_data["inputs"], 
            {
                "ner": train_data["nerLabels"],
                "gender": train_data["genderLabels"],
                "race": train_data["raceLabels"]
            },
            validation_data=(val_data["inputs"], {
                "ner": val_data["nerLabels"],
                "gender": val_data["genderLabels"],
                "race": val_data["raceLabels"]
            }),
            epochs=epochs,
            batch_size=batch_size
        )

    def score(self, data, batch_size=32):

        y_pred = self.model.predict(data["inputs"], batch_size=batch_size)

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

    def getBiasedPValues(self, data, num_iterations=1000):

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