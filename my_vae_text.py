#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
    
from keras.layers import Bidirectional, Dense, Embedding, Input, Lambda, LSTM, RepeatVector, TimeDistributed, Layer, Activation, Dropout
from keras.utils import pad_sequences
from keras.layers import ELU, PReLU, LeakyReLU
#from keras.layers.advanced_activations import ELU
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
#from keras import objectives
from scipy import spatial

import pandas as pd
import numpy as np
import codecs
import csv
import os


# ### Directories and text loading
# Initially we will set the main directories and some variables regarding the characteristics of our texts.
# We set the maximum sequence length to 15, the maximun number of words in our vocabulary to 12000 and we will use 50-dimensional embeddings. Finally we load our texts from a csv. The text file is the train file of the Quora Kaggle challenge containing around 808000 sentences.

# In[ ]:


BASE_DIR = '/home/dguha/Python_Sentiment/Quora_Question_Pair/'
Glove_Dir = '/home/dguha/Python_Sentiment/Youtube_Sentiment/input_glove/'
#path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

TRAIN_DATA_FILE = BASE_DIR + 'questions.csv'#'train_micro.csv'
GLOVE_EMBEDDING = Glove_Dir + 'glove.6B.50d.txt'
VALIDATION_SPLIT = 0.2
MAX_SEQUENCE_LENGTH = 15
MAX_NB_WORDS = 12000
EMBEDDING_DIM = 50

texts = [] 
with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        texts.append(values[3])
        texts.append(values[4])
print('Found %s texts in train.csv' % len(texts))


# ### Text Preprocessing
# To preprocess the text we will use the tokenizer and the text_to_sequences function from Keras
# 

# In[ ]:


tokenizer = Tokenizer(MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index #the dict values start from 1 so this is fine with zeropadding
index2word = {v: k for k, v in word_index.items()}
print('Found %s unique tokens' % len(word_index))
sequences = tokenizer.texts_to_sequences(texts)
data_1 = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data_1.shape)
NB_WORDS = (min(tokenizer.num_words, len(word_index)) + 1 ) #+1 for zero padding
data_1_val = data_1[801000:807000] #select 6000 sentences as validation data


# ### Sentence generator
# In order to reduce the memory requirements we will gradually read our sentences from the csv through Pandas as we feed them to the model

# In[ ]:


def sent_generator(TRAIN_DATA_FILE, chunksize):
    reader = pd.read_csv(TRAIN_DATA_FILE, chunksize=chunksize, iterator=True)
    for df in reader:
        #print(df.shape)
        #df=pd.read_csv(TRAIN_DATA_FILE, iterator=False)
        val3 = df.iloc[:,3:4].values.tolist()
        val4 = df.iloc[:,4:5].values.tolist()
        flat3 = [item for sublist in val3 for item in sublist]
        flat4 = [str(item) for sublist in val4 for item in sublist]
        texts = [] 
        texts.extend(flat3[:])
        texts.extend(flat4[:])
        
        sequences = tokenizer.texts_to_sequences(texts)
        data_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        yield [data_train, data_train]


# ### Word embeddings
# We will use pretrained Glove word embeddings as embeddings for our network. We create a matrix with one embedding for every word in our vocabulary and then we will pass this matrix as weights to the keras embedding layer of our model

# In[ ]:


embeddings_index = {}
f = open(GLOVE_EMBEDDING, encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

glove_embedding_matrix = np.zeros((NB_WORDS, EMBEDDING_DIM))
for word, i in word_index.items():
    if i < NB_WORDS:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be the word embedding of 'unk'.
            glove_embedding_matrix[i] = embedding_vector
        else:
            glove_embedding_matrix[i] = embeddings_index.get('unk')
print('Null word embeddings: %d' % np.sum(np.sum(glove_embedding_matrix, axis=1) == 0))


# ### VAE model
# Our model is based on a seq2seq architecture with a bidirectional LSTM encoder and an LSTM decoder and ELU activations.
# We feed the latent representation at every timestep as input to the decoder through "RepeatVector(max_len)".
# To avoid the one-hot representation of labels we use the "tf.contrib.seq2seq.sequence_loss" that requires as labels only the word indexes (the same that go in input to the embedding matrix) and calculates internally the final softmax (so the model ends with a dense layer with linear activation). Optionally the "sequence_loss" allows to use the sampled softmax which helps when dealing with large vocabularies (for example with a 50k words vocabulary) but in this I didn't use it.
# Moreover, due to the pandas iterator that reads the csv both the train size and validation size must be divisible by the batch_size.

# In[ ]:


batch_size = 100
max_len = MAX_SEQUENCE_LENGTH
emb_dim = EMBEDDING_DIM
latent_dim = 32
intermediate_dim = 96
epsilon_std = 1.0
num_sampled=500
act = ELU()

#y = Input(batch_shape=(None, max_len, NB_WORDS))
x = Input(batch_shape=(None, max_len))
x_embed = Embedding(NB_WORDS, emb_dim, weights=[glove_embedding_matrix],
                            input_length=max_len, trainable=False)(x)
h = Bidirectional(LSTM(intermediate_dim, return_sequences=False, recurrent_dropout=0.2), merge_mode='concat')(x_embed)
#h = Bidirectional(LSTM(intermediate_dim, return_sequences=False), merge_mode='concat')(h)
h = Dropout(0.2)(h)
h = Dense(intermediate_dim, activation='linear')(h)
h = act(h)
h = Dropout(0.2)(h)
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_sigma / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

# we instantiate these layers separately so as to reuse them later
repeated_context = RepeatVector(max_len)
decoder_h = LSTM(intermediate_dim, return_sequences=True, recurrent_dropout=0.2)
decoder_mean = TimeDistributed(Dense(NB_WORDS, activation='linear'))#softmax is applied in the seq2seqloss by tf
h_decoded = decoder_h(repeated_context(z))
x_decoded_mean = decoder_mean(h_decoded)

vae = Model(x, x_decoded_mean)

def vae_loss(x, x_decoded_mean):
        mse = tf.keras.losses.MeanSquaredError()
        xent_loss = mse(x, x_decoded_mean)
        #xent_loss = objectives.mse(x, x_decoded_mean)
        #xent_loss = K.sum(metrics.categorical_crossentropy(x, x_decoded_mean), axis=-1)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        #loss = kl_loss
        return loss

opt = Adam(learning_rate=0.01) #SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
vae.compile(optimizer='adam', loss=vae_loss)
vae.summary()

# placeholder loss
#def zero_loss(y_true, y_pred):
#    return K.zeros_like(y_pred)
#
##=========================== Necessary only if you want to use Sampled Softmax =======================#
##Sampled softmax
#logits = tf.constant(np.random.randn(batch_size, max_len, NB_WORDS), tf.float32)
#targets = tf.constant(np.random.randint(NB_WORDS, size=(batch_size, max_len)), tf.int32)
#proj_w = tf.constant(np.random.randn(NB_WORDS, NB_WORDS), tf.float32)
#proj_b = tf.constant(np.zeros(NB_WORDS), tf.float32)
#
#def _sampled_loss(labels, logits):
#    labels = tf.cast(labels, tf.int64)
#    labels = tf.reshape(labels, [-1, 1])
#    logits = tf.cast(logits, tf.float32)
#    return tf.cast(
#                    tf.nn.sampled_softmax_loss(
#                        proj_w,
#                        proj_b,
#                        labels,
#                        logits,
#                        num_sampled=num_sampled,
#                        num_classes=NB_WORDS),
#                    tf.float32)
#softmax_loss_f = _sampled_loss
#====================================================================================================#

# Custom VAE loss layer
#class CustomVariationalLayer(Layer):
#    def __init__(self, **kwargs):
#        self.is_placeholder = True
#        super(CustomVariationalLayer, self).__init__(**kwargs)
#        self.target_weights = tf.constant(np.ones((batch_size, max_len)), tf.float32)
#
#    def vae_loss(self, x, x_decoded_mean):
#        #xent_loss = K.sum(metrics.categorical_crossentropy(x, x_decoded_mean), axis=-1)
#        labels = tf.cast(x, tf.int32)
#        xent_loss = objectives.mse(x, x_decoded_mean)
#        kl_loss = - 0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
#        return K.mean(xent_loss + kl_loss)
#
#    def call(self, inputs):
#        x = inputs[0]
#        x_decoded_mean = inputs[1]
#        print(x.shape, x_decoded_mean.shape)
#        loss = self.vae_loss(x, x_decoded_mean)
#        self.add_loss(loss, inputs=inputs)
#        # we don't use this output, but it has to have the correct shape:
#        return K.ones_like(x)

#loss_layer = CustomVariationalLayer()([x, x_decoded_mean])
#vae = Model(x, [loss_layer])


# ### Model training
# We train our model for 100 epochs through keras ".fit_generator". The number of steps per epoch is equal to the number of sentences that we have in the train set (800000) divided by the batch size; the additional /2 is due to the fact that our csv has two sentnces per line so in the end we have to read with our generator only 400000 lines per epoch.
# For validation data we pass the same array twice since input and labels of this model are the same. 
# If we didn't use the "tf.contrib.seq2seq.sequence_loss" (or another similar function) we would have had to pass as labels the sequence of word one-hot encodings with dimension (batch_size, seq_len, vocab_size) consuming a lot of memory.

# In[ ]:


def create_model_checkpoint(dir, model_name):
    filepath = dir + '/' + model_name + ".h5" #-{epoch:02d}-{decoded_mean:.2f}
    directory = os.path.dirname(filepath)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=False)
    return checkpointer

checkpointer = create_model_checkpoint('models', 'vae_seq2seq')

nb_epoch=100
n_steps = (800000/2)/batch_size #we use the first 800000
for counter in range(nb_epoch):
    print('-------epoch: ',counter,'--------')
    vae.fit(sent_generator(TRAIN_DATA_FILE, batch_size/2),
                          steps_per_epoch=n_steps, epochs=1, callbacks=[checkpointer],
                          validation_data=(data_1_val, data_1_val))
    
vae.save('models/vae_lstm800k32dim96hid.h5')


# ### Project and sample sentences from the latent space
# Now we build an encoder model model that takes a sentence and projects it on the latent space and a decoder model that goes from the latent space back to the text representation

# In[ ]:


# build a model to project sentences on the latent space
encoder = Model(x, z_mean)

# build a generator that can sample sentences from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(repeated_context(decoder_input))
_x_decoded_mean = decoder_mean(_h_decoded)
_x_decoded_mean = Activation('softmax')(_x_decoded_mean)
generator = Model(decoder_input, _x_decoded_mean)


# ### Test on validation sentences

# In[ ]:


index2word = {v: k for k, v in word_index.items()}
sent_encoded = encoder.predict(data_1_val, batch_size = 16)
x_test_reconstructed = generator.predict(sent_encoded)
                                         
sent_idx = 672
reconstructed_indexes = np.apply_along_axis(np.argmax, 1, x_test_reconstructed[sent_idx])
#np.apply_along_axis(np.max, 1, x_test_reconstructed[sent_idx])
#np.max(np.apply_along_axis(np.max, 1, x_test_reconstructed[sent_idx]))
word_list = list(np.vectorize(index2word.get)(reconstructed_indexes))
word_list
original_sent = list(np.vectorize(index2word.get)(data_1_val[sent_idx]))
original_sent


# ### Sentence processing and interpolation

# In[ ]:


# function to parse a sentence
def sent_parse(sentence, mat_shape):
    sequence = tokenizer.texts_to_sequences(sentence)
    padded_sent = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    return padded_sent#[padded_sent, sent_one_hot]

# input: encoded sentence vector
# output: encoded sentence vector in dataset with highest cosine similarity
def find_similar_encoding(sent_vect):
    all_cosine = []
    for sent in sent_encoded:
        result = 1 - spatial.distance.cosine(sent_vect, sent)
        all_cosine.append(result)
    data_array = np.array(all_cosine)
    maximum = data_array.argsort()[-3:][::-1][1]
    new_vec = sent_encoded[maximum]
    return new_vec

# input: two points, integer n
# output: n equidistant points on the line between the input points (inclusive)
def shortest_homology(point_one, point_two, num):
    dist_vec = point_two - point_one
    sample = np.linspace(0, 1, num, endpoint = True)
    hom_sample = []
    for s in sample:
        hom_sample.append(point_one + s * dist_vec)
    return hom_sample

# input: original dimension sentence vector
# output: sentence text
def print_latent_sentence(sent_vect):
    sent_vect = np.reshape(sent_vect,[1,latent_dim])
    sent_reconstructed = generator.predict(sent_vect)
    sent_reconstructed = np.reshape(sent_reconstructed,[max_len,NB_WORDS])
    reconstructed_indexes = np.apply_along_axis(np.argmax, 1, sent_reconstructed)
    np.apply_along_axis(np.max, 1, x_test_reconstructed[sent_idx])
    np.max(np.apply_along_axis(np.max, 1, x_test_reconstructed[sent_idx]))
    word_list = list(np.vectorize(index2word.get)(reconstructed_indexes))
    w_list = [w for w in word_list if w]
    print(' '.join(w_list))
    #print(word_list)
        
def new_sents_interp(sent1, sent2, n):
    tok_sent1 = sent_parse(sent1, [15])
    tok_sent2 = sent_parse(sent2, [15])
    enc_sent1 = encoder.predict(tok_sent1, batch_size = 16)
    enc_sent2 = encoder.predict(tok_sent2, batch_size = 16)
    test_hom = shortest_homology(enc_sent1, enc_sent2, n)
    for point in test_hom:
        print_latent_sentence(point)


# ### Example
# Now we can try to parse two sentences and interpolate between them generating new sentences

# In[ ]:


sentence1=['where can i find a book on machine learning']
mysent = sent_parse(sentence1, [15])
mysent_encoded = encoder.predict(mysent, batch_size = 16)
print_latent_sentence(mysent_encoded)
print_latent_sentence(find_similar_encoding(mysent_encoded))

sentence2=['how can i become a successful entrepreneur']
mysent2 = sent_parse(sentence2, [15])
mysent_encoded2 = encoder.predict(mysent2, batch_size = 16)
print_latent_sentence(mysent_encoded2)
print_latent_sentence(find_similar_encoding(mysent_encoded2))
print('-----------------')

new_sents_interp(sentence1, sentence2, 6)


# ### Results
# After training with these parameters for 100 epochs I got these results from interpolating between these two sentences:
# 
# sentence1=['where can i find a book on machine learning']
# sentence2=['how can i become a successful entrepreneur']
# 
# Generated sentences:
# - ------------------------------------------- -
# -  where can i find a book on machine learning
# -  where can i find a a machine book
# -  how can i write a a machine book
# -  how can i become a successful architect
# -  how can i become a successful architect
# -  how can i become a successful entrepreneur
# - ------------------------------------------- -
# 
# As we can see the results are not yet completely satisfying because not all the sentences are grammatically correct and in the interpolation the same sentence has been generated multiple times but anyway the model, even in this preliminary version seems to start working.
# There are certainly many improvements that could be done like:
# -  removing all the sentences longer than 15 instead of just truncating them
# -  introduce the equivalent of word dropout used in the original paper for this decoder architecture 
# -  parameter tuning (this model trains in few hours on a GTX950M with 2GB memory so it's definitely possible to try larger nets)
# -  Using word embeddings with higher dimensionality
# -  train on a more general dataset (Quora sentences are all questions)
# 
# Stay tuned for future refinings of the model!
