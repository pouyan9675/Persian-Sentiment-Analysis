import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from string import punctuation
from hazm import Lemmatizer, Normalizer, word_tokenize
import re
from collections import Counter
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.python.client import device_lib
from statistics import mean


class Model:
    def __init__(self):
        # Hyperparamters
        self.BATCH_SIZE = 64
        self.EPOCHES = 3
        self.LEARNING_RATE = 1e-5

        with open('fa-stopwords.txt', 'r', encoding='utf8') as f:
            self.stopwords = [x for x in f.read().split()]


    def _numbers_to_english(self, text):
        text = text.replace('۰', '0')
        text = text.replace('۱', '1')
        text = text.replace('۲', '2')
        text = text.replace('۳', '3')
        text = text.replace('۴', '4')
        text = text.replace('۵', '5')
        text = text.replace('۶', '6')
        text = text.replace('۷', '7')
        text = text.replace('۸', '8')
        text = text.replace('۹', '9')
        return text


    def preprocess(self, cm):
        cm = ''.join([c for c in str(cm) if c not in punctuation])
        cm = self._numbers_to_english(cm)
        cm = re.sub(r"[0-9]", '', cm)
        cm = cm.replace('\u200c', ' ').replace('\n', '').replace('\r', '').replace('ي', 'ی').replace('ك', 'ک')
        normalizer = Normalizer()
        cm = normalizer.normalize(cm)
        tokens = word_tokenize(cm)
        cm = ' '.join([x for x in tokens if x not in self.stopwords])
        return cm


    def _encode_comment(self, comment):
        terms = comment.split()
        cm = []
        for t in terms:
            if t in self.word2id:
                cm.append(self.word2id[t])
            else:
                cm.append(self.word2id['OOV'])
        return cm


    def predict(self, text):
        text = self.preprocess(text)
        en_cm = [encode_comment(text)]
        input_cm = sequence.pad_sequences(en_cm, maxlen=self.max_words)
        probabilities = model.predict(input_cm)
        print(' (+) Positive: {:.1f}%'.format(probabilities[0][2] * 100))
        print(' (/) Neutral:  {:.1f}%'.format(probabilities[0][0] * 100))
        print(' (-) Negative: {:.1f}%'.format(probabilities[0][1] * 100))


    def _build_model(self):
        embedding_size = 100
        self.model = Sequential()
        self.model.add(Embedding(input_dim=self.word_embeddings.shape[0], output_dim=self.word_embeddings.shape[1], weights=[self.word_embeddings], trainable=False, input_length=self.max_words))
        self.model.add(Dropout(rate=0.2))
        self.model.add(Bidirectional(LSTM(100)))
        self.model.add(Dropout(rate=0.4))
        self.model.add(Dense(3, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

        print(self.model.summary())


    def _load_word_embedding(self, words, file_name="persian_glove300d.txt"):
        self.word2id = {}
        self.word_embeddings = []

        fEmbeddings = open("embeddings/" + file_name, encoding="utf-8")

        for line in fEmbeddings:
            split = line.strip().split(" ")
            word = str(split[0])
            
            if len(self.word2id) == 0:
                self.word2id['PAD'] = len(self.word2id)
                vector = np.zeros(len(split)-1) # Zero vector for 'PADDING' word
                self.word_embeddings.append(vector)
                
                self.word2id['OOV'] = len(self.word2id)
                vector = np.random.uniform(-0.25, 0.25, len(split)-1)
                self.word_embeddings.append(vector)

            if word.lower() in words:
                vector = np.array([float(num) for num in split[1:]])
                self.word_embeddings.append(vector)
                self.word2id[word] = len(self.word2id)

        self.word_embeddings = np.array(self.word_embeddings) # shape[0] == vocab, shape[1] == embedding dimen
        fEmbeddings.close()


    def train(self, dataset_name='2-Users Comments.xlsx'):
        print('Training started...')
        print('Opening dataset...')
        # print(device_lib.list_local_devices())
        # print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

        comments = pd.read_excel('dataset/' + dataset_name)

        print('Preprocessing...')
        comments.dropna(subset=['comment'], inplace=True)

        indices = comments[comments['recommend'] == '\\N'].index
        comments = comments.drop(indices)

        le = LabelEncoder()
        le.fit(comments['recommend'])
        
        comments['enc_recommend'] = le.transform(comments['recommend'])
        comments['comment'] = comments['comment'].astype(str) + ' ' + comments['title'].astype(str)


        comments['comment'] = comments['comment'].apply(self.preprocess)

        indices = comments[comments['comment'].str.len() == 0].index
        comments = comments.drop(indices)

        coherent = ' '.join(comments['comment'])
        tokens = coherent.split()
        words = Counter(tokens)

        self._load_word_embedding(words)

        comments['enc_comment'] = comments['comment'].apply(self._encode_comment)

        X_train, X_test, y_train, y_test = train_test_split(
            comments.enc_comment,
            comments.enc_recommend,
            test_size=0.1
        )

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        words_length = {len(i) for i in comments.enc_comment}
        self.max_words = int(max(words_length))

        X_train = sequence.pad_sequences(X_train, maxlen=self.max_words)
        X_test = sequence.pad_sequences(X_test, maxlen=self.max_words)
        
        self._build_model()


        X_valid, y_valid = X_train[:self.BATCH_SIZE], y_train[:self.BATCH_SIZE]
        X_train2, y_train2 = X_train[self.BATCH_SIZE:], y_train[self.BATCH_SIZE:]

        self.model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=self.BATCH_SIZE, epochs=self.EPOCHES)