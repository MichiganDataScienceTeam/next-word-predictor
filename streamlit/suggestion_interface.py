import streamlit as st
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

import pickle
import json

import nltk
from nltk.tokenize import word_tokenize

import os
import keras_nlp
import keras_core as keras
import time


def predict_next_words(model, text, max_sequence_len, word_to_index, index_to_word):
    # Tokenize the input string
    token_list = [word_to_index[word] for word in word_tokenize(text) if word in word_to_index]

    # Pad the token sequence
    token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')

    # Predict the tokens of the next words
    output = np.array(model.predict(token_list))[0]
    predicted_indicies = np.argpartition(output, -5)[-5:]

    # Convert the tokens back to words
    predicted_words = []
    probs = []
    for idx in predicted_indicies:
        predicted_words.append(index_to_word[idx].lower())
        probs.append(str(round(output[idx] * 100, 2)) + '%')

    return predicted_words, probs


def predict_next_n_words(model, text, n, max_sequence_len, word_to_index, index_to_word):
    predicted_sequences = []

    for i in range(5):
        temp_text = text
        predicted_sequence = []

        # Tokenize the input string
        token_list = [word_to_index[word] for word in word_tokenize(temp_text) if word in word_to_index]

        # Pad the token sequence
        token_list = tf.keras.utils.pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')

        # Predict the token of the next word
        output = np.array(model.predict(token_list))[0]
        predicted_idx = np.argpartition(output, -5)[-(i+1)]

        # Convert the token back to a word
        #predicted_word = index_to_word.get(predicted_idx[0], '')
        predicted_word = index_to_word[predicted_idx].lower()

        # Append the predicted word to the sequence and to the text (for the next iteration)
        predicted_sequence.append(predicted_word)
        temp_text += " " + predicted_word

        for j in range(n-1):
            # Tokenize the input string
            token_list = [word_to_index[word] for word in word_tokenize(temp_text) if word in word_to_index]

            # Pad the token sequence
            token_list = tf.keras.utils.pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')

            # Predict the token of the next word
            output = np.array(model.predict(token_list))[0]
            predicted_idx = np.argmax(output, axis=-1)

            # Convert the token back to a word
            #predicted_word = index_to_word.get(predicted_idx[0], '')
            predicted_word = index_to_word[predicted_idx].lower()

            # Append the predicted word to the sequence and to the text (for the next iteration)
            predicted_sequence.append(predicted_word)
            temp_text += " " + predicted_word
        predicted_sequences.append(' '.join(predicted_sequence))

    return predicted_sequences


def update_text(next_word):
    st.session_state.input += ' ' + next_word

# These functions run once
@st.cache_resource
def setup():
    nltk.download('punkt')
    os.environ["KERAS_BACKEND"] = "tensorflow"


@st.cache_resource
def import_shakespeare_lstm():
    model = tf.keras.models.load_model('exports/sherlock_LSTM.h5')
    max_sequence_len = 122

    with open('exports/word_to_index.json', 'r') as json_file:
        word_to_index = json.loads(json_file.read())
    with open('exports/index_to_word.json', 'r') as json_file:
        index_to_word = json.loads(json_file.read())

    return model, index_to_word, word_to_index, max_sequence_len


@st.cache_resource
def import_amazon_gru():
    model = tf.keras.models.load_model('exports/amazon_reviews_GRU.h5')
    max_sequence_len = 199

    file = open("exports/amzn_idx_to_word.pkl", 'rb')
    index_to_word = pickle.load(file)
    file.close()
    file = open("exports/amzn_word_to_idx.pkl", 'rb')
    word_to_index = pickle.load(file)
    file.close()

    return model, index_to_word, word_to_index, max_sequence_len


@st.cache_resource
def import_gpt():
    # Pretrained model
    preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
        "gpt2_base_en",
        sequence_length=128,
    )
    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
        "gpt2_base_en", preprocessor=preprocessor
    )

    st.session_state.gpt = gpt2_lm


setup()
model, index_to_word, word_to_index, max_sequence_len = import_amazon_gru()

if 'gpt' not in st.session_state:
    preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
        "gpt2_base_en",
        sequence_length=128,
    )
    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
        "gpt2_base_en", preprocessor=preprocessor
    )
    st.session_state.gpt = gpt2_lm

#import_gpt()
#compile_compute_graph()

st.title('Next Word Prediction')

st.divider()

num_words = st.slider(label='Number of words', min_value=1, max_value=20)

text = st.text_input(label='', key='input', placeholder='Type here...')

if text:
    if num_words == 1:
        next_words, probs = predict_next_words(model, text, max_sequence_len, word_to_index, index_to_word)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.button(next_words[0], on_click=update_text, args=(next_words[0],), use_container_width=True, key=0)
            #st.write(probs[0])
        with col2:
            st.button(next_words[1], on_click=update_text, args=(next_words[1],), use_container_width=True, key=1)
            #st.write(probs[1])
        with col3:
            st.button(next_words[2], on_click=update_text, args=(next_words[2],), use_container_width=True, key=2)
            #st.write(probs[2])
        with col4:
            st.button(next_words[3], on_click=update_text, args=(next_words[3],), use_container_width=True, key=3)
            #st.write(probs[3])
        with col5:
            st.button(next_words[4], on_click=update_text, args=(next_words[4],), use_container_width=True, key=4)
            #st.write(probs[4])
   
        start = time.time()
        gpt_prediction = st.session_state.gpt.generate(text, max_length=len(text)+50)
        end = time.time()

        st.write(gpt_prediction)
        st.write(f"Time: {end - start:.2f}s")

    else:
        next_words = predict_next_n_words(model, text, num_words, max_sequence_len, word_to_index, index_to_word)

        st.button(next_words[0], on_click=update_text, args=(next_words[0],), use_container_width=False, key=0)
        st.button(next_words[1], on_click=update_text, args=(next_words[1],), use_container_width=False, key=1)
        st.button(next_words[2], on_click=update_text, args=(next_words[2],), use_container_width=False, key=2)
        st.button(next_words[3], on_click=update_text, args=(next_words[3],), use_container_width=False, key=3)
        st.button(next_words[4], on_click=update_text, args=(next_words[4],), use_container_width=False, key=4)
        
        st.divider()

        start = time.time()
        gpt_prediction = st.session_state.gpt.generate(text, max_length=len(text)+150)
        end = time.time()

        #next_words = gpt_prediction.split(' ', len(text))[-1]
        #st.write(next_words)

        st.write(gpt_prediction)
        st.write(f"Time: {end - start:.2f}s")

# Ignore this
def hide():
    #elif num_words <= 10:
    next_words = predict_next_n_words(model, text, num_words, max_sequence_len, word_to_index, index_to_word)

    col1, col2 = st.columns(2)
    with col1:
        st.button(next_words[0], on_click=update_text, args=(next_words[0],), use_container_width=False, key=0)
        st.button(next_words[1], on_click=update_text, args=(next_words[1],), use_container_width=False, key=1)
        st.button(next_words[2], on_click=update_text, args=(next_words[2],), use_container_width=False, key=2)
        st.button(next_words[3], on_click=update_text, args=(next_words[3],), use_container_width=False, key=3)
        st.button(next_words[4], on_click=update_text, args=(next_words[4],), use_container_width=False, key=4)
    
    start = time.time()
    gpt_prediction = st.session_state.gpt.generate(text, max_length=len(text)+50)
    end = time.time()

    with col2:
        st.write(gpt_prediction)
        st.write(f"Time: {end - start:.2f}s")    
