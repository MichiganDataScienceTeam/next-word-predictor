import streamlit as st
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

import regex as re

import pickle
import json

import nltk
from nltk.tokenize import word_tokenize

import os
# import keras_nlp
# import keras_core as keras
import time


def update_text(next_word):
    st.session_state.input += ' ' + next_word


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

        j = 1
        while j < n:
            # Tokenize the input string
            token_list = [word_to_index[word] for word in word_tokenize(temp_text) if word in word_to_index]

            # Pad the token sequence
            token_list = tf.keras.utils.pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')

            # Predict the token of the next word
            output = np.array(model.predict(token_list))[0]
            
            #predicted_idx = np.argmax(output, axis=-1)
            predicted_idx = np.random.choice(np.argpartition(output, -1)[-1:], size=1)[0]

            # Convert the token back to a word
            predicted_word = index_to_word[predicted_idx].lower()

            # Append the predicted word to the sequence and to the text (for the next iteration)
            
            if predicted_word == predicted_sequence[-1]:
                predicted_idx = np.argpartition(output, -2)[-2]
                predicted_word = index_to_word[predicted_idx].lower()

            if predicted_word == ',':
                if predicted_sequence[-1] == '.' or predicted_sequence[-1] == '!' or \
                   predicted_sequence[-1] == '?' or predicted_sequence[-1] == ':':
                    predicted_sequence[-1] = ','
            else:
                predicted_sequence.append(predicted_word)
                j += 1
                
            temp_text += " " + predicted_word

        predicted_sequence = ' '.join(predicted_sequence)
        predicted_sequence = predicted_sequence.replace(' ,', ',')
        predicted_sequence = re.sub(r",+", ',', predicted_sequence)
        predicted_sequence = predicted_sequence.replace(' .', '.')
        predicted_sequence = predicted_sequence.replace(' !', '!')
        predicted_sequence = predicted_sequence.replace(' ?', '?')
        predicted_sequence = predicted_sequence.replace(' :', ':')
        predicted_sequence = predicted_sequence.replace(" '", "'")

        st.button(label=predicted_sequence, on_click=update_text, args=(predicted_sequence,), use_container_width=False, key=i)


def color_slider():
    st.markdown(
        f''' 
        <style> div.stSlider > div[data-baseweb = "slider"] > div[data-testid="stTickBar"] > div {{
            background: rgb(1 1 1 / 0%); 
        }} 
        </style> 
        <style> div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"]{{
            background-color: rgb(14, 38, 74); 
            box-shadow: rgb(14 38 74 / 20%) 0px 0px 0px 0.2rem;
        }} 
        </style> 
        <style> div.stSlider > div[data-baseweb="slider"] > div > div > div > div {{ 
            color: rgb(14, 38, 74); 
        }} 
        </style> 
        <style> div.stSlider > div[data-baseweb = "slider"] > div > div {{
            background: linear-gradient(
                to right, rgb(118, 17, 235) 0%, 
                rgb(118, 17, 235) {float(num_words-1)/(max_words-1)*100}%, 
                rgba(151, 166, 195, 0.25) {float(num_words-1)/(max_words-1)*100}%, 
                rgba(151, 166, 195, 0.25) 100%
            ); 
        }} 
        </style>
        ''', 
        unsafe_allow_html = True
    )  


def predict_with_gpt():
    start = time.time()
    gpt_prediction = st.session_state.gpt.generate(text, max_length=len(text)+150)
    end = time.time()

    #next_words = gpt_prediction.split(' ', len(text))[-1]
    #st.write(next_words)

    st.write(gpt_prediction)
    st.write(f"Time: {end - start:.2f}s")


# These functions run once
@st.cache_resource
def setup():
    nltk.download('punkt')
    os.environ["KERAS_BACKEND"] = "tensorflow"


@st.cache_resource
def import_shakespeare_lstm():
    model = tf.keras.models.load_model('model/model.h5')
    max_sequence_len = 122

    with open('model/word_to_index.json', 'r') as json_file:
        word_to_index = json.loads(json_file.read())
    with open('model/index_to_word.json', 'r') as json_file:
        index_to_word = json.loads(json_file.read())

    return model, word_to_index, index_to_word, max_sequence_len


@st.cache_resource
def import_amazon_gru():
    model = tf.keras.models.load_model('exports/amazon_reviews_GRU.h5')
    max_sequence_len = 198

    file = open("exports/amzn_idx_to_word.pkl", 'rb')
    index_to_word = pickle.load(file)
    file.close()
    file = open("exports/amzn_word_to_idx.pkl", 'rb')
    word_to_index = pickle.load(file)
    file.close()

    return model, index_to_word, word_to_index, max_sequence_len


@st.cache_resource
def import_gpt():
    """
    # Pretrained model
    preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
        "gpt2_base_en",
        sequence_length=128,
    )
    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
        "gpt2_base_en", preprocessor=preprocessor
    )

    st.session_state.gpt = gpt2_lm
    """


setup()
model, index_to_word, word_to_index, max_sequence_len = import_amazon_gru()

def hide():
    """
    if 'gpt' not in st.session_state:
        preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
            "gpt2_base_en",
            sequence_length=128,
        )
        gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
            "gpt2_base_en", preprocessor=preprocessor
        )
        st.session_state.gpt = gpt2_lm
    """

#import_gpt()
#compile_compute_graph()

st.title('Next Word Prediction')

st.divider()

max_words = 50
num_words = st.slider(label='Number of words', min_value=1, max_value=max_words)
#color_slider()

text = st.text_input(label='input', key='input', placeholder='Type here...', label_visibility="hidden")

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

    else:
        predict_next_n_words(model, text, num_words, max_sequence_len, word_to_index, index_to_word)

    st.divider()
    #predict_with_gpt()
