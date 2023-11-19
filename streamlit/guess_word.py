import random
import streamlit as st
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

import pickle

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

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
        predicted_words.append(index_to_word[idx])#[str(idx)].lower())
        probs.append(str(round(output[idx] * 100, 2)) + '%')

    word_to_prob = {}
    for i in range(len(probs)):
        word_to_prob[predicted_words[i]] = probs[i]

    #random.shuffle(predicted_words)

    return predicted_words, probs, word_to_prob

def predict_next_n_words(model, text, n, max_sequence_len, word_to_index, index_to_word):
    predicted_sequence = []

    for _ in range(n):
        # Tokenize the input string
        token_list = [word_to_index[word] for word in word_tokenize(text) if word in word_to_index]

        # Pad the token sequence
        token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')

        # Predict the token of the next word
        predicted_idx = np.argmax(model.predict(token_list), axis=-1)

        # Convert the token back to a word
        predicted_word = index_to_word.get(predicted_idx[0], '')

        # Append the predicted word to the sequence and to the text (for the next iteration)
        predicted_sequence.append(predicted_word)
        text += " " + predicted_word

    return ' '.join(predicted_sequence)

st.title('Word Guessing Game')
st.write('''

''')

def display_prompt(generated_text):
    st.session_state.generate = generated_text

with st.sidebar:
    st.divider()
    st.header("Instructions")
    st.write("Write or generate the prompt and guess the word that you think has the highest chance of coming next.")
    
    st.divider()
    st.header("Settings")
    prompt_type = st.selectbox(
        "Prompt Type",
        ("Generated Prompt", "Custom Prompt")
        )
    if prompt_type == "Generated Prompt":
        prompt_length = st.slider(
            "Prompt Length",
            1,10
        )
    st.divider()

model = tf.keras.models.load_model('amazon/amazon_reviews_GRU.h5')
max_sequence_len = 198

with open('amazon/amzn_idx_to_word.pkl', 'rb') as handle:
    index_to_word = pickle.load(handle)
with open('amazon/amzn_word_to_idx.pkl', 'rb') as handle:
    word_to_index = pickle.load(handle)

if prompt_type == "Generated Prompt":
    col1, col2 = st.columns([5,1])
    generate_prompt = col2.button(label="Generate", use_container_width=True)
    if generate_prompt:
        generated_text = index_to_word[random.randint(0,len(index_to_word)-1)]
        generated_text = generated_text + " " + predict_next_n_words(model, generated_text, prompt_length-1, max_sequence_len, word_to_index, index_to_word)
        display_prompt(generated_text.capitalize())
    text = col1.text_input(label="PLACEHOLDER", value="", key='generate', placeholder='Generate...', disabled=True, label_visibility="collapsed")
else:
    text = st.text_input(label="PLACEHOLDER", key='input', placeholder='Type prompt here...', disabled=False, label_visibility="collapsed")

if text:
    def show_answers(placeholder):
        with col1:
            print(word_to_prob[next_words[0]][:-1], max_prob)
            if word_to_prob[next_words[0]][:-1]==str(max_prob):
                st.markdown(f''':green[{word_to_prob[next_words[0]]}]''')
            else:
                st.markdown(f''':red[{word_to_prob[next_words[0]]}]''')
        with col2:
            if word_to_prob[next_words[1]][:-1]==str(max_prob):
                st.markdown(f''':green[{word_to_prob[next_words[1]]}]''')
            else:
                st.markdown(f''':red[{word_to_prob[next_words[1]]}]''')
        with col3:
            if word_to_prob[next_words[2]][:-1]==str(max_prob):
                st.markdown(f''':green[{word_to_prob[next_words[2]]}]''')
            else:
                st.markdown(f''':red[{word_to_prob[next_words[2]]}]''')
        with col4:
            if word_to_prob[next_words[3]][:-1]==str(max_prob):
                st.markdown(f''':green[{word_to_prob[next_words[3]]}]''')
            else:
                st.markdown(f''':red[{word_to_prob[next_words[3]]}]''')
        with col5:
            if word_to_prob[next_words[4]][:-1]==str(max_prob):
                st.markdown(f''':green[{word_to_prob[next_words[4]]}]''')
            else:
                st.markdown(f''':red[{word_to_prob[next_words[4]]}]''')

    def create_guess(random_index):
        button1 = st.button(next_words[random_index], on_click=show_answers, args=(next_words[random_index],), use_container_width=True)
        
    next_words, probs, word_to_prob = predict_next_words(model, text, max_sequence_len, word_to_index, index_to_word)

    max_prob = 0
    for i in range(len(probs)):
        if float(probs[i][:-1]) > max_prob:
            max_prob = float(probs[i][:-1])

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        create_guess(0)
    with col2:
        create_guess(1)
    with col3:
        create_guess(2)
    with col4:
        create_guess(3)
    with col5:
        create_guess(4)