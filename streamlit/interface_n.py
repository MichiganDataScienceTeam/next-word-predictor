import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import nltk

nltk.download('punkt')

st.markdown("""
<style>
	[data-testid="stHeader"] {
		background-image: linear-gradient(90deg, rgb(57, 163, 161), rgb(170, 115, 199));
	}
</style>""",
unsafe_allow_html=True)

# Load dictionaries
file = open("idx_to_word_data.txt", 'rb')
idx_to_word = pickle.load(file)
file.close()

file = open("word_to_idx_data.txt", 'rb')
word_to_idx = pickle.load(file)
file.close()

max_sequence_len = 123

model = tf.keras.models.load_model('model.h5')

def predict_next_n_words(model, text, n, max_sequence_len, word_to_index, index_to_word):
    """
    Predict the next n words based on the input text.

    Args:
    - model (tf.keras.Model): Trained model for prediction.
    - text (str): Input string.
    - n (int): Number of words to predict.
    - max_sequence_len (int): Maximum length of input sequences.
    - word_to_index (dict): Mapping from words to their respective indices.
    - index_to_word (dict): Mapping from indices to their respective words.

    Returns:
    - str: Predicted sequence of words.
    """

    predicted_sequence = []

    for _ in range(n):
        # Tokenize the input string
        token_list = [word_to_index[word] for word in nltk.tokenize.word_tokenize(text) if word in word_to_index]

        # Pad the token sequence
        token_list = tf.keras.utils.pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

        # Predict the token of the next word
        predicted_idx = np.argmax(model.predict(token_list), axis=-1)

        # Convert the token back to a word
        predicted_word = index_to_word.get(predicted_idx[0], '')

        # Append the predicted word to the sequence and to the text (for the next iteration)
        predicted_sequence.append(predicted_word)
        text += " " + predicted_word

    return ' '.join(predicted_sequence)

with st.sidebar:
    st.markdown(
        "<h1>About Our Model</h1>",
        unsafe_allow_html=True,
    )
    "Natural Language Processing model that predicts the next 'n' words, specified by the user, given a user input."
    st.divider()
    st.markdown(
        "<h3>Designed by Michigan Data Science Team in Fall 2023</h3>",
        unsafe_allow_html=True,
    )
    one, two = st.columns(2)
    with one: 
        st.image("MDST_logo.jpg", width= 200)

st.title("➱ Next Word Predictor") 
st.divider()
number = st.slider("Pick a number of words to predict", 1, 100)
st.divider()
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hey there!👋 What sequence would you like to predict?"}]

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message(msg["role"], avatar=("./userImg.jpg")).write(msg["content"])
    else:
        st.chat_message(msg["role"]).write(msg["content"])

count = 0

if prompt := st.chat_input("Enter your message here", key = count):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar=("./userImg.jpg")).write(prompt)
    response = predict_next_n_words(model, prompt, number, max_sequence_len, word_to_idx, idx_to_word)
    st.session_state.messages.append({"role": "assistant", "content": prompt + " " + response})
    st.chat_message("assistant").write(prompt + " " + response)
    count += 1