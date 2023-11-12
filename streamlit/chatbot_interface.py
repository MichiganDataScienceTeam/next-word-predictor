import streamlit as st
import tensorflow as tf
import pickle

# Adjust as necessary
model = tf.keras.models.load_model('../model.h5')
index_to_word = pickle.load(open('index_to_word.pkl', 'rb'))
word_to_index = pickle.load(open('word_to_index.pkl', 'rb'))

MAX_SEQUENCE_LEN = 122

### USAGE: python -m streamlit run chatbot_interface.py

def predict_next_word(input: str) -> str:
    input_token = [word_to_index[word] if word in word_to_index else 0 for word in input.split()]
    input_token = tf.keras.preprocessing.sequence.pad_sequences([input_token], maxlen=MAX_SEQUENCE_LEN, padding='pre')
    prediction = model.predict(input_token)

    return index_to_word[prediction.argmax()]

if __name__ == "__main__":
    st.title("Next Word Predictor") 

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Type in a sequence of words, I'll predict the next word for you!"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        msg = {"role": "assistant", "content": predict_next_word(prompt)}
        st.session_state.messages.append(msg)
        st.chat_message("assistant").write(msg['content'])
    