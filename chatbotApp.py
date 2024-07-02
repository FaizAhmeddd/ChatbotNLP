import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import load_model

# Load necessary resources
lemmatizer = WordNetLemmatizer()
# nltk.download('punkt')
# nltk.download('wordnet')
model = load_model('chatbot_model.h5')

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
intents = json.loads(open('intents.json').read())

# Initialize chat history and suggestions
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []

# Function to update user input
def update_user_input(value):
    st.session_state.user_input = value

# Chatbot functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words) 
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = np.random.choice(i['responses'])
            suggestions = i['patterns']  # Suggestions for next possible question
            break
    return result, suggestions

# Streamlit GUI

st.title("E-Commerce Store Assistant")

def main():
    # Sidebar for suggestions
    st.sidebar.title("Suggestions")

    # Container for chat history
    chat_container = st.container()

    # Text input for user prompt
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    user_input = st.text_input("Prompt:", value=st.session_state.user_input, key="user_input")

    # Clear existing suggestions
    st.session_state.suggestions = []

    # Submit button
    if st.button("Send") and user_input.strip() != "":
        st.session_state.chat_history.append({"user": user_input})
        response, suggestions = chatbot_response(user_input)
        st.session_state.chat_history.append({"bot": response})
        if suggestions:
            st.session_state.suggestions = suggestions

    # Display suggestions as buttons
    for suggestion in st.session_state.suggestions:
        if st.sidebar.button(suggestion):
            update_user_input(suggestion)

    # Display entire chat history
    with chat_container:
        for item in st.session_state.chat_history:
            if "user" in item:
                st.image("user.png", width=32)
                st.markdown(f"{item['user']}")
            elif "bot" in item:
                st.image("aiChatbot.png", width=32)
                st.markdown(f"{item['bot']}")

   

def chatbot_response(text):
    ints = predict_class(text, model)
    if ints:
        res, suggestions = getResponse(ints, intents)
        return res, suggestions
    else:
        return "I'm sorry, I didn't understand that. Please refine your qeury!", None

if __name__ == "__main__":
    main()
