
# Import required libraries for NLP, model loading, and GUI
import json
import pickle
import random

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tkinter import *

from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import SGD

nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

# Try to load the trained chatbot model if it exists.
# This avoids crashing when the file is imported by another script.
try:
    model = load_model('chatbot_model.h5')
except Exception:
    model = None


# Prepare the dataset from the chatbot intents file
words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']
intents_file = open('intents.json', encoding='utf-8').read()
intents = json.loads(intents_file)

# Loop through every intent and every pattern to build the vocabulary and training documents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each sentence into individual words
        word = nltk.word_tokenize(pattern)
        words.extend(word)

        # Store each pattern with its matching intent tag
        documents.append((word, intent['tag']))

        # Keep a unique list of all tags/classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print(documents)

# Normalize words: lowercase and lemmatize them
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))

# Sort the intent classes so the output index stays consistent
classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

# Save vocabulary and labels for later prediction use
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Build the training data: each sentence becomes a bag-of-words vector
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]

    for word in words:
        bag.append(1 if word in pattern_words else 0)

    output_row = output_empty.copy()
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)

train_x = np.array([t[0] for t in training])
train_y = np.array([t[1] for t in training])

print("Training data created")

# Model architecture: input layer -> hidden layer -> hidden layer -> output layer
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model using SGD optimizer
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model and save it to disk
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")

# Load the saved vocabulary and labels for the chatbot runtime
intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


# Clean the user input before prediction
def clean_up_sentence(sentence):
    # tokenize the pattern - splitting words into an array
    sentence_words = nltk.word_tokenize(sentence)
    # lower-case and reduce each word to its base form
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# Return a bag-of-words array: 1 if a word exists, otherwise 0
def bag_of_words(sentence, words, show_details=True):
    # Tokenize the input sentence
    sentence_words = clean_up_sentence(sentence)

    # Create a zero vector matching the vocabulary length
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                # Mark the position as present in the sentence
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % word)
    return np.array(bag)


# Predict the most likely intent for the given sentence
def predict_class(sentence):
    # Convert the sentence to its bag-of-words form
    p = bag_of_words(sentence, words, show_details=False)

    # Predict probabilities for every intent class
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    print(res)

    # Keep only predictions above the confidence threshold
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


# Return a random response for the predicted intent
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


# Handle sending a message from the GUI to the chatbot
def send():
    # Read the text from the input box and remove extra spaces
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "You: " + msg + '\n\n')
        ChatBox.config(foreground="#446665", font=("Verdana", 12))

        # Predict intent and choose a response
        ints = predict_class(msg)
        print(ints)
        res = getResponse(ints, intents)

        ChatBox.insert(END, "Bot: " + res + '\n\n')

        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)


# Create the main window
root = Tk()
root.title("Aiolearn Chatbot")
root.geometry("400x500")
root.resizable(width=TRUE, height=TRUE)

# Create Chat window
ChatBox = Text(root, bd=0, bg="white", height="8", width="50", font="Arial")
ChatBox.config(state=DISABLED)

# Bind scrollbar to Chat window
scrollbar = Scrollbar(root, command=ChatBox.yview, cursor="heart")
ChatBox['yscrollcommand'] = scrollbar.set

# Create the send button
SendButton = Button(root, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#f9a602", activebackground="#3c9d9b", fg='#000000',
                    command=send)

# Create the text input box
EntryBox = Text(root, bd=0, bg="white", width="29", height="5", font="Arial")
# EntryBox.bind("<Return>", send)


# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatBox.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

root.mainloop()
