
# Import required libraries for NLP, model loading, and GUI
import json
import pickle
import random
import threading
import importlib

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tkinter import *

try:
    sr = importlib.import_module("speech_recognition")
    pyttsx3 = importlib.import_module("pyttsx3")
    VOICE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    sr = None
    pyttsx3 = None
    VOICE_AVAILABLE = False

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

# Retrain automatically when the saved model belongs to a different intent set.
try:
    configured_intents = json.loads(open('intents.json', encoding='utf-8').read())
    configured_classes = sorted(intent['tag'] for intent in configured_intents['intents'])
    saved_classes = pickle.load(open('classes.pkl', 'rb'))
    if model is not None and saved_classes != configured_classes:
        print("Saved model does not match intents.json; retraining is required")
        model = None
except (FileNotFoundError, json.JSONDecodeError, KeyError, pickle.PickleError):
    model = None


if model is None:
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
else:
    print("Loaded existing chatbot model")

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
    if not ints:
        return "I could not understand the question. Please ask about admissions, courses, exams, tuition, student services, or university contacts."

    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


# Handle sending a message from the GUI to the chatbot.
def send(event=None):
    msg = EntryBox.get("1.0", 'end-1c').strip()
    if not msg:
        return "break"

    EntryBox.delete("1.0", END)
    append_message("You", msg, "user")
    set_status("Thinking", "#f0b429")
    SendButton.config(state=DISABLED)
    root.after(180, respond, msg)
    return "break"


def respond(msg):
    ints = predict_class(msg)
    response = getResponse(ints, intents)
    append_message("SCUT Assistant", response, "bot")
    if VOICE_AVAILABLE:
        threading.Thread(target=speak_response, args=(response,), daemon=True).start()
    set_status("Online", "#20b486")
    SendButton.config(state=NORMAL)
    EntryBox.focus_set()


def append_message(author, message, message_type):
    ChatBox.config(state=NORMAL)
    ChatBox.insert(END, author + "\n", message_type + "_author")
    ChatBox.insert(END, message + "\n\n", message_type + "_message")
    ChatBox.config(state=DISABLED)
    ChatBox.yview(END)


def set_status(text, color):
    StatusLabel.config(text="●  " + text, foreground=color)


def speak_response(response):
    try:
        speaker = pyttsx3.init()
        speaker.setProperty("rate", 165)
        speaker.say(response)
        speaker.runAndWait()
    except Exception as error:
        print("Voice output unavailable:", error)


def listen_and_send():
    if not VOICE_AVAILABLE:
        append_message("System", "Voice mode needs SpeechRecognition, PyAudio, and pyttsx3. Text mode is still available.", "bot")
        return

    VoiceButton.config(state=DISABLED)
    set_status("Listening", "#f0b429")
    threading.Thread(target=voice_worker, daemon=True).start()


def voice_worker():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as microphone:
            recognizer.adjust_for_ambient_noise(microphone, duration=0.5)
            audio = recognizer.listen(microphone, timeout=5, phrase_time_limit=12)
        message = recognizer.recognize_google(audio, language="en-US")
        root.after(0, lambda: send_voice_message(message))
    except sr.WaitTimeoutError:
        root.after(0, lambda: voice_failed("No speech was detected. Please try again."))
    except sr.UnknownValueError:
        root.after(0, lambda: voice_failed("I could not understand the audio. Please speak in English."))
    except Exception as error:
        print("Voice input unavailable:", error)
        root.after(0, lambda: voice_failed("The microphone is unavailable. You can use text mode instead."))


def send_voice_message(message):
    VoiceButton.config(state=NORMAL)
    append_message("You", message, "user")
    set_status("Thinking", "#f0b429")
    SendButton.config(state=DISABLED)
    root.after(180, respond, message)


def voice_failed(message):
    VoiceButton.config(state=NORMAL)
    set_status("Online", "#20b486")
    append_message("System", message, "bot")


# Compact university-themed interface for a phone-sized window.
root = Tk()
root.title("SCUT Student Assistant")
root.geometry("360x600")
root.minsize(320, 480)
root.configure(bg="#edf2f5")

NAVY = "#102a43"
INK = "#17324d"
MUTED = "#718096"
GOLD = "#f0b429"
TEAL = "#20b486"
PANEL = "#ffffff"

Header = Frame(root, bg=NAVY, height=66)
Header.pack(fill=X)
Header.pack_propagate(False)

Brand = Frame(Header, bg=NAVY)
Brand.pack(side=LEFT, padx=16, pady=8)
Label(Brand, text="SCUT  |  Student Assistant", bg=NAVY, fg=GOLD,
    font=("Segoe UI", 13, "bold")).pack(anchor=W)

StatusLabel = Label(Header, text="●  Online", bg=NAVY, fg=TEAL, font=("Segoe UI", 10, "bold"))
StatusLabel.pack(side=RIGHT, padx=14, pady=20)

Content = Frame(root, bg="#edf2f5")
Content.pack(fill=BOTH, expand=True, padx=8, pady=(8, 8))

ChatArea = Frame(Content, bg=PANEL, highlightbackground="#d6e0e8", highlightthickness=1)
ChatArea.pack(fill=BOTH, expand=True)

scrollbar = Scrollbar(ChatArea, relief=FLAT, bd=0, bg="#c8d4de", activebackground="#9fb3c2")
scrollbar.pack(side=RIGHT, fill=Y)
ChatBox = Text(ChatArea, bd=0, bg=PANEL, padx=12, pady=12, wrap=WORD,
               font=("Segoe UI", 10), yscrollcommand=scrollbar.set,
               cursor="xterm", state=DISABLED)
ChatBox.pack(side=LEFT, fill=BOTH, expand=True)
scrollbar.config(command=ChatBox.yview)
ChatBox.tag_configure("user_author", foreground="#27736e", font=("Segoe UI", 9, "bold"))
ChatBox.tag_configure("user_message", foreground=INK, lmargin1=20, lmargin2=20, spacing3=7)
ChatBox.tag_configure("bot_author", foreground="#b57a00", font=("Segoe UI", 9, "bold"))
ChatBox.tag_configure("bot_message", foreground=INK, lmargin1=20, lmargin2=20, spacing3=10)
append_message("SCUT Assistant", "Welcome to the SCUT Student Assistant. I am ready to help.", "bot")

Composer = Frame(Content, bg=PANEL, highlightbackground="#d6e0e8", highlightthickness=1)
Composer.pack(fill=X)
EntryBox = Text(Composer, bd=0, bg=PANEL, fg=INK, insertbackground=TEAL,
                height=2, wrap=WORD, font=("Segoe UI", 10), padx=9, pady=8)
EntryBox.pack(side=LEFT, fill=BOTH, expand=True)
SendButton = Button(Composer, text="Send", command=send, bg=TEAL, fg="white",
                    activebackground="#168c68", activeforeground="white", relief=FLAT,
                    bd=0, padx=12, pady=10, font=("Segoe UI", 9, "bold"), cursor="hand2")
SendButton.pack(side=RIGHT, padx=5, pady=6)
VoiceButton = Button(Composer, text="Voice", command=listen_and_send, bg="#f7f9fb", fg=INK,
                     activebackground="#dcecf0", activeforeground=INK, relief=FLAT,
                     bd=0, padx=8, pady=10, font=("Segoe UI", 9, "bold"), cursor="hand2")
VoiceButton.pack(side=RIGHT, padx=(0, 0), pady=6)
EntryBox.bind("<Return>", send)


def quick_question(topic):
    questions = {
        "Admissions": "Where can I find admission information?",
        "Course registration": "How do I register for courses?",
        "Exam schedule": "Where can I find the exam schedule?",
        "Student portal": "How do I access the student portal?"
    }
    EntryBox.insert("1.0", questions[topic])
    EntryBox.focus_set()


root.mainloop()
