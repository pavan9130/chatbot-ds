#importing required libraries

from flask import Flask, render_template,session, request,send_from_directory,current_app,send_file
import json
import pickle
import json
import os
import random
import nltk
import _datetime
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import numpy as np
from tensorflow.keras.models import load_model


#reading intents.json file and loading pickle files and model file
intents_file = open('intents.json',encoding='utf-8').read()
intents = json.loads(intents_file)
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('chatbot_model.h5')

#this function takes english sentence and gives  base word list 
def clean_up_sentence(sentence):
    # tokenize the pattern - splitting words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stemming every word - reducing to base form
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for words that exist in sentence taking reference of our 'words file'
def bag_of_words(sentence, words, show_details=True):
    # tokenizing patterns
    sentence_words = clean_up_sentence(sentence)
    # bag of words - vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,word in enumerate(words):
            if word == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % word)
    return(np.array(bag))



#this will predict all probabilities of each tag and selects max probability
def predict_class(sentence):
    # filter below  threshold predictions
    p = bag_of_words(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sorting strength probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        break
    return return_list

#this will get response based on tag
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    #print(tag)
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

# this takes user input and sends to above functions and sends back to HTML
# this also handle other functions 
def GetAnswer(user_response):
    answer=[]
    if user_response !='bye' and user_response !='yes' and user_response !='no':
                ints = predict_class(user_response)
                for (index, d) in enumerate(ints):
                   answer.append(d['intent'])                 
                if (len(ints)==0):
                 answer.append('status')
                 answer.append('Sorry I am still learning. Please elaborate or ask other Question!')
                else:
                 answer.append(getResponse(ints, intents))
                 session['question']=user_response
                 session['response']=getResponse(ints, intents)
    elif(user_response=='yes' or user_response=='YES' ):
        answer.append('status')
        answer.append('Good to know that')
    elif(user_response=='no' or user_response=='NO' ):
        q = open("Questions.txt", "a")
        q.write("\n")
        q.write('Q'+':'+session['question'])
        q.write("\n")
        q.write('A'+':'+session['response'])
        q.write("\n")
        q.close()
        answer.append('status')
        answer.append('Sorry I am still learning . I will update myself soon')
    else:
        answer.append('status')
        answer.append("Bye! take care..")
    return answer


app = Flask(__name__,template_folder='Templates')
app.static_folder = 'static'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = 'chatbot123'

#this will start log file and also renders HTML
@app.route("/")
def main():
     f = open("myfile.txt", "a")
     f.write("\n")
     f.write("\n")
     f.write('%s'%_datetime.datetime.today())
     f.write("\n")
     f.close()
     return render_template("main.html")

# this will get response and also created log file
@app.route("/get")
def get_bot_response():
     userText = request.args.get('msg')
     userText=userText.lower()
     li=GetAnswer(userText)
     f = open("myfile.txt", "a")
     #f.write('%s'%_datetime.datetime.today())
     #f.write("\n")
     f.write('You'+':'+userText)
     f.write("\n")
     f.write('Robo'+':'+str(li[1]))
     f.write("\n")
     f.close()
     return json.dumps(li)

#This will download only unsatisfied questions

@app.route("/downloadQAFile")
def downloadQAFile ():
    #For windows you need to use drive name [ex: F:/Example.pdf]
    path = "Questions.txt"
    return send_file(path, as_attachment=True, cache_timeout=0)

#this will download entire conversation

@app.route("/downloadLogFile")
def downloadLogFile ():
    #For windows you need to use drive name [ex: F:/Example.pdf]
    path = "myfile.txt"
    return send_file(path, as_attachment=True, cache_timeout=0)

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r
if __name__ == "__main__":
    app.run() 