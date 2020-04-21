from flask import Flask, render_template, request
from dialogue_manager import *
 
 
    
app = Flask(__name__)
dialogue_manager = DialogueManager()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(dialogue_manager.generate_answer(userText))
 

if __name__ == "__main__":
    app.run(host='0.0.0.0')
