from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

filename = 'nlp_model.pkl'
cv = pickle.load(open("transform.pkl", "rb"))
cl = pickle.load(open(filename, "rb"))


app= Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == "POST":
        message = request.form["message"]
        data = [message]
        data = cv.transform(data).toarray()
        result = cl.predict(data)

    return render_template("result.html", prediction = result)
    

if __name__ == "__main__":
    app.run(debug=True)
    
