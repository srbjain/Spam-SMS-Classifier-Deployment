import pandas as pd
import nltk
import re
import pickle
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
le = WordNetLemmatizer()
nltk.download('wordnet')

filename = 'nlp_model.pkl'

dataset = pd.read_csv("smsspamcollection/SMSSpamCollection", sep="\t", names=["label","sms"])

corpus=[]
for i in range (len(dataset)):
    review = re.sub("[^a-zA-Z]", " ", dataset["sms"][i])
    review = review.lower()
    review =review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words("english")]
    review = " ".join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
pickle.dump(cv, open("transform.pkl", "wb"))

dataset["Spam"] = dataset["label"].map({"ham":0, "spam":1})
y = dataset["Spam"]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)



from sklearn.naive_bayes import MultinomialNB
cl = MultinomialNB()
cl.fit(X_train,y_train)
cl.score(X_train,y_train)
pickle.dump(cl, open(filename, "wb"))



#y_pred = cl.predict(X_test)

#from sklearn.metrics import confusion_matrix, accuracy_score
#cm = confusion_matrix(y_test, y_pred)
#print(accuracy_score(y_test, y_pred))
