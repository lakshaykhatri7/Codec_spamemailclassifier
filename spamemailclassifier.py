import pandas as pd

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Check sample
print(df.head())


from sklearn.preprocessing import LabelEncoder

# Convert labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])  # spam=1, ham=0

# Optional text cleaning
import re
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

df['message'] = df['message'].apply(clean_text)


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['message'])
y = df['label']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))



def predict_spam(message):
    message_clean = clean_text(message)
    vector = tfidf.transform([message_clean])
    prediction = model.predict(vector)
    return "Spam" if prediction[0] == 1 else "Ham"

# Try a message
print(predict_spam("You won a lottery! Call now."))
