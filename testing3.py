import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the supplied SMS dataset. Run this script from the folder containing spam.csv.
df = pd.read_csv("spam.csv", encoding="latin-1")

# This dataset stores its label and message in the first two columns.
df = df.iloc[:, :2].copy()
df.columns = ["label", "message"]
df = df.dropna(subset=["label", "message"])
df["label"] = df["label"].astype(str).str.strip().str.lower()
df = df[df["label"].isin(["ham", "spam"])]
df["label"] = df["label"].map({"ham": 0, "spam": 1})

X_sms = df["message"].astype(str)
y_sms = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X_sms, y_sms, test_size=0.2, random_state=42, stratify=y_sms
)

# Fit the vectorizer only on training messages to avoid vocabulary leakage.
vectorizer = CountVectorizer()
X_train_count = vectorizer.fit_transform(X_train)
X_test_count = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_count, y_train)
y_pred = model.predict(X_test_count)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nConfusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

examples = [
    "Congratulations! You won a movie ticket!",
    "Hey, are you coming to class today?",
    "Free entry to win a prize, reply now!",
]
print("\nExample predictions:")
for message, prediction in zip(examples, model.predict(vectorizer.transform(examples))):
    print(f"{'SPAM' if prediction == 1 else 'HAM'}: {message}")
