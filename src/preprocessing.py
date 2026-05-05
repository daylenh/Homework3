import re

def preprocess(texts):
    cleaned = []
    for t in texts:
        t = t.lower()
        # remove common task words
        t = re.sub(r'\b(summarize|summary|translate|translation|classify|classification|reason|analysis|analyze)\b', '', t)
        # remove punctuation
        t = re.sub(r'[^a-z\s]', '', t)
        # remove extra spaces
        t = re.sub(r'\s+', ' ', t).strip()
        cleaned.append(t)
    return cleaned