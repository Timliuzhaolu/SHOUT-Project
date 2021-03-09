import pandas as pd
import random
import numpy as np
import simpletransformers
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import wandb
from sklearn.metrics import accuracy_score
from lime.lime_text import LimeTextExplainer
import pickle


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
from collections import Counter


df = pd.read_csv('document_level_labels/age_13_id69.csv')
df = df[['encoded_message', '69']]
df = df[df['69'] == 1]
df = df.reset_index()
length = df.shape[0]
samples = random.sample(range(1, length), 100)
model = ClassificationModel(
    'longformer',
    'doc_multiclass_cls/age_13',
    num_labels=2
) 
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
def predict_probs(texts):
    predictions = model.predict(texts)
    x = np.array(predictions[1])
    return np.apply_along_axis(softmax, 1, x)

explainer = LimeTextExplainer()

words = []
for i in samples:    
    text = df['encoded_message'][i]
    label = df['69'][i]
    label = label.astype('int64')

    exp = explainer.explain_instance(text, 
                                 predict_probs, 
                                 num_features = 6, 
                                 num_samples = 3000,
                                 labels=(label,)
                                )
    
    words_dict = exp.as_list(label = label)
    positive = [(x,y) for (x, y) in words_dict if y > 0]
    first_two = positive[:2]
    two_words = [x for (x, y) in first_two]

    words.extend(two_words)
    
with open("Wordclouds/age_13.txt", "wb") as fp:   #Pickling
    pickle.dump(words, fp)


with open("Wordclouds/age_13.txt", "rb") as fp:   # Unpickling
    a = pickle.load(fp)

lowercase_list = [x.lower() for x in a]

word_could_dict = Counter(lowercase_list)
wordcloud = WordCloud(stopwords = stop_words, width = 1000, height = 500, background_color="white").generate_from_frequencies(word_could_dict)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()





