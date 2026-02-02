# # from sklearn import model_selection
# # print(dir(model_selection))


# # import os
# # from sklearn.model_selection import train_test_split
# # import logging

# import os
# import logging
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from nltk.stem.porter import PorterStemmer
# from nltk.corpus import stopwords
# import string
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('punkt_tab')

# def transform_text(text):
#     """
#     Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
#     """
#     ps = PorterStemmer()
#     # Convert to lowercase
#     text = text.lower()
#     # Tokenize the text
#     text = nltk.word_tokenize(text)
#     # Remove non-alphanumeric tokens
#     text = [word for word in text if word.isalnum()]
#     # Remove stopwords and punctuation
#     text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
#     # Stem the words
#     text = [ps.stem(word) for word in text]
#     # Join the tokens back into a single string
#     return " ".join(text)


# transform_text("You've won tkts to the EURO2004 CUP FINAL or å£800 CASH, to collect CALL 09058099801 b4190604, POBOX 7876150ppm")


import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import yaml
from dvclive import Live