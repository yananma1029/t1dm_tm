import os
import string
# We use octis library for data preprocessing more info see https://github.com/MIND-Lab/OCTIS
from octis.preprocessing.preprocessing import Preprocessing
path = os.getcwd()

# Initialize preprocessing
preprocessor = Preprocessing(vocabulary=None, max_features=None, remove_numbers = True,
                             remove_punctuation=True, remove_stopwords_spacy = True,
                             lemmatize=True, 
                             min_chars=1, min_words_docs=1,
                             split = False, save_original_indexes = True)
# preprocess

documents_path = os.path.join(path, 'your file path')
dataset = preprocessor.preprocess_dataset(documents_path)

# save the preprocessed dataset
save_path = os.path.join(path, 'your saved file path')
dataset.save(save_path)
