# We used CAST to identify topics. More instructions see https://github.com/yananma1029/CAST
from octis.dataset.dataset import Dataset # load preprocessed data
from CAST import CAST
import json

# We used octis to preprocess data and then load the preprocessed data
dataset = Dataset()
dataset.load_custom_dataset_from_folder("octis_dataset/your dataset folder")
corpus = dataset.get_corpus()
documents = [" ".join(words) for words in corpus]

MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'  # support other sentence embedding models

params = {"nr_topics": 20,
          "n_dimensions": 5,
          "min_cluster_size": 5,
          "min_count": 25,
          "candidate_mode": 'word_level',
          "self_sim_threshold" : 0.4, # Threshold to filter out functional words.
          }

topic_model = CAST(documents=documents, **params)

twords, _ = topic_model.pipeline()
twords_list = list(twords.values())

top_sen = topic_model.search_docs_by_topic(topic_number=None, num_docs=10) # Identify top-10 most representative sentences
sen_list = top_sen['Top_Sentences'].tolist()

result = {
    "Model": MODEL_NAME,
    "Dataset Size": len(documents),
    "Params": params,
    "Topic Words": twords_list,
    "Top Sentences": sen_list,}

with open(f"your saved file name", "w") as file:
    json.dump(result, file)



