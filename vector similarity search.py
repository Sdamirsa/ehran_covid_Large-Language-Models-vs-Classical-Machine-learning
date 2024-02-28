# loading the libraries
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from datasets import load_dataset
import pandas as pd
from datasets import load_dataset

# loading and preparing the data
data = load_dataset("Moreza009/Internal_validation")

def merge_columns(example):
    example["prediction"] = example["patient medical hidtory"] + \
        " ----->: " + str(example["Inhospital Mortality"])
    return example

data['train'] = data['train'].map(merge_columns) #!!! plz check if this works # please print this data and send the picture to me plz

documents = data['train']["prediction"] #!!! rename the documents

# loading the embedding model
embeddings = HuggingFaceEmbeddings()
knowledge_base = FAISS.from_texts(documents, embeddings)

def merge_columns(example):
    example["prediction"] = example["patient medical hidtory"] + \
        " ----->: " + str(example["Inhospital Mortality"])
    return example

# finding the test accuracy using the retriver
data['test'] = data['test'].map(merge_columns) #!!! this is incorrect

data['test']["prediction"]

y_pred = []
for i in data['test']["prediction"]:
    documents = knowledge_base.similarity_search(i, k=1)
    characters = [doc.page_content for doc in documents]
    for doc, char in zip(documents, characters):
        string = f"{char}"
        if "survives" in string:
            y_pred.append(1)
        else:
            y_pred.append(0)
print(y_pred)


y_true = []
for i in data['test']["prediction"]:
    if "survives" in i:
        y_true.append(1)
    else:
        y_true.append(0)
result = pd.DataFrame({"y_pred": y_pred, "y_true": y_true})


result.to_excel("Vector_database_internal_results.xlsx", index=False)
