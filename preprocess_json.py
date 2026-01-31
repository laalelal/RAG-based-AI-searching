'''import requests
import os
import json

def create_embedding(text_list):
    r=requests.post("http://localhost:11434/api/embed",json={
        "model":"bge-m3",
        "input":text_list
    })

    embedding = r.json()["embeddings"]
    return embedding


jsons=os.listdir("jsons") # list all the jsons
#print(jsons)
my_dicts=[]
chunk_id=0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content=json.load(f)
    embeddings=create_embedding([c['text'] for c in content['chunks']])

    for i,chunk in enumerate(content['chunks']):
        
        chunk['chunk_id']=chunk_id
        chunk['embedding']=embeddings[i]
        chunk_id+=1
        my_dicts.append(chunk)
        print(chunk)
    break  

print(my_dicts)
a=create_embedding(["cat sat on the mat","Aryan dances on a mat"])
print(a)
#print(len(a))        # 2
#print(len(a[0]))     # ~1024
#print(len(a[1]))

'''
'''import requests
import os
import json

def create_embedding(text_list):
    r=requests.post("http://localhost:11434/api/embed",json={
        "model":"bge-m3",
        "input":text_list
    })

    embedding = r.json()["embeddings"]
    return embedding


jsons=os.listdir("jsons") # list all the jsons
#print(jsons)
my_dicts=[]
chunk_id=0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content=json.load(f)
    embeddings=create_embedding([c['text'] for c in content['chunks']])

    for i,chunk in enumerate(content['chunks']):
        
        chunk['chunk_id']=chunk_id
        chunk['embedding']=embeddings[i]
        chunk_id+=1
        my_dicts.append(chunk)
        print(chunk)
    break  

print(my_dicts)
a=create_embedding(["cat sat on the mat","Aryan dances on a mat"])
print(a)
#print(len(a))        # 2
#print(len(a[0]))     # ~1024
#print(len(a[1]))


import requests
import os
import json
import math

def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )

    if r.status_code != 200:
        raise RuntimeError(f"Ollama HTTP error {r.status_code}: {r.text}")

    data = r.json()

    if "embeddings" not in data:
        raise RuntimeError(f"Ollama error: {data}")

    #  NaN protection
    for emb in data["embeddings"]:
        if any(isinstance(v, float) and math.isnan(v) for v in emb):
            raise RuntimeError("NaN detected in embedding")

    return data["embeddings"]


jsons = os.listdir("jsons")
my_dicts = []
chunk_id = 0

BATCH_SIZE = 1   #  CRITICAL FOR YOUR DATA

for json_file in jsons:
    with open(f"jsons/{json_file}", encoding="utf-8") as f:
        content = json.load(f)

    chunks = content["chunks"]

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        embeddings = create_embedding(texts)

        for j, chunk in enumerate(batch):
            chunk["chunk_id"] = chunk_id
            chunk["embedding"] = embeddings[j]

            my_dicts.append(chunk)
            print(chunk)

            chunk_id += 1

    break   # same behavior as your original code is this successfully runs? '''
import requests
import os
import json
import math
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity  

MIN_TEXT_LEN = 20      # skip very small texts
BATCH_SIZE = 3

def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "bge-m3", "input": text_list}
    )

    if r.status_code != 200:
        raise RuntimeError(r.text)

    data = r.json()

    if "embeddings" not in data:
        raise RuntimeError(data)

    # NaN check
    for emb in data["embeddings"]:
        if any(math.isnan(v) for v in emb):
            raise RuntimeError("NaN embedding")

    return data["embeddings"]


jsons = os.listdir("jsons")
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}", encoding="utf-8") as f:
        content = json.load(f)

    print(f"create embeddings for {json_file}...")

    chunks = content["chunks"]

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]

        # filter tiny texts
        batch = [c for c in batch if len(c["text"].strip()) >= MIN_TEXT_LEN]
        if not batch:
            continue

        texts = [c["text"] for c in batch]

        try:
            embeddings = create_embedding(texts)

        except RuntimeError as e:
            #  fallback: try one-by-one
            for c in batch:
                try:
                    emb = create_embedding([c["text"]])[0]
                    c["chunk_id"] = chunk_id
                    c["embedding"] = emb
                    my_dicts.append(c)
                    #print(c)
                    chunk_id += 1
                except:
                    print(" Skipped bad chunk:", c["text"][:50])
            continue

        for j, chunk in enumerate(batch):
            chunk["chunk_id"] = chunk_id
            chunk["embedding"] = embeddings[j]
            chunk_id += 1
            my_dicts.append(chunk)
    
            


#print(my_dicts)
df=pd.DataFrame.from_records(my_dicts)
#print(df)
# save the dataframe
joblib.dump(df,'embeddings.joblib')

#print(" no error throws")