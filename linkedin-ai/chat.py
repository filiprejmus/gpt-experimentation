import pinecone
import os
import openai
from tqdm.auto import tqdm
import pandas as pd
import time
import json


pc_api_key = os.environ.get("PINECONE_API_KEY")
openai.api_key = os.environ.get("OPENAI_API_KEY")

MODEL = "text-embedding-ada-002"

pinecone.init(api_key=pc_api_key, environment="gcp-starter")
if "openai" not in pinecone.list_indexes():
    pinecone.create_index("openai", dimension=1536)

index = pinecone.Index("openai")
# products = pd.read_csv("finger_food.csv")

# for i in tqdm(range(len(products))):
#     lines_batch = products.loc[i, "description"]
#     res = openai.Embedding.create(input=lines_batch, engine=MODEL)
#     embeds = [record["embedding"] for record in res["data"]]
#     meta = [{"price": products.loc[i, "price"], "name": products.loc[i, "name"]}]
#     to_upsert = zip([str(i)], embeds, meta)
#     index.upsert(vectors=list(to_upsert))

# query = "meats"
# start_query = time.time()
# xq = openai.Embedding.create(input=query, engine=MODEL)["data"][0]["embedding"]
# res = index.query([xq], top_k=5, include_metadata=True)
# end_query = time.time()
# for match in res["matches"]:
#     print(f"{match['score']:.2f}: {match['metadata']['name']}")

vector = index.fetch(["0"])["vectors"]["0"]
print(vector["metadata"])
