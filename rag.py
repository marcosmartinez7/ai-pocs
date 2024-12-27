import pandas as pd
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os



df = pd.read_csv('./top_rated_wines.csv')
df = df[df['variety'].notna()] # remove any NaN values as it blows up serialization
data = df.to_dict('records')
#print(df)


encoder = SentenceTransformer('all-MiniLM-L6-v2') # Model to create embeddings
qdrant = QdrantClient(":memory:") # Create in-memory Qdrant instance


# Create collection to store wines
qdrant.recreate_collection(
    collection_name="top_wines",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(), # Vector size is defined by used model
        distance=models.Distance.COSINE
    )
)

# vectorize
qdrant.upload_records(
    collection_name="top_wines",
    records=[
        models.Record(
            id=idx,
            vector=encoder.encode(doc["region"]).tolist(),
            payload=doc
        ) for idx, doc in enumerate(data) # data is the variable holding all the wines
    ]
)

# Search uruguay 
hits = qdrant.search(
    collection_name="top_wines",
    query_vector=encoder.encode("A wine from Montevideo, Uruguay").tolist(),
    limit=3
)
for hit in hits:
  print(hit.payload, "score:", hit.score)

search_results = [hit.payload for hit in hits]

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are chatbot, a wine specialist. Your top priority is to help guide users into selecting amazing wine and guide them with their requests."},
        {"role": "user", "content": "Suggest me two wines from montevideo uruguay, related with defensor sporting. Then another one not related with defensor, but from uruguay"},
        {"role": "assistant", "content": str(search_results)}
    ]
)
print(completion.choices[0].message)
