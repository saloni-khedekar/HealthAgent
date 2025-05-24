import os
import base64
import pymongo
import pandas as pd
import ollama
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_neo4j import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper

dotenv_path = "C:/Users/A/Desktop/Internship_Project/models1.env"
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    raise FileNotFoundError("models.env file not found!")

# Neo4j Credentials
NEO4J_URI = os.getenv("NEO4J_URI", "").strip()
NEO4J_USER = os.getenv("NEO4J_USER", "").strip()
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "").strip()
if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD:
    raise ValueError("Missing Neo4j credentials. Check your models.env file!")

# MongoDB Connection
MONGO_URI = "mongodb+srv://mayurr:12345@cluster0.hllwy4r.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(MONGO_URI)
db = client["fracsnet"]
products_collection = db["products"]

# Google Search Credentials
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "").strip()
if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
    raise ValueError("GOOGLE_API_KEY or GOOGLE_CSE_ID is missing!")


CSV_FILE_PATH = "C:/Users/A/Desktop/Internship_Project/Cleaned_Dataset.csv"
df = pd.read_csv(CSV_FILE_PATH)
df.fillna("", inplace=True)
df["combined_text"] = df.apply(lambda row: " ".join(row.astype(str)), axis=1)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = OllamaLLM(model="mistral", base_url="http://localhost:11434")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID)

def get_product_names():
    """Fetch all product names from MongoDB."""
    return [doc["product_name"] for doc in products_collection.find({}, {"product_name": 1})]

def get_product_image(product_name):
    """Retrieve base64-encoded image from MongoDB and decode it."""
    product_data = products_collection.find_one({"product_name": product_name}, {"image_base64": 1})
    if product_data and "image_base64" in product_data:
        return base64.b64decode(product_data["image_base64"])
    return None

def show_product_image(product_name):
    """Fetch and display the product image from MongoDB."""
    image_data = get_product_image(product_name)
    if image_data:
        img = Image.open(BytesIO(image_data))
        img.show()  # Open in default viewer
    else:
        print("No image found for the given product.")


def insert_product_with_embedding(tx, product_info, embedding):
    query = """
    MERGE (p:Product {ProductName: $ProductName})
    ON CREATE SET
        p.Nutrient_category = $Nutrient_category,
        p.Description = $Description,
        p.Contents = $Contents,
        p.Formulatedfor = $Formulated_For,
        p.Benefits = $Benefits,
        p.ProductImage = $Product_Image,
        p.PdfLink = $Pdf_link,
        p.embeddings = $embedding
    """
    tx.run(query, **product_info, embedding=embedding)

def store_embeddings_in_neo4j():
    """Store dataset embeddings in Neo4j for vector similarity search."""
    with driver.session() as session:
        for _, row in df.iterrows():
            product_info = row.to_dict()
            embedding = embedding_model.embed_query(row["combined_text"])
            session.write_transaction(insert_product_with_embedding, product_info, embedding)
    print("✅ Products with embeddings inserted into Neo4j!")


def get_neo4j_recommendations(query_text, top_k=5):
    query_embedding = embedding_model.embed_query(query_text)
    with driver.session() as session:
        results = session.run(
            """
            CALL db.index.vector.queryNodes('product_embeddings', $top_k, $embedding)
            YIELD node, score
            RETURN node.ProductName AS name, node.Description AS description,
                   node.ProductImage AS image, node.PdfLink AS pdf_link, score
            """,
            top_k=top_k, embedding=query_embedding
        )
        recommendations = [{
            "ProductName": record["name"],
            "Description": record["description"],
            "ProductImage": record["image"],
            "Reference": record["pdf_link"]
        } for record in results]
    return recommendations if recommendations else None

def dataset_recommendation(user_query):
    dataset_lower = df.astype(str).apply(lambda x: x.str.lower())
    mask = dataset_lower.apply(lambda col: col.str.contains(user_query.lower(), na=False, case=False))
    matched_rows = df[mask.any(axis=1)]
    return matched_rows.to_dict(orient="records") if not matched_rows.empty else None

recommendation_tool = Tool(
    name="Dataset Recommendation",
    func=dataset_recommendation,
    description="Fetches product recommendations from the dataset."
)

agent = initialize_agent(
    tools=[recommendation_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

def recommend_product(user_query):
    neo4j_recommendations = get_neo4j_recommendations(user_query, top_k=3)
    if neo4j_recommendations:
        return neo4j_recommendations
    dataset_results = dataset_recommendation(user_query)
    if dataset_results:
        return dataset_results

    print("No direct matches found. Performing Google Search...")
    return search.run(user_query)

def chat_with_agent():
    print("\nEnter a health-related query (or type 'exit' to quit):")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == "exit":
            print("Chatbot session ended.")
            break
        
        recommendations = recommend_product(user_input)
        prompt = f"User query: {user_input}. Based on the following recommendations:\n{recommendations}"
        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
        print("\nResponse:", response["message"]["content"])

if __name__ == "__main__":
    store_embeddings_in_neo4j()
    chat_with_agent()