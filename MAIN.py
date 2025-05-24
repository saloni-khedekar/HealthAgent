import streamlit as st
import requests
import pymongo
import pandas as pd
import base64
import os
import re
from dotenv import load_dotenv # Corrected typo here
from neo4j import GraphDatabase
from PIL import Image
from io import BytesIO
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool # Kept for context, but direct tool call is now preferred
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_community import GoogleSearchAPIWrapper

# --- BEGIN SYSTEM AND USER PROMPT DEFINITIONS ---
SYSTEM_PROMPT = """
You are a Health Product Recommendation Agent with specialized capabilities. Your role is to intelligently route queries and provide appropriate responses based on the following strict guidelines:

1. QUERY CLASSIFICATION:
   - First determine the query type:
     a) Health Product Request (e.g. "supplements for sleep")
     b) Health Information Request (e.g. "what is insomnia?")
     c) Non-Health Query (e.g. "weather today")

2. RESPONSE PROTOCOLS:

For HEALTH PRODUCT REQUESTS (a):
- You should indicate that the 'HealthProductRecommender' tool needs to be used.
- You MUST state the specific input for the tool formatted as "username|actual query for products". For example: "Action: Use HealthProductRecommender. Input: john_doe|supplements for better sleep".
- Do NOT try to invent products or provide recommendations directly if it's a product request; your role is to trigger the tool.
- The application displaying your response will handle the actual tool call, product card presentation, safety disclaimer, and "Need more specifics?" prompt based on your action.

For HEALTH INFORMATION REQUESTS (b):
- DO NOT recommend products or trigger the product recommendation tool.
- Use your Grok LLM knowledge to provide:
  1) A concise definition/explanation for the health information query.
  2) Append the following note verbatim: "Note: This chatbot specializes in health product recommendations. For personalized supplement advice, please ask about specific needs related to products or supplements."

For NON-HEALTH QUERIES (c):
- Immediately respond with:
  "This system only handles health-related inquiries. Your query about [topic] cannot be processed. Please ask about health products or conditions." You should try to infer [topic] from the user's query.

3. CLARIFICATION PROTOCOL:
When queries are ambiguous (e.g. "I'm tired"):
- Ask: "To help me understand your needs better, are you seeking product recommendations for [interpreted need from query, e.g., 'tiredness' or 'fatigue'], or are you looking for general health information about it?"
- Wait for user clarification before proceeding. Your response should solely be this clarification question.

4. SAFETY MECHANISMS:
- For HEALTH INFORMATION REQUESTS, your generated information should be factual and general. Always use phrases like "may support," "is thought to," or "some studies suggest" rather than definitive medical claims like "will treat" or "cures."
- The application will prepend a safety disclaimer to actual product recommendations shown to the user.

5. AGENT ARCHITECTURE (Conceptual for your understanding):
- You are part of a system with two conceptual agents:
  A) Product Recommender (tool-enabled, which you trigger via the 'Action:' directive)
  B) Health Info Agent (this is you, when you provide information directly using your Grok LLM capabilities)

Example Execution Flow (Your expected output):

User: "What causes diabetes?"
Your Output:
"Diabetes is a chronic condition characterized by high levels of blood glucose (sugar). It occurs when the pancreas doesn't produce enough insulin, or when the body cannot effectively use the insulin it produces. Insulin is a hormone that regulates blood sugar. There are several types of diabetes, including Type 1, Type 2, and gestational diabetes, each with different underlying causes and risk factors.
Note: This chatbot specializes in health product recommendations. For personalized supplement advice, please ask about specific needs related to products or supplements."

User: "Best vitamins for runners" (Username is jane_doe)
Your Output:
"Action: Use HealthProductRecommender. Input: jane_doe|best vitamins for runners"

User: "I'm feeling down" (Username is mark_p)
Your Output:
"To help me understand your needs better, are you seeking product recommendations for 'feeling down' or 'low mood', or are you looking for general health information about it?"
"""

# USER_INPUT_PROMPT_TEMPLATE is integrated into the main llm_input_prompt construction
# in the chat section.

# --- END SYSTEM AND USER PROMPT DEFINITIONS ---


st.set_page_config(page_title="HealthAgent", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

        body {
            font-family: 'Roboto', sans-serif;
            background-color: #F0F2F5;
        }
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #1E3A5F;
            font-weight: 500;
        }
        h1 { font-size: 2.8em; margin-bottom: 0.5em; text-align: center; }
        h2 { font-size: 2em; margin-bottom: 0.7em; color: #275A8D; }
        h3 { font-size: 1.6em; margin-bottom: 0.5em; color: #275A8D; }
        .stButton>button {
            background-color: #0066CC;
            color: white;
            padding: 0.75em 1.5em;
            border: none;
            border-radius: 8px;
            font-weight: 500;
            font-size: 1em;
            transition: background-color 0.3s ease, transform 0.1s ease;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .stButton>button:hover {
            background-color: #0052A3;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        .stButton>button:active {
            background-color: #004080;
            transform: translateY(0px);
        }
        .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div {
            background-color: #FFFFFF;
            border: 1px solid #D1D9E1;
            padding: 12px 15px;
            border-radius: 8px;
            font-size: 1em;
            color: #333333;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
        }
        .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus, .stSelectbox>div>div:focus-within {
            border-color: #0066CC;
            box-shadow: 0 0 0 0.2rem rgba(0,102,204,.25);
        }
        .css-1d391kg { /* Sidebar class */
            background-color: #FFFFFF;
            border-right: 1px solid #E0E7EF;
            box-shadow: 2px 0 10px rgba(0,0,0,0.05);
        }
        .css-1d391kg .stRadio label { /* Sidebar radio labels */
            padding: 0.8em 1em;
            border-radius: 6px;
            margin-bottom: 0.5em;
            transition: background-color 0.2s ease, color 0.2s ease;
            font-size: 1.05em;
            color: #1E3A5F;
        }
        .css-1d391kg .stRadio label:hover {
            background-color: #E6F0FA;
            color: #0066CC;
        }
        .css-1d391kg .stRadio [data-testid="stMarkdownContainer"] p { /* Sidebar radio label text */
           font-weight: 500;
        }
        .form-container {
            max-width: 550px;
            margin: 2rem auto;
            padding: 2.5rem;
            background-color: #FFFFFF;
            border-radius: 12px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        }
        .chat-input-container { /* Not directly used in current structure, but good for future */
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 1rem 2rem;
            background-color: #FFFFFF;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.08);
            z-index: 100;
        }
        .chat-messages-container {
            padding-bottom: 80px; /* Space for a fixed chat input if it were outside the main flow */
            padding-right: 1rem;
        }
        .user-bubble {
            background-color: #0066CC;
            color: white;
            padding: 10px 18px;
            border-radius: 20px 20px 5px 20px;
            max-width: 75%;
            margin-left: auto;
            margin-bottom: 12px;
            text-align: left;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            word-wrap: break-word;
        }
        .bot-bubble {
            background-color: #E9ECEF;
            color: #1E3A5F;
            padding: 10px 18px;
            border-radius: 20px 20px 20px 5px;
            max-width: 75%;
            margin-right: auto;
            margin-bottom: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            word-wrap: break-word;
        }
        .product-card {
            border: 1px solid #E0E7EF;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            background-color: #FFFFFF;
            box-shadow: 0 4px 12px rgba(0,0,0,0.06);
            transition: box-shadow 0.3s ease-in-out;
            height: 100%; /* Ensure cards in a row are same height */
        }
        .product-card:hover {
            box-shadow: 0 6px 18px rgba(0,0,0,0.1);
        }
        .product-card img {
            border-radius: 8px;
            margin-bottom: 1rem;
            width: 100%;
            height: 200px;  /* Fixed height */
            object-fit: cover;  /* Crop to fill container */
            }
        .product-title {
            font-size: 1.20em;
            font-weight: 600;
            color: #0052A3;
            margin-bottom: 0.5rem;
            height: 3em; /* Approx 2 lines */
            overflow: hidden;
            text-overflow: ellipsis;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
        }
        .product-desc {
            font-size: 0.90em;
            color: #4A5568;
            margin-bottom: 1rem;
            height: 80px; /* Fixed height for description area */
            overflow-y: auto; /* Scroll if desc is too long */
        }
        .product-link {
            font-size: 0.9em;
            color: #0066CC;
            text-decoration: none;
            font-weight: 500;
        }
        .product-link:hover {
            text-decoration: underline;
            color: #004080;
        }
        .centered-content {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }
        .warning-message {
            text-align: center;
            padding: 1.5rem;
            background-color: #FFF3CD;
            color: #856404;
            border: 1px solid #FFEEBA;
            border-radius: 8px;
            margin: 1rem auto;
            max-width: 600px;
        }
        .success-message {
            text-align: center;
            padding: 1.5rem;
            background-color: #D4EDDA;
            color: #155724;
            border: 1px solid #C3E6CB;
            border-radius: 8px;
            margin: 1rem auto;
            max-width: 600px;
        }
        .error-message {
            text-align: center;
            padding: 1rem;
            background-color: #F8D7DA;
            color: #721C24;
            border: 1px solid #F5C6CB;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        .recommendation-section-header {
            font-size: 1.4em;
            color: #1E3A5F;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            border-bottom: 2px solid #0066CC;
            padding-bottom: 0.3em;
        }
    </style>
""", unsafe_allow_html=True)

# Load environment variables
if os.path.exists("models1.env"):
    load_dotenv("models1.env")
else:
    try:
        NEO4J_URI = st.secrets["NEO4J_URI"]
        NEO4J_USER = st.secrets["NEO4J_USER"]
        NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]
        MONGO_URI = st.secrets["MONGO_URI"]
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
        GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") 
        GOOGLE_CSE_ID = st.secrets.get("GOOGLE_CSE_ID") 

        os.environ["NEO4J_URI"] = NEO4J_URI
        os.environ["NEO4J_USER"] = NEO4J_USER
        os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD
        os.environ["MONGO_URI"] = MONGO_URI
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY
        if GOOGLE_API_KEY: os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        if GOOGLE_CSE_ID: os.environ["GOOGLE_CSE_ID"] = GOOGLE_CSE_ID

    except KeyError as e:
        st.error(f"Missing secret configuration for {e}. Please ensure all secrets are set for deployment.")
        st.stop()
    except Exception as e: 
        st.error(f"Error loading secrets: {e}")
        st.stop()


# Database Connections and Model Initializations
try:
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
    )
    client = pymongo.MongoClient(os.getenv("MONGO_URI"))
    db = client["fracsnet"]
    users_col = db["users"]
    products_col = db["products"]
    interactions_col = db["interactions"]
    chat_history_col = db["chat_history"]
except Exception as e:
    st.error(f"Database connection error: {e}. Please check your connection strings and credentials in .env or secrets.")
    st.stop()

try:
    df = pd.read_csv("D:/Internship_Project/Internship_Project/Cleaned_Dataset.csv").fillna("")
    df["combined_text"] = df[["ProductName", "Nutrient_category","Description","Formulated_For","HealthConcern", "Benefits"]].astype(str).agg(" ".join, axis=1)
except FileNotFoundError:
    st.error("Error: The dataset file 'Cleaned_Dataset.csv' was not found. Please check the path.")
    st.stop()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

try:
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'} 
    )
except Exception as e:
    st.error(f"Failed to initialize HuggingFace Embeddings model: {e}")
    st.stop()

try:
    llm = ChatGroq(
        temperature=0.7, 
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-70b-8192"
    )
except Exception as e:
    st.error(f"Failed to initialize Groq LLM: {e}. Ensure GROQ_API_KEY is set in your .env file or secrets.")
    st.stop()

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="output")


google_api_key = os.getenv("GOOGLE_API_KEY")
google_cse_id = os.getenv("GOOGLE_CSE_ID")
search = None
if google_api_key and google_cse_id:
    try:
        search = GoogleSearchAPIWrapper(
            google_api_key=google_api_key,
            google_cse_id=google_cse_id
        )
    except Exception as e:
        st.warning(f"Could not initialize Google Search: {e}. Search functionality will be limited.")
else:
    st.warning("Google API Key or CSE ID not found. Web search functionality via Google will be disabled.")


def get_neo4j_recommendations(query_text, top_k=6):
    query_embedding = embedding_model.embed_query(query_text)
    keywords = set(re.findall(r'\b\w+\b', query_text.lower())) 

    with driver.session() as session:
        results = session.run("""
            CALL db.index.vector.queryNodes('product_embeddings', $top_k_fetch, $embedding)
            YIELD node, score
            RETURN node.ProductName AS name,
                   node.Description AS description,
                   node.ProductImage AS image,
                   node.PdfLink AS pdf_link,
                   node.Benefits AS benefits,
                   node.HealthConcern AS concern,
                   score
        """, top_k_fetch=top_k * 2, embedding=query_embedding) 

        filtered_recs = []
        for record in results:
            item = dict(record)
            text_blob = " ".join([
                str(item.get("description", "")),
                str(item.get("concern", "")),
                str(item.get("benefits", ""))
            ]).lower()
            
            item_keywords = set(re.findall(r'\b\w+\b', text_blob))
            
            overlap = keywords.intersection(item_keywords)
            keyword_boost = 0.05 * len(overlap) if overlap else 0 
            current_score = float(item.get("score", 0.0))
            adjusted_score = current_score + keyword_boost
            
            if len(overlap) > 1 or current_score > 0.75: 
                item["adjusted_score"] = adjusted_score
                filtered_recs.append(item)
        
        sorted_recs = sorted(filtered_recs, key=lambda x: x.get("adjusted_score", 0.0), reverse=True)
        return sorted_recs[:top_k]


def get_user_based_recommendations(username, top_k=3):
    user_interactions = list(interactions_col.find({"username": username}).sort("timestamp", -1).limit(10))
    product_names = [ui["product_name"] for ui in user_interactions if "product_name" in ui]
    if not product_names:
        return None
    
    df_user_interacted_products = df[df["ProductName"].isin(product_names)]
    if df_user_interacted_products.empty:
        return None

    combined_keywords = " ".join(df_user_interacted_products["combined_text"].tolist())
    if not combined_keywords.strip():
        return None
    return get_neo4j_recommendations(combined_keywords, top_k=top_k)


def hybrid_recommendation_tool(username, user_query): 
    user = users_col.find_one({"username": username})
    if not user: 
        st.warning("User not found for hybrid recommendation.")
        if search:
             return [{"name": "Google Search Results", "description": search.run(user_query), "is_search": True}]
        return []


    query_text = f"User profile: Age {user.get('age', 'N/A')}, Gender {user.get('gender', 'N/A')}, User Type: {user.get('usertype', 'N/A')}. User query: {user_query}. Looking for effective health supplements or natural remedies."
    
    content_based_recs = get_neo4j_recommendations(query_text, top_k=6)
    user_cf_recs = get_user_based_recommendations(username, top_k=3)
    
    hybrid_recs = []
    seen_product_names = set()

    if content_based_recs:
        for rec in content_based_recs:
            if rec and rec.get("name") and rec["name"] not in seen_product_names:
                hybrid_recs.append(rec)
                seen_product_names.add(rec["name"])

    if user_cf_recs:
        for rec in user_cf_recs:
            if rec and rec.get("name") and rec["name"] not in seen_product_names:
                rec["source"] = "user_history_similarity" 
                hybrid_recs.append(rec)
                seen_product_names.add(rec["name"])
    
    final_recs = hybrid_recs[:6] 

    if not final_recs and search: 
        search_query = f"Supplements or natural remedies for '{user_query}' considering age {user.get('age', 'N/A')} and gender {user.get('gender', 'N/A')}"
        search_results = search.run(search_query)
        return [{"name": "Google Search Results", "description": search_results, "is_search": True}]
    elif not final_recs:
        return [] 
            
    return final_recs


def get_product_image_display(product_name_or_item):
    image_data = None
    if isinstance(product_name_or_item, dict) and "image" in product_name_or_item:
        image_data = product_name_or_item["image"]
    elif isinstance(product_name_or_item, str):
        product_doc_mongo = products_col.find_one({"product_name": product_name_or_item}, {"image_base64": 1})
        if product_doc_mongo and "image_base64" in product_doc_mongo and product_doc_mongo["image_base64"]:
            image_data = product_doc_mongo["image_base64"]
        else: 
            with driver.session() as session:
                result = session.run("MATCH (p:Product {ProductName: $name}) RETURN p.ProductImage AS image", name=product_name_or_item)
                record = result.single()
                if record and record["image"]:
                    image_data = record["image"]
    
    if image_data:
        try:
            if isinstance(image_data, str) and image_data.startswith("http"): 
                response = requests.get(image_data, stream=True, timeout=10)
                response.raise_for_status() 
                return Image.open(response.raw)
            
            if not re.match(r'^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$', image_data):
                return None 
            decoded_image = base64.b64decode(image_data)
            return Image.open(BytesIO(decoded_image))
        except requests.exceptions.RequestException:
            return None
        except Exception: 
            return None
    return None


def record_interaction(username, product_name, interaction_type="view"):
    if not product_name or not isinstance(product_name, str): 
        return
    interactions_col.insert_one({
        "username": username, 
        "product_name": product_name, 
        "interaction_type": interaction_type,
        "timestamp": pd.Timestamp.now()
    })

def record_chat(username, query, response):
    if isinstance(response, list):
        loggable_response = []
        for item in response:
            if isinstance(item, dict) and "name" in item:
                loggable_response.append({"name": item["name"], "description_preview": str(item.get("description", ""))[:100]})
            else: 
                 loggable_response.append(str(item)[:200]) 
        final_response_log = str(loggable_response) 
    else:
        final_response_log = str(response)

    chat_history_col.insert_one({
        "username": username, 
        "query": str(query), 
        "response": final_response_log[:5000], 
        "timestamp": pd.Timestamp.now()
    })

def get_user_chat_history(username):
    return list(chat_history_col.find({"username": username}, {"_id": 0}).sort("timestamp", pymongo.ASCENDING))

def match_product_by_name(user_query):
    query_lower = user_query.lower().strip()
    product_names_lower = df["ProductName"].str.lower()
    
    exact_matches = df[product_names_lower == query_lower]
    if not exact_matches.empty:
        return exact_matches["ProductName"].iloc[0]

    for product_name in df["ProductName"]:
        pn_lower = product_name.lower()
        if pn_lower in query_lower:
            if len(query_lower) < len(pn_lower) + 15 or len(pn_lower) > len(query_lower) * 0.7: 
                return product_name
    return None


def recommendation_agent_tool_wrapper(input_str):
    try:
        username, query = input_str.split("|", 1)
        recommendations = hybrid_recommendation_tool(username.strip(), query.strip())
        
        if not recommendations:
            return "I couldn't find specific product recommendations in our database for that. Could you try rephrasing or providing more details?"
        
        if isinstance(recommendations[0], dict) and recommendations[0].get("is_search"):
            return recommendations[0]["description"] 
            
        formatted_recs = []
        for rec in recommendations:
            if isinstance(rec, dict) and "name" in rec:
                formatted_recs.append(f"Product: {rec['name']}. Description: {rec.get('description', 'N/A')[:150]}...")
        
        if not formatted_recs:
             return "No specific product recommendations found based on the input. You might want to try a broader search or rephrase your query."
        return "Recommended products based on your query:\n" + "\n".join(formatted_recs)
    except ValueError:
        return "Error: Input for recommendation tool is not in 'username|query' format."
    except Exception as e:
        st.error(f"Error in recommendation_agent_tool_wrapper: {e}")
        return f"An error occurred while fetching recommendations: {str(e)}"

tools = [
    Tool(
        name="HealthProductRecommender", 
        func=recommendation_agent_tool_wrapper, 
        description="Use this tool to get health supplement or natural remedy recommendations. Input should be a string in the format 'username|user_query'. For example, 'john_doe|I have trouble sleeping'."
    )
]

try:
    agent = initialize_agent(
        tools, llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        memory=memory, verbose=True, 
        handle_parsing_errors="Check your output and make sure it conforms to the format!", 
        max_iterations=5, 
        early_stopping_method="generate" 
    )
except Exception as e:
    st.error(f"Failed to initialize Langchain Agent: {e}")
    agent = None 


# --- Streamlit UI ---
INITIAL_PAGE = "🏠 Home"

if "current_page" not in st.session_state:
    st.session_state.current_page = INITIAL_PAGE
if "username" not in st.session_state: 
    st.session_state.username = None
if "user_details" not in st.session_state:
    st.session_state.user_details = {}

# This callback updates st.session_state.current_page when the radio button is changed by the user.
def update_current_page_from_radio_selection():
    # The value of the radio button (the selected page string) is in st.session_state.sidebar_radio_value_holder
    if 'sidebar_radio_value_holder' in st.session_state:
        st.session_state.current_page = st.session_state.sidebar_radio_value_holder

st.sidebar.title("Navigation")

# Dynamic page options based on login state
if st.session_state.username:
    user_display_name = st.session_state.user_details.get('name', st.session_state.username) 
    page_options_list = ["🏠 Home", f"👤 {user_display_name}", "💬 Health Assistant", "📜 Chat History", "🚪 Logout"]
else:
    page_options_list = ["🏠 Home", "🔐 Login", "✍️ Register"]

# Determine the index for the radio button based on st.session_state.current_page
try:
    if st.session_state.current_page not in page_options_list:
        st.session_state.current_page = INITIAL_PAGE 
    current_page_index = page_options_list.index(st.session_state.current_page)
except ValueError:
    st.session_state.current_page = INITIAL_PAGE 
    current_page_index = 0 

st.sidebar.radio(
    "Go to",
    options=page_options_list,
    index=current_page_index,
    key='sidebar_radio_value_holder', 
    on_change=update_current_page_from_radio_selection
)

page = st.session_state.current_page


if page == "✍️ Register":
    with st.container():
        st.markdown("<div class='form-container'>", unsafe_allow_html=True)
        st.header("Create Your HealthAgent Account")
        name = st.text_input("Full Name", key="reg_name")
        username = st.text_input("Username", key="reg_username")
        email = st.text_input("Email", key="reg_email")
        password = st.text_input("Password", type="password", key="reg_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm_password")
        user_type_options = ["Patient", "Doctor", "Health Enthusiast", "Researcher", "Other"]
        usertype = st.selectbox("I am a...", user_type_options, key="reg_usertype")
        age = st.number_input("Age", min_value=1, max_value=120, step=1, key="reg_age")
        gender_options = ["Male", "Female", "Other", "Prefer not to say"]
        gender = st.selectbox("Gender", gender_options, key="reg_gender")
        region = st.text_input("Region/Country (Optional)", key="reg_region")

        if st.button("Register Account", use_container_width=True, key="btn_register_account"):
            error = False
            if not all([name, username, email, password, confirm_password, usertype, age, gender]):
                st.markdown("<p class='error-message'>Please fill out all required fields.</p>", unsafe_allow_html=True)
                error = True
            if password != confirm_password:
                st.markdown("<p class='error-message'>Passwords do not match.</p>", unsafe_allow_html=True)
                error = True
            if users_col.find_one({"username": username}):
                st.markdown(f"<p class='error-message'>Username '{username}' already exists.</p>", unsafe_allow_html=True)
                error = True
            if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                st.markdown(f"<p class='error-message'>Please enter a valid email address.</p>", unsafe_allow_html=True)
                error = True
            elif users_col.find_one({"mail": email}): 
                st.markdown(f"<p class='error-message'>Email '{email}' is already registered.</p>", unsafe_allow_html=True)
                error = True
            
            if not error:
                users_col.insert_one({
                    "name": name, "username": username, "mail": email, 
                    "password": password, 
                    "usertype": usertype, "age": age, "gender": gender, "region": region,
                    "created_at": pd.Timestamp.now()
                })
                st.markdown("<div class='success-message'>Registration successful! You can now log in.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "🔐 Login":
    with st.container():
        st.markdown("<div class='form-container'>", unsafe_allow_html=True)
        st.header("Welcome Back!")
        st.markdown("Log in to access your personalized health assistant.")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", use_container_width=True, key="btn_login_account"):
            user = users_col.find_one({"username": login_username, "password": login_password})
            if user:
                st.session_state.username = user["username"]
                user['_id'] = str(user['_id']) 
                st.session_state.user_details = user
                
                user_display_name_for_page = user.get('name', user["username"])
                st.session_state.current_page = f"👤 {user_display_name_for_page}"
                st.rerun()
            else:
                st.markdown("<p class='error-message'>Invalid username or password.</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; margin-top:1rem;'>Don't have an account? Select 'Register' from the sidebar.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "🚪 Logout":
    user_was_logged_in = st.session_state.username is not None
    
    # Clear all session state keys to ensure a clean logout
    keys_to_delete = list(st.session_state.keys())
    for key in keys_to_delete:
        del st.session_state[key]
    
    # Re-initialize essential session state variables after clearing
    st.session_state.username = None
    st.session_state.user_details = {}
    st.session_state.messages = [] 
    if 'memory' in globals() and hasattr(memory, 'clear'):
        memory.clear() 
    st.session_state.current_page = INITIAL_PAGE 
    # sidebar_radio_value_holder will be re-initialized by the radio widget itself on rerun
    
    if user_was_logged_in:
        st.markdown("<div class='success-message' style='margin-top: 2rem;'>You have been successfully logged out.</div>", unsafe_allow_html=True)
    
    st.rerun()


elif page == "💬 Health Assistant":
    if not st.session_state.username:
        st.markdown("<div class='warning-message'>Please login first to access the Health Assistant.</div>", unsafe_allow_html=True)
        if st.button("Go to Login", key="btn_chat_goto_login"):
            st.session_state.current_page = "🔐 Login"
            st.rerun()
    else:
        st.header(f"Hello, {st.session_state.user_details.get('name', st.session_state.username)}! How can I help you today?")
        st.markdown("Ask about health concerns, symptoms, or supplements. For supplement advice, be specific about your needs.")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        st.markdown("<div class='chat-messages-container'>", unsafe_allow_html=True)
        for i, message in enumerate(st.session_state.messages): 
            bubble_class = "user-bubble" if message["role"] == "user" else "bot-bubble"
            avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
            
            if isinstance(message["content"], list) and message.get("type") == "recommendations":
                all_recs = message["content"]
                top_recs = [r for r in all_recs if isinstance(r,dict) and r.get("name") and not r.get("is_search")][:3]
                also_like_recs = [r for r in all_recs if isinstance(r,dict) and r.get("name") and not r.get("is_search")][3:6]

                if top_recs:
                    st.markdown("<h4 class='recommendation-section-header'>🏆 Top Recommendations For You</h4>", unsafe_allow_html=True)
                    cols = st.columns(len(top_recs))
                    for idx, item_data in enumerate(top_recs):
                        with cols[idx]:
                            st.markdown(f"<div class='product-card'>", unsafe_allow_html=True)
                            st.markdown(f"<p class='product-title'>{item_data['name']}</p>", unsafe_allow_html=True)
                            img = get_product_image_display(item_data) 
                            if img:
                                    st.image(img.resize((300, 200)),  # Resize to 300x200 pixels4
                                    width=300,  # Set display width
                                    use_container_width=False
                                ) 
                            st.markdown(f"<p class='product-desc'>{item_data.get('description', 'No description available.')[:200]}...</p>", unsafe_allow_html=True)
                            if item_data.get("pdf_link") and isinstance(item_data.get("pdf_link"), str) and item_data.get("pdf_link").startswith("http"):
                                st.markdown(f"<a href='{item_data['pdf_link']}' target='_blank' class='product-link'>📄 View PDF</a>", unsafe_allow_html=True)
                            
                            button_key = f"detail_top_{i}_{idx}_{item_data['name']}"
                            if st.button(f"More on {item_data['name'][:15]}...", key=button_key, use_container_width=True):
                                record_interaction(st.session_state.username, item_data["name"], "click_more_info_chat")
                                st.toast(f"Details for {item_data['name']} could be shown here or logged.")
                            st.markdown("</div>", unsafe_allow_html=True)
                
                if also_like_recs:
                    st.markdown("<h4 class='recommendation-section-header'>💡 You Might Also Like</h4>", unsafe_allow_html=True)
                    cols_also = st.columns(len(also_like_recs))
                    for idx, item_data in enumerate(also_like_recs):
                        with cols_also[idx]:
                            st.markdown(f"<div class='product-card'>", unsafe_allow_html=True)
                            st.markdown(f"<p class='product-title'>{item_data['name']}</p>", unsafe_allow_html=True)
                            img = get_product_image_display(item_data)
                            if img:
                                    st.image(img.resize((300, 200)),  # Resize to 300x200 pixels4
                                    width=300,  # Set display width
                                    use_container_width=False
                                )
                            st.markdown(f"<p class='product-desc'>{item_data.get('description', 'No description available.')[:200]}...</p>", unsafe_allow_html=True)
                            if item_data.get("pdf_link") and isinstance(item_data.get("pdf_link"), str) and item_data.get("pdf_link").startswith("http"):
                                st.markdown(f"<a href='{item_data['pdf_link']}' target='_blank' class='product-link'>📄 View PDF</a>", unsafe_allow_html=True)
                            
                            button_key_also = f"detail_also_{i}_{idx}_{item_data['name']}"
                            if st.button(f"More on {item_data['name'][:15]}...", key=button_key_also, use_container_width=True):
                                record_interaction(st.session_state.username, item_data["name"], "click_more_info_chat")
                                st.toast(f"Details for {item_data['name']} could be shown here or logged.")
                            st.markdown("</div>", unsafe_allow_html=True)
            
            elif isinstance(message["content"], str): 
                 st.markdown(f"<div class='{bubble_class}'>{avatar} {message['content']}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


        user_query = st.chat_input("Your message:", key="chat_query_input")

        if user_query:
            current_time = pd.Timestamp.now()
            st.session_state.messages.append({"role": "user", "content": user_query, "timestamp": current_time})

            with st.spinner("⚕️ HealthAgent is thinking..."):
                bot_response_content = ""
                bot_response_type = "text" 

                matched_product_name = match_product_by_name(user_query)
                if matched_product_name:
                    product_info_df = df[df["ProductName"] == matched_product_name]
                    if not product_info_df.empty:
                        product_info = product_info_df.iloc[0].to_dict()
                        bot_response_content = [{
                            "name": product_info["ProductName"],
                            "description": product_info.get("Description", "N/A"),
                            "image": product_info.get("ProductImage"), 
                            "pdf_link": product_info.get("PdfLink"),
                            "benefits": product_info.get("Benefits"),
                            "concern": product_info.get("HealthConcern")
                        }]
                        bot_response_type = "recommendations"
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": "Consult your healthcare provider before using any supplements. Based on your query, I found this specific product:", 
                            "timestamp": current_time
                        })
                        record_interaction(st.session_state.username, matched_product_name, "direct_match_chat_view")
                        record_chat(st.session_state.username, user_query, f"Direct match: {matched_product_name}")
                    else: 
                        matched_product_name = None 
                
                if not matched_product_name: 
                    history_for_prompt = st.session_state.messages[:-1] 
                    conversation_history_str = "\n".join(
                        [f"{msg['role']}: {str(msg['content'])[:200]}" for msg in history_for_prompt[-5:]] 
                    )

                    llm_input_prompt = f"""{SYSTEM_PROMPT}

Current Conversation History (last few messages):
{conversation_history_str}

User Profile:
Username: {st.session_state.username}
Age: {st.session_state.user_details.get('age', 'N/A')}
Gender: {st.session_state.user_details.get('gender', 'N/A')}
User Type: {st.session_state.user_details.get('usertype', 'N/A')}

Current User Query: "{user_query}"

Instructions for you, Health Product Recommendation Agent:
1. Analyze the "Current User Query" in the context of "Current Conversation History" and "User Profile".
2. Based on the rules in the initial SYSTEM_PROMPT (especially section 1. QUERY CLASSIFICATION and 2. RESPONSE PROTOCOLS), decide the query type.
3. Generate the appropriate response according to the protocol for that query type.
4. If the query is a "Health Product Request", you MUST state that the 'HealthProductRecommender' tool should be used by outputting EXACTLY: "Action: Use HealthProductRecommender. Input: {st.session_state.username}|[actual query for products]". For example, if the user asks "what vitamins for energy", and username is "testuser", you output: "Action: Use HealthProductRecommender. Input: testuser|vitamins for energy". Do NOT add any other text before or after this action line if this is your decision.
5. If the query is for "Health Information", provide the information directly using your knowledge, followed by the specified note.
6. If the query is "Non-Health", provide the specified rejection message, inferring the [topic].
7. If the query is "Ambiguous", ask the specified clarification question, inferring the [interpreted need].
8. Ensure all safety disclaimers and notes are included as per the SYSTEM_PROMPT for direct informational responses.

Your Response:"""
                    
                    try:
                        llm_decision_response = llm.invoke(llm_input_prompt).content.strip()
                        
                        action_prefix = f"Action: Use HealthProductRecommender. Input: {st.session_state.username}|"
                        if llm_decision_response.startswith(action_prefix):
                            extracted_query_for_tool = llm_decision_response[len(action_prefix):].strip()
                            
                            if not extracted_query_for_tool: 
                                bot_response_content = "I was about to search for products, but the specific need wasn't clear from your query. Could you please rephrase or provide more details about what you're looking for?"
                                record_chat(st.session_state.username, user_query, bot_response_content)
                            else:
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": "Consult your healthcare provider before using these supplements. I'm looking for some recommendations for you...", 
                                    "timestamp": current_time
                                })
                                raw_recommendations = hybrid_recommendation_tool(st.session_state.username, extracted_query_for_tool)

                                if raw_recommendations and isinstance(raw_recommendations[0], dict) and raw_recommendations[0].get("is_search"):
                                    search_summary = raw_recommendations[0]["description"]
                                    bot_response_content = f"I couldn't find specific products in our database for '{extracted_query_for_tool}', but here's some information from a web search: {search_summary}\n\nNeed more specifics or have other questions?"
                                    record_chat(st.session_state.username, user_query, f"Fallback search triggered by LLM: {search_summary[:200]}")
                                elif raw_recommendations:
                                    bot_response_content = raw_recommendations 
                                    bot_response_type = "recommendations"
                                    product_names_recommended = [rec.get('name', 'Unknown') for rec in raw_recommendations if isinstance(rec, dict)]
                                    for rec_item in raw_recommendations:
                                        if isinstance(rec_item, dict) and "name" in rec_item:
                                            record_interaction(st.session_state.username, rec_item["name"], "llm_recommendation_view")
                                    record_chat(st.session_state.username, user_query, f"LLM triggered recommendations: {', '.join(product_names_recommended)}")
                                    st.session_state.messages.append({
                                        "role": "assistant", 
                                        "content": bot_response_content, 
                                        "type": "recommendations", 
                                        "timestamp": current_time
                                    })
                                    st.session_state.messages.append({ 
                                        "role": "assistant",
                                        "content": "Need more specifics or have other questions about these products?",
                                        "timestamp": current_time
                                    })
                                    bot_response_content = None 
                                else: 
                                    bot_response_content = f"I looked for products related to '{extracted_query_for_tool}' but couldn't find specific recommendations in our database at the moment. You could try rephrasing or asking for general information. Need more specifics or have other questions?"
                                    record_chat(st.session_state.username, user_query, f"LLM triggered no specific recommendations for: {extracted_query_for_tool}")
                        
                        else: 
                            bot_response_content = llm_decision_response
                            record_chat(st.session_state.username, user_query, bot_response_content)

                    except Exception as e:
                        st.error(f"Error processing your query with LLM: {e}")
                        bot_response_content = f"Sorry, I encountered an error trying to understand your request: {str(e)[:100]}"
                        record_chat(st.session_state.username, user_query, f"LLM Error: {e}")

                if bot_response_content is not None: 
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": bot_response_content, 
                        "type": bot_response_type, 
                        "timestamp": current_time
                    })
            st.rerun()


elif page == "📜 Chat History":
    if not st.session_state.username:
        st.markdown("<div class='warning-message'>Please login first to view your chat history.</div>", unsafe_allow_html=True)
        if st.button("Go to Login", key="btn_history_goto_login"):
            st.session_state.current_page = "🔐 Login"
            st.rerun()
    else:
        st.header(f"📜 Chat History for {st.session_state.user_details.get('name', st.session_state.username)}")
        st.markdown("Review your past conversations and recommendations.")
        chats = get_user_chat_history(st.session_state.username)
        if not chats:
            st.info("You have no chat history yet. Start a conversation with the Health Assistant!")
        else:
            for chat_entry in reversed(chats): 
                st.markdown("---")
                query_time = chat_entry.get('timestamp', 'N/A')
                try:
                    query_time_str = pd.to_datetime(query_time).strftime('%Y-%m-%d %H:%M:%S')
                except: 
                    query_time_str = str(query_time)

                st.markdown(f"<p style='font-size:0.9em; color:grey;'>🗓️ {query_time_str}</p>", unsafe_allow_html=True)
                
                user_query_display = chat_entry.get('query', '[No query recorded]')
                st.markdown(f"<div class='user-bubble' style='margin-left:0; margin-right:auto; max-width:90%; background-color:#D1E7FD; color:#0A58CA;'>🧑‍💻 **You:** {user_query_display}</div>", unsafe_allow_html=True)
                
                bot_response_data = chat_entry.get('response', '[No response recorded]')
                display_response = str(bot_response_data)
                if len(display_response) > 700: 
                    display_response = display_response[:700] + "..."
                st.markdown(f"<div class='bot-bubble' style='max-width:90%;'>🤖 **Bot:** {display_response}</div>", unsafe_allow_html=True)
            st.markdown("---")


elif page == "🏠 Home" or (page.startswith("👤") and st.session_state.username): 
    st.title("Welcome to HealthAgent!")
    st.markdown("")
    
    if st.session_state.username and st.session_state.user_details:
        user_data = st.session_state.user_details 
        st.subheader(f"Dashboard for {user_data.get('name', st.session_state.username)}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"""
                **Name:** {user_data.get('name', 'N/A')}  
                **Username:** {user_data.get('username', 'N/A')}  
                **User Type:** {user_data.get('usertype', 'N/A')}  
                **Age:** {user_data.get('age', 'N/A')}  
                **Gender:** {user_data.get('gender', 'N/A')}
            """)
        with col2:
            st.success("Quick Actions:")
            if st.button("💬 Go to Health Assistant", use_container_width=True, key="btn_dash_goto_chat"):
                st.session_state.current_page = "💬 Health Assistant"
                st.rerun()
            if st.button("📜 View Chat History", use_container_width=True, key="btn_dash_goto_history"):
                st.session_state.current_page = "📜 Chat History"
                st.rerun()
        
        st.markdown("---")
        st.subheader("Recently Viewed/Interacted Items")
        recent_interactions_pipeline = [
            {"$match": {"username": st.session_state.username, "product_name": {"$ne": None}}},
            {"$sort": {"timestamp": -1}},
            {"$group": {"_id": "$product_name", "last_interacted": {"$first": "$timestamp"}}},
            {"$sort": {"last_interacted": -1}},
            {"$limit": 3},
            {"$project": {"product_name": "$_id", "_id": 0}}
        ]
        recent_product_names_docs = list(interactions_col.aggregate(recent_interactions_pipeline))
        recent_product_names = [doc['product_name'] for doc in recent_product_names_docs]

        if recent_product_names:
            cols = st.columns(len(recent_product_names) if recent_product_names else 1)
            for i, name in enumerate(recent_product_names):
                with cols[i]:
                    product_info_df = df[df["ProductName"] == name]
                    if not product_info_df.empty:
                        item_data = product_info_df.iloc[0].to_dict()
                        st.markdown(f"<div class='product-card'>", unsafe_allow_html=True)
                        st.markdown(f"<p class='product-title'>{item_data['ProductName']}</p>", unsafe_allow_html=True)
                        img = get_product_image_display(item_data.get('ProductImage') or item_data['ProductName']) 
                        if img:
                                st.image(img.resize((300, 200)),  # Resize to 300x200 pixels4
                                width=300,  # Set display width
                                use_container_width=False
                                )
                        st.markdown(f"<p class='product-desc'>{item_data.get('Description', 'N/A')[:100]}...</p>", unsafe_allow_html=True)
                        if item_data.get("PdfLink") and isinstance(item_data.get("PdfLink"), str) and item_data.get("PdfLink").startswith("http"):
                           st.markdown(f"<a href='{item_data['PdfLink']}' target='_blank' class='product-link'>📄 View PDF</a>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        with colas[i]: 
                            st.info(f"Details for '{name}' could not be fully loaded from dataset.")
        else:
            st.info("No recent product interactions found. Explore products via the Health Assistant!")
    else: 
        st.markdown("""
            
            
            **Please Login or Register to get started!**
        """)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔐 Login", use_container_width=True, key="btn_home_login"):
                st.session_state.current_page = "🔐 Login"
                st.rerun()
        with col2:
            if st.button("✍️ Register", use_container_width=True, key="btn_home_register"):
                st.session_state.current_page = "✍️ Register"
                st.rerun()

        st.markdown("---")
        

# Removed the problematic final block that tried to set sidebar_radio_value_holder directly.
# The on_change callback and index setting for the radio button now handle synchronization.
