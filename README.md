# 💊HealthAgent:An AI powered Framework for Personalized Health Supplement Recommendations Using Langchain and Neo4j 💊
Your friendly AI assistant for health product recommendations and information!
---
## 👋 Overview

Hello there! HealthAgent is a smart web app built to help you find health supplements and get easy-to-understand health info. Think of it as your personal guide in the world of health products! 🏥💡

You can chat with it, ask about things like "vitamins for energy" or "what helps with sleep?", and it will try its best to give you useful suggestions or explanations.

![image](https://github.com/user-attachments/assets/b4c2625f-526a-4016-90bc-cf2b99e26a4d)

---

## ✨ Key Features

* **Smart Chatbot 💬:** Asks you questions to understand what you need.
* **Product Recommendations 🛍️:** Suggests health products based on your queries.
* **Health Info Guru 🧠:** Explains health topics in simple terms.
* **Personalized for You 👤:** Uses your profile (like age, user type) to give better suggestions (if you're logged in!).
* **Safe & Sound 🙏:** Reminds you that it's not a doctor and you should always consult one for serious health advice.
* **User Accounts 🔒:** Sign up and log in to get a more tailored experience and see your chat history.
* **See Your Past Chats 📜:** Easily look back at your previous conversations.
* **Quick Product Look-up 🔍:** If you type a product name, it tries to show you info about that specific product.

---

## 🛠️ Tech Stack

We've used some cool technologies to build HealthAgent:

* **🐍 Python:** The main programming language.
* **🎈 Streamlit:** To create the easy-to-use web interface.
* **🧠 Langchain:** For making our AI agent smart and connecting it to tools.
* **🤖 Groq & Llama3:** The powerful Large Language Model that helps the agent think and respond.
* **🕸️ Neo4j:** A graph database to store product info and find connections between them quickly (like finding similar products).
* **📄 MongoDB:** A database to keep user details, chat history, and what products you've looked at.
* **🤗 HuggingFace Transformers:** For understanding the meaning behind your text queries (embeddings).
* **🐼 Pandas:** For handling our product data.
---

## 🏗️ How HealthAgent Works (System Architecture)

Think of HealthAgent like a team working together:

1.  **You (The User) & The App (Streamlit) 💻:** You type your question into the chat.
2.  **The Brain (LLM Agent - Langchain & Groq) 🤔:**
    * It reads your question and your user profile (if logged in).
    * It decides: "Is this a product question, an info question, or something else?"
3.  **If you ask for Products... 🧴**
    * The Brain tells the **Recommender Tool** to find some ideas.
    * The Recommender Tool looks in two places:
        * **Neo4j Database (Smart Search):** It uses "vector search" (like a super-smart keyword search) to find products that match what you're looking for.
        * **Your Past Activity (MongoDB):** If you've used the app before, it might suggest things based on what you liked previously.
    * It combines these ideas and shows you the best ones.
    * If it can't find anything, it might quickly search Google for you!
4.  **If you ask for Health Info... 📖**
    * The Brain uses its own knowledge (from the Llama3 model) to explain it simply.
5.  **Data Storage 🗄️:**
    * **MongoDB:** Keeps your login info, chat messages, and which products you interact with.
    * **Neo4j:** Stores all the product details and their special "embeddings" (text fingerprints) for fast searching.
    * **CSV File:** The main `Cleaned_Dataset.csv` file holds the initial product information.

![image](https://github.com/user-attachments/assets/4bfab6b9-07ed-42c9-8bba-82d2bdf1163b)
