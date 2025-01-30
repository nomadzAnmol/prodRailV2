import os
import json
import uvicorn
import re
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from playwright.sync_api import sync_playwright
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables (for local dev)
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Allow specific frontend origins (or use "*" to allow all)
origins = [
    "http://127.0.0.1:5500",  # Your local frontend
    "http://localhost:5500",  # In case you use localhost
    "https://nomadictrailz.com",  # Add your deployed frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow only these origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)
# Set API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("Missing required environment variables")


# Initialize OpenAI and Pinecone clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Pinecone index configuration
INDEX_NAME = "testnomadz"
NAMESPACE = "global"

index = pc.Index(INDEX_NAME)

# Define keywords and exclusion lists
keywords = ['off', 'offers', 'coupons', 'flat', 'discount']
exclude_phrases = ['check the landing page', 'expired coupons', 'login', 'signup']
redundant_phrases = [
    'verified', 'activate offer', 'get deal', 'click here', 'see details', 'explore now'
]

# Request model
class QueryRequest(BaseModel):
    query: str

def clean_offer_text(text):
    """Cleans redundant phrases from offer text."""
    for phrase in redundant_phrases:
        text = re.sub(re.escape(phrase), '', text, flags=re.IGNORECASE)
    cleaned_text = ' '.join(text.split()).strip()
    return ' '.join(cleaned_text.split()[:5])

def scrape_offers(url):
    """Scrapes offer details from the given URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        relevant_sections = soup.find_all(['p', 'div'], class_=True)
        offers = []

        for section in relevant_sections:
            text_content = section.get_text().lower().strip()
            if any(phrase in text_content for phrase in exclude_phrases):
                continue
            if any(keyword in text_content for keyword in keywords):
                if re.search(r'\b\d+%|\b₹\d+', text_content):
                    offers.append(section.text.strip()[:150])

        if offers:
            middle_index = len(offers) // 2
            return clean_offer_text(offers[middle_index])
        else:
            return 'No relevant offers found.'
    except requests.exceptions.RequestException:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                page.goto(url)
                content = page.content()
                browser.close()
                soup = BeautifulSoup(content, 'html.parser')
                relevant_sections = soup.find_all(['p', 'div'], class_=True)
                offers = []

                for section in relevant_sections:
                    text_content = section.get_text().lower().strip()
                    if any(phrase in text_content for phrase in exclude_phrases):
                        continue
                    if any(keyword in text_content for keyword in keywords):
                        if re.search(r'\b\d+%|\b₹\d+', text_content):
                            offers.append(section.text.strip()[:150])

                if offers:
                    middle_index = len(offers) // 2
                    return clean_offer_text(offers[middle_index])
                else:
                    return 'No relevant offers found.'
        except Exception as e:
            return f"Error retrieving offers: {e}"

def generate_query_embedding(query_text):
    """Generates an embedding for the user query using OpenAI."""
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=query_text
        )
        return response.data[0].embedding
    except Exception as e:
        return None

def query_pinecone(query_vector, top_k=5):
    """Queries Pinecone with the generated query embedding."""
    try:
        return index.query(vector=query_vector, top_k=top_k, include_metadata=True, namespace=NAMESPACE)
    except Exception as e:
        return None

def format_response(query_response):
    """Formats Pinecone query results."""
    if query_response and "matches" in query_response:
        return [{
            "score": match['score'],
            "source_url": match['metadata'].get('source_url', 'N/A'),
            "website_name": match['metadata'].get('website_name', 'N/A'),
            "text_snippet": match['metadata'].get('text', 'No snippet available')
        } for match in query_response['matches']]
    return None

def generate_chatgpt_response(messages):
    """Generates a response using ChatGPT."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Sorry, I couldn't generate a response."

# FastAPI route for querying deals
@app.post("/get-deals")
async def get_deals(request: QueryRequest):
    try:
        user_query = request.query
        if not user_query:
            raise HTTPException(status_code=400, detail="Query is required.")

        query_vector = generate_query_embedding(user_query)
        if not query_vector:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding.")

        query_response = query_pinecone(query_vector)
        if not query_response or "matches" not in query_response:
            return {"response": "No relevant results found."}

        results = format_response(query_response)
        if not results:
            return {"response": "No relevant results found."}

        source_urls = list({match['source_url'] for match in results if match['source_url'] != 'N/A'})
        scraped_offers = {url: scrape_offers(url) for url in source_urls}

        messages = [{"role": "system", "content": "Provide concise answers based on provided information."}] + [
            {"role": "user", "content": f"Source: {match['source_url']}\nText: {match['text_snippet']}"}
            for match in results
        ]
        chatgpt_response = generate_chatgpt_response(messages)

        return {
            "response": chatgpt_response,
            "sources": source_urls,
            "offers": scraped_offers
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI route for health check
@app.get("/")
def health_check():
    return {"status": "API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)