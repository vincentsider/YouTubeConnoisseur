# Import the necessary libraries
import logging
from urllib.parse import urlparse, parse_qs
from flask import Flask, render_template, request, redirect, url_for, jsonify
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import openai
import os
import re
import pinecone
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi
import requests
from googleapiclient.discovery import build
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
import hashlib
from google.cloud import translate_v2 as translate
import json
from google.oauth2 import service_account
from langchain import OpenAI, LLMChain
from typing import List, Dict, Any
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.agents import Tool, LLMSingleActionAgent, AgentType, initialize_agent, AgentExecutor, BaseSingleActionAgent
from langchain.memory import ConversationBufferMemory
from langchain.utilities import SerpAPIWrapper
from langchain.agents.agent import Agent, AgentOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    BaseChatPromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.llms import OpenAI
from transformers import GPT2Tokenizer

#client_secret_file = 'client_secret_1911968826-ch6c77gbacv8jc2bmie7b7qa9529o8ld.apps.googleusercontent.com.json'
#scopes = ['https://www.googleapis.com/auth/youtube.force-ssl']
# Start the OAuth 2.0 flow
#flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, scopes)
#credentials = flow.run_local_server(port=0)
# Get the access token
#access_token = credentials.token
#print('Access Token:', access_token)

# Initialize logging
#logging.basicConfig(level=logging.DEBUG)

# Retrieve API keys from the environment variables
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")

# Load the JSON content from the Replit secret
service_account_info = json.loads(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_CONTENT"))

# Create credentials from the service account info
credentials = service_account.Credentials.from_service_account_info(service_account_info)

# Create a Cloud Translation API client
translate_client = translate.Client(credentials=credentials)

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, deployment_mode=PINECONE_ENVIRONMENT)
index_name = "video-transcripts"

# Create a Pinecone index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536, metric="cosine", shards=1)

# Create Pinecone index object
pinecone_index = pinecone.Index(index_name=index_name)
  
# Initialize OpenAI
openai.api_key = OPENAI_API_KEY



# Function to get video details from the YouTube API using the video ID
def get_video_details(video_id):
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet",
        "id": video_id,
        "key": os.environ.get("YOUTUBE_API_KEY")
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"An error occurred: {response.text}")
        return None

    video_data = response.json()
    video_title = video_data['items'][0]['snippet']['title']
    return video_title

def extract_video_id(url):
    parsed_url = urlparse(url)
    
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    elif parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            return parse_qs(parsed_url.query)['v'][0]
        elif parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/')[2]
        elif parsed_url.path.startswith('/v/'):
            return parsed_url.path.split('/')[2]
    else:
        raise ValueError("Invalid YouTube URL")

# Function to retrieve the video transcript
def get_video_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = ' '.join([t['text'] for t in transcript])
        return transcript_text
    except Exception as e:
        print(f"Error retrieving transcript: {e}")
        return None
      
# Define a function to remove emojis
def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


def translate_text(text):
    if not text or text.strip() == '':
        return ''

    # Detect the source language of the text
    try:
        # Use the Cloud Translation API to detect the language
        result = translate_client.detect_language(text)
        source_language = result['language']
    except Exception as e:
        print(f"Error detecting language: {e}")
        return ''

    # If the source language is already English, return the original text
    if source_language == 'en':
        return text

    # Translate the text to English using the Cloud Translation API
    try:
        result = translate_client.translate(text, target_language='en')
        # Extract the translated text from the result
        translated_text = result['translatedText']
        return translated_text
    except Exception as e:
        print(f"Error translating text: {e}")
        return ''

# Define a function to get video comments from a YouTube video ID
def get_video_comments(video_id, max_results=3, after=None):
    # Define the URL and query parameters for the YouTube API request
    url = f"https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": max_results,
        "textFormat": "plainText",
        "key": YOUTUBE_API_KEY,
        "publishedAfter": after,
    }

    # Send a GET request to the YouTube API and store the response
    response = requests.get(url, params=params)

    # Check if the response is not successful and return None
    if response.status_code != 200:
        print(f"An error occurred: {response.text}")
        return None

    # Parse the response JSON data
    data = response.json()

    # Initialize an empty list to store comments
    comments = []
    nextPage_token = None
    while True:
        # Iterate through the items in the response data
        for item in data["items"]:
            # Extract the comment ID, original comment, and clean the comment by removing emojis
            comment_id = item["id"]
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            #print(f"Processing comment: '{comment}'")
            comment_clean = remove_emojis(comment)

            # Translate non-English comments
            translated_comment = translate_text(comment_clean)
            #print("Translated comment:", translated_comment)

            # Store the comment, translated comment, and sentiment analysis results in the comments list
            comments.append({
                'id': comment_id,
                'original': comment,
                'translated': translated_comment,
            })
            #Get the nextPageToken for the next set of comments, if available
        nextPage_token = data.get('nextPageToken')
        if not nextPage_token:
            break
        params['pageToken'] = nextPage_token
        response = requests.get(url, params=params)
        data = response.json()

    return comments
  
# Define a function to reply to a comment on a YouTube video
def reply_comment(video_id, parent_id, response, api_key):
    # Define the URL and headers for the YouTube API request
    url = "https://www.googleapis.com/youtube/v3/comments"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    # Define the data payload for the request
    data = {
        "snippet": {
            "parentId": parent_id,
            "textOriginal": response
        }
    }
    # Send a POST request to the YouTube API to reply to the comment
    response = requests.post(url, headers=headers, json=data)
    # Check if the response is successful and return the JSON data, otherwise print an error message
    if response.status_code == 200:
        print(f"Replied to comment: {parent_id}")
        return {"status": "success", "data": response.json()}
    else:
        print(f"Not inserted in YouTube Yet: {response.status_code}<br>")
        return {"status": "error", "message": f"Not inserted in YouTube Yet: {response.status_code}"}
      
# Define a function to generate embeddings using a language model (e.g., GPT-3.5)
def generate_embedding(text, model="text-embedding-ada-002"):
    try:
        # Replace newlines with spaces in the input text
        text = text.replace("\n", " ")
        # Set the OpenAI API key
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        # Send a request to OpenAI API to create embeddings using the specified model
        response = openai.Embedding.create(
            input=[text],
            model=model,
        )
        # Extract the embedding from the API response
        embedding = response['data'][0]['embedding']
        return embedding
    except Exception as e:
        # Print an error message if there is an issue generating the embedding
        print(f"Error generating embedding: {e}")
        return None

# Define which tools the agent can use to answer user queries
search = SerpAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    )
]

# Define a custom prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    template: str
    input_variables: List[str] = ["input", "transcript"]

    def format_messages(self, **kwargs) -> str:
        # Format the prompt template with the provided input variables
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]

# Define the custom prompt to be used in the agent
prompt = CustomPromptTemplate(template="""
You have the following information:
Transcript: {transcript}
Based on this, decide whether to reply to the following comment:
Comment: {input}
Please provide a decision on whether to reply to the comment and, if so, a detailed response.
Consider replying only to meaningful comments that are related to the content of the video, or to particularly positive comments.
Your output should start with "Decision:" followed by either "Reply - this is my reasoning to reply: (insert your reasoning)" or "No Reply- this is my reasoning not to reply: (insert your reasoning)".
If you decide to reply, include the response on a new line starting with "Response (if replying): Your response here".
If you decide not to reply, you may leave the response blank.
""")

# Define a custom output parser
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Dict[str, Any]:
        # Extract the decision and response from the LLM output
        decision_match = re.search(r'Decision: (.+)', llm_output)
        response_match = re.search(r'Response \(if replying\):([\s\S]*)', llm_output)
        decision = decision_match.group(1).strip() if decision_match else None
        response = response_match.group(1).strip() if response_match else None
        return {'decision': decision, 'response': response}

# Instantiate the custom output parser
output_parser = CustomOutputParser()

# Create an instance of the language model (GPT-4)
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7, max_tokens=3000)

# Create an instance of the Agent class
agent = LLMSingleActionAgent(
    llm_chain=LLMChain(llm=llm, prompt=prompt),
    output_parser=output_parser,
    stop=["nostop"]  # Add the stop parameter
)


app = Flask(__name__)

# Define the route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for processing video comments
@app.route("/process", methods=["POST"])
def process():
    #print("Entering the process function")  # Debugging print statement
    video_url = request.form.get('video_url')

    # Check if the video_url is provided
    if not video_url:
        return "Bad Request: Missing required fields.", 400

    # Extract the video ID from the video URL
    video_id = extract_video_id(video_url)

    # Check if the video ID is valid
    if not video_id:
        return "Bad Request: Invalid video URL.", 400

    # Retrieve the video comments
    video_comments = get_video_comments(video_id)

    # Check if the video comments were retrieved successfully
    if not video_comments:
        return "Error retrieving video comments.", 400

    # Retrieve video details and transcript
    video_title = get_video_details(video_id)
    video_transcript = get_video_transcript(video_id)

    # Check if the video details and transcript were retrieved successfully
    if not video_title or not video_transcript:
        return "Error retrieving video details or transcript.", 400
    
    # Create an empty list to store the results
    results = [] 
  
    # Process one comment at a time
    for index, comment in enumerate(video_comments):
      # Add a break statement to stop processing after 10 comments
        if index >= 3:
          break
        comment_text = comment['translated']
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Truncate the transcript to a certain number of tokens, e.g., 3000
        transcript_tokens = tokenizer.tokenize(video_transcript)
        truncated_transcript = tokenizer.convert_tokens_to_string(transcript_tokens[:3000])
        # Use the truncated_transcript in the agent_input
        agent_input = {"input": comment_text, "transcript": truncated_transcript}
        intermediate_steps = []  # Define intermediate_steps as an empty list
        agent_output = agent.plan(intermediate_steps, **agent_input)
        decision = agent_output['decision']
        response = agent_output['response']

        # Print the decision and response
        print(f"Decision: {decision}")
        print(f"Response: {response}")

        # If the decision is to reply, post the reply as a comment on the video
        if decision == "Reply" and response:
            
            reply_status = reply_comment(video_id, comment['id'], response, YOUTUBE_API_KEY)
            if reply_status['status'] == 'error':
                results.append(reply_status['message'])
                   
        # Append the comment, decision, and response to the results list (outside the if block)
        results.append(
    {
        'comment': comment['translated'],
        'decision': decision,
        'response': response if response else None
    }
)
        #break
    # Generate embedding and upsert to Pinecone
    embedding = generate_embedding(video_transcript)
    if embedding is not None:
        video_id_hash_str = hashlib.sha256(video_id.encode()).hexdigest()
        upsert_dict = {video_id_hash_str: embedding}
        embedding_array = np.array(embedding)
        ids_batch = [video_id_hash_str]
        embeds = [embedding_array.astype(float).tolist()]
        to_upsert = zip(ids_batch, embeds)
        pinecone_index.upsert(vectors=list(to_upsert))  

    # Return the results as an HTML formatted string and a return statement to handle cases where no response is generated
    return jsonify({"results": results if results else "No response generated for any comment."})

# Start the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)



