-Install the required packages using the following command:

pip install -r requirements.txt

-Set up environment variables

GOOGLE_APPLICATION_CREDENTIALS_CONTENT=your_google_application_credentials_content
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
SERPAPI_API_KEY=your_serpapi_api_key
YOUTUBE_API_KEY=your_youtube_api_key

Replace your_* with your actual API keys and credentials.

-Run the application

Now that you have set up the environment variables, you can run the application:
python main.py