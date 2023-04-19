import openai
import os

# Set your OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Define the prompt
prompt = """
You have the following information:
Transcript: This global program is designed to equip mid-to-senior level finance executives with insights and practical skills to harness AI and machine learning.
Based on this, decide whether to reply to the following comment:
Comment: I would like to attend, I am working in the banking sector as a CFO, is this for me?
Please provide a decision on whether to reply to the comment and, if so, a detailed response.
Consider replying to positive comments even if they are not directly related to the content of the video.
Your output should start with "Decision:" followed by either "Reply" or "No Reply".
If you decide to reply, include the response on a new line starting with "Response (if replying): Your response here".
"""

# Call the OpenAI API
try:
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
    )
    # Print the response
    print(response["choices"][0]["text"])
except Exception as e:
    print(f"Error: {e}")
