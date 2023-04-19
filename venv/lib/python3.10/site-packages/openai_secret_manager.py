import openai_secret_manager
import openai


def get_secret(key_name):
    return openai_secret_manager.get_secret(key_name)


def set_secret(key_name, secret):
    openai_secret_manager.set_secret(key_name, secret)

def get_openai_api_key():
    """
    A helper function to retrieve the OpenAI API key from the secrets vault.
    Returns:
        str: The OpenAI API key.
    """
    secrets = openai_secret_manager.get_secret("openai")
    return secrets["api_key"]

def set_openai_api_key(api_key):
    """
    A helper function to set the OpenAI API key which can be used for making requests.
    Args:
        api_key (str): The OpenAI API key to set.
    """
    openai.api_key = api_key