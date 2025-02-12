from openai import OpenAI
import anthropic
from langchain_community.llms import Ollama
import abc
import json


class LLM(abc.ABC):
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def generate(self, prompt, max_tokens=4000):
        pass

    def parse_json(self, response: str) -> dict:
        json_start, json_end = response.find('{'), response.rfind('}') + 1

        try:
            out = response[json_start:json_end]
            out = json.loads(out)
        except Exception as e:
            print(e)
            print(response)
            return None
        
        return out
        


class GPTChat(LLM):
    def __init__(self, model_name='gpt-3.5-turbo-0125', api_key=None):
        super().__init__()
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name

    
    def generate(self, prompt, max_tokens=4000, temperature=1, system_message="You are a helpful assistant.", json=True):
        kwargs = {}
        if json:
            kwargs["response_format"]={"type": "json_object"}
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            print(e)
            return None
        return response.choices[0].message.content

class NvidiaChat(GPTChat):
    def __init__(self, model_name='gpt-4-turbo', api_key=None):
        super().__init__(model_name=model_name, api_key=api_key)
        self.client = OpenAI(api_key=self.api_key, base_url="https://integrate.api.nvidia.com/v1")
    
    def generate(self, prompt, max_tokens=4000, temperature=1, system_message="You are a helpful assistant.", json=True):
        kwargs = {}
        if json:
            kwargs["response_format"]={"type": "json_object"}
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            print(response)
        except Exception as e:
            print(e)
            return None

        return response

class OllamaClass(LLM):
    def __init__(self, model_name='llama3'):
        super().__init__()
        self.client = Ollama(model=model_name)
        self.model_name = model_name

    def generate(self, prompt, max_tokens=4000, temperature=1, system_message="You are a helpful assistant.", json=True):
        kwargs = {}
        if json:
            kwargs["response_format"]={"format": "json"}
        try:
            response = self.client.invoke(prompt, **kwargs)
        except Exception as e:
            print(e)
            return None
        return response

class ClaudeChat(LLM):
    def __init__(self, model_name='claude-3-5-sonnet-20241022', api_key=None):
        super().__init__()
        self.model_name = model_name
        self.api_key = api_key
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt, max_tokens=4000, temperature=1, system_message="You are a helpful assistant.", json=True):
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_message,
                messages=[
                    {"role": "user", "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
        except Exception as e:
            print(e)
            return None
        return response.content[0].text

class OpenRouterChat(GPTChat):
    """LLM class for accessing OpenRouter's GPT API via a custom endpoint."""
    
    def __init__(self, model_name='deepseek/deepseek-chat', api_key=None, base_url="https://openrouter.ai/api/v1"):
        """
        Initialize the OpenRouterChat instance.
        
        Args:
            model_name (str): The model name to use.
            api_key (str, optional): The API key for OpenRouter. If not provided, you may set it via the OPENROUTER_API_KEY environment variable.
            base_url (str, optional): The base URL for OpenRouter API.
        """
        super().__init__(model_name=model_name, api_key=api_key)
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

