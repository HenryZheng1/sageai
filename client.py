from langchain_openai import ChatOpenAI
from openai import AzureOpenAI
from pinecone import Pinecone
from transformers import pipeline
from openai import OpenAI

class ClientHelper:
    pass
class AzureClient(AzureOpenAI,ClientHelper):
    def __init__(self, endpoint, api_key, api_version):
        AzureOpenAI.__init__(self, azure_endpoint=endpoint,
                             api_key=api_key, api_version=api_version)


class LangChainClient(ChatOpenAI, ClientHelper):
    def __init__(self, api_key):
        ChatOpenAI.__init__(self, api_key=api_key, model="gpt-4o-mini")

    def generate_response(self, messages):
        msg = self.invoke(messages)
        return msg.content



class HuggingFaceClient(
    ClientHelper
):
    def __init__(self, model_name, **kwargs):
        self.generator = pipeline(
            "text-generation", model=model_name, **kwargs)

    def generate_text(self, prompt, max_length=200, **kwargs):
        return self.generator(prompt, max_length=max_length, truncation=True, **kwargs)[0]["generated_text"]


class PerplexityClient(OpenAI, ClientHelper):
    def __init__(self, api_key):
        OpenAI.__init__(self, api_key=api_key,
                        base_url="https://api.perplexity.ai")

    def generate_response(self, prompt):
        # from pprint import pprint
        try:
            response = self.chat.completions.create(
                model="llama-3.1-sonar-large-128k-online",
                # model = 'llama-3.1-sonar-small-128k-online',
                messages=prompt
            )
        except Exception as e:
            return f"Error from GPT-4o: {e}"
        # if ratelimit error, wait 30 seconds and try again
        # pprint(vars(response))
        return response.choices[0].message.content.strip()


class PineconeClient(Pinecone):
    def __init__(self, api_key, index_name=None) -> None:
        super().__init__(api_key=api_key)
        if index_name:
            self.index = self.Index(index_name)
    def clear(self):
        response = self.index.delete(delete_all=True)
        print("Delete response:", response)
        print("All vectors have been removed from the index.")