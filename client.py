from openai import AzureOpenAI
from pinecone import Pinecone

class ClientHelper:
    pass
class AzureClient(AzureOpenAI,ClientHelper):
    def __init__(self, endpoint, api_key, api_version):
        AzureOpenAI.__init__(self, azure_endpoint=endpoint,
                             api_key=api_key, api_version=api_version)
class PineconeClient(Pinecone):
    def __init__(self, api_key, index_name=None) -> None:
        super().__init__(api_key=api_key)
        if index_name:
            self.index = self.Index(index_name)
    def clear(self):
        response = self.index.delete(delete_all=True)
        print("Delete response:", response)
        print("All vectors have been removed from the index.")