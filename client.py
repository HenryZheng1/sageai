from openai import AzureOpenAI


def initialize_azure_client(azure_endpoint, api_key, azure_api_version):
    """
    Create and return an AzureOpenAI client object.
    """
    return AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=azure_api_version
#    


class ClientHelper:
    pass


class AzureClient(AzureOpenAI,ClientHelper):
    def __init__(self, endpoint, api_key, api_version):
        AzureOpenAI.__init__(self,azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
        # self.client = initialize_azure_client(
        #     azure_endpoint=azure_endpoint,
        #     api_key=api_key,
        #     azure_api_version=azure_api_version
        # )
from pinecone import Pinecone
class PineconeClient(Pinecone):
    def __init__(self, api_key, index_name=None) -> None:
        super().__init__(api_key=api_key)
        if index_name:
            self.index = self.Index(index_name)