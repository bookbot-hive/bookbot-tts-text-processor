from azure.cosmos import CosmosClient
import logging


logger = logging.getLogger("cosmos")
logger.setLevel(logging.INFO)

logging.getLogger("azure.core").setLevel(logging.ERROR)


class Cosmos:
    def __init__(self, url: str, key: str, database_name: str, language: str = "en"):
        self.client = self._get_cosmos_client(url, key)
        self.database_name = database_name
        self.word_universal_container = self._get_container_client(self.client, "WordUniversal")
        self.language = language
        if not self.language:
            logger.warning("Language not set yet, using default language")

    @staticmethod
    def _get_cosmos_client(url: str, key: str) -> CosmosClient:
        return CosmosClient(url, credential=key)

    def _get_container_client(self, client: CosmosClient, container_name: str):
        database = client.get_database_client(self.database_name)
        return database.get_container_client(container_name)

    def get_query(self):
        return f"SELECT * FROM c WHERE c.language = '{self.language}' AND NOT IS_DEFINED(c.deletedAt)"

    def get_all_records(self):
        query = self.get_query()
        query_iterable = self.word_universal_container.query_items(
            query=query, partition_key="default", max_item_count=10000
        )
        pager = query_iterable.by_page()
        existing_records = []
        while True:
            page = list(pager.next())
            existing_records += page
            continuation_token = pager.continuation_token

            if not continuation_token:
                break
            pager = query_iterable.by_page(continuation_token)
        return existing_records
