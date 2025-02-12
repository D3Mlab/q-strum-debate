import abc
import llm
import logging

class LLMComparer(abc.ABC):
    def __init__(self, logger: logging.Logger, llm: llm.LLM, dataset_name: str = None):
        self.logger = logger
        self.llm = llm
        self.dataset_name = dataset_name
        pass

    @abc.abstractmethod
    def compare(self, query: str, dest1: str, sentences1: list[str], dest2: str, sentences2: list[str], intermediate_results: dict = {}) -> tuple[dict, dict]:
        pass