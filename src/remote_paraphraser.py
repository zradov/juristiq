import logging
from llm_utils import (
    get_genai_client
)
from collections import defaultdict
from typing import Callable, Any, Dict
from genai_output_parsers import Parser
from genai_output_parsers import TitleSubtitleParser


logger = logging.getLogger(__name__)


class RemoteParaphraser:

    def __init__(self, 
                 genai_provider_name="deep_seek"):
        """
        Initializes an instance of the RemoteParaphraser class.

        Args:
            genai_provider_name: a name of the GenAI provider.

        Returns:
            None
        """
        self.client = get_genai_client(genai_provider_name)


    def rephrase(self, 
                 prompt_builder: Callable[..., Dict],
                 output_parser: Parser, 
                 versions_count: int,
                 item_size: int,
                 **kwargs) -> dict[str, list[str]]:
        """
        Runs rephrasing of the provided text where the specified number of versions will
        be generated for each text item.

        Args:
            text: a list of text items, sentences, paragraphs or individual words
            versions_count: how many versions of each text item need to be generated

        Returns:
             a dictionary where keys refers to the parent titles and values to the subtitles related to the parent titles.
        """
        messages = prompt_builder(versions_count=versions_count, **kwargs)
        response = self.client.send_request(messages=messages)
        phrases = output_parser.parse(response, item_size)

        return phrases
       
        
