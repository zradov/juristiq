import re
import logging
from typing import Any, Tuple
from collections import defaultdict


logger = logging.getLogger(__name__)


class Parser:
    """
    A base parser class to parse the output of GenAI API requests.
    """
    def __init__(self, pattern: str):
        """
        Initializes the parser with a regex pattern to identify titles and subtitles.

        Args:
            pattern: a regex pattern to identify titles and subtitles.
        """
        self.pattern = pattern
        self.regex = re.compile(pattern, re.IGNORECASE)


    def _get_item(self, matches: re.Match) -> Tuple[bool, Any]:
        """
        Extracts the item from the regex match object.

        Args:
            matches: a regex match object.
        """
        pass


    def _post_processing(self, 
                         phrases: dict[str, list[Any]],
                         item_size: int) ->  dict[str, list[Any]]:
        """
        Post-processes the parsed phrases to ensure that each parent title 
        has the expected number of subtitles.

        Args:
            phrases: a dictionary where keys refers to the parent titles and values to the subtitles related to the parent titles.
            item_size: the expected number of subtitles for each parent title.

        Returns:
            a dictionary where keys refers to the parent titles and values to the subtitles related to the parent titles.
        """
        return phrases


    def parse(self, output: str, item_size: int) -> dict[str, list[Any]]:
        """
        Parses the output the GenAI API request in returns the structured response.
        It expects that the output conforms to the following format:
        1. Title1
        1.1 Title1_Subtitle1
        1.2 Title1_Subtitle2
        2. Title2
        2.1 Title2_Subtitle1
        
        Args:
            output: the string response of the GenAI API request.

        Returns:
            a dictionary where keys refers to the parent titles and values to the subtitles related to the parent titles.
        """
        phrases = defaultdict(list)
        lines = output.split("\n")
        parent_title = None
        
        for line in lines:
            matches = self._process_line(line)

            if matches:
                is_parent, item = self._get_item(matches)

                if is_parent:
                    parent_title = item
                    continue

                phrases[parent_title].append(item)

        phrases = self._post_processing(phrases, item_size)

        return phrases
     

    def _process_line(self, line: str) -> re.Match | None:
        """
        Processes a single line of the output.

        Args:
            line: a single line of the output.

        Returns:
            a regex match object if the line matches the pattern, None otherwise.
        """
        line = line.strip()
        
        if not line:
            logger.info(f"Line '{line}' is empty.")
            return
        matches = self.regex.match(line)
        if not matches:
            logger.warn(f"Line '{line}' does not match the pattern '{self.pattern}'.")
            return
        
        return matches
     

class TitleSubtitleParser(Parser):
    """
    A parser class to parse the output of GenAI API requests that contain titles and subtitles.
    """
    def __init__(self):
        super().__init__(r"^(?P<index>\d+)[.]((?P<subindex>\d+)[.]?)?\s+(?P<title>.+)$")


    def _get_item(self, matches: re.Match) -> Tuple[bool, Any]:
        _, _, subindex, title = matches.groups()
        is_parent = not subindex

        return is_parent, title
    

class ClauseAugmentationParser(Parser):
    """
    A parser class to parse the output of GenAI API requests that contain clauses and their augmentations.
    """
    def __init__(self):
        super().__init__(r"^\s*[[]?(?P<key>[^:\]]+)[]]?\s*(:\s*(?P<value>.+[.])\s*)?$")


    def _post_processing(self, 
                         phrases: dict[str, list[Any]], 
                         item_size: int) -> dict[str, list[Any]]:
        result = defaultdict(list)

        for clause, phrases in phrases.items():
            for i in range(0, len(phrases), item_size):
                merged_dict = {}
                # because of the issue https://github.com/zradov/juristiq/issues/1 
                # the code below is wrapped in try/except block
                try:
                    for j in range(i, i + item_size):
                        merged_dict.update(phrases[j])
                except IndexError as ex:
                    logger.error(f"Indexing error: {ex}")
                    # Don't add partially populated dictionary to final results dictionary. 
                    break
                result[clause].append(merged_dict)

        return result


    def _get_item(self, matches: re.Match) -> Tuple[bool, Any]:
        key, _, value = matches.groups()
        is_parent = not value

        return is_parent, key if is_parent else {key: value}
