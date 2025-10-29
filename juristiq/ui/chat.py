
from typing import Dict, Generator
from juristiq.config.inference import InferenceParams
from juristiq.cloud.utils import get_bedrock_runtime_client


class ChatClient:
    """
    Utility class that supports chatbot UI operations such as initializing
    the Amazon Bedrock Converse API client and processing the queries.
    """
    def __init__(self, inference_params: InferenceParams):

        self.model_id = None
        self.system_prompt = None
        self.inference_params = inference_params
        self.bedrock_client = get_bedrock_runtime_client()


    def _get_converse_stream_params(self, query: str) -> Dict:
        """
        Returns arguments used when invoking the converse stream API operation.

        Args:
            query: an user query

        Returns:
            a dictionary containing mandatory arguments for the converse stream 
            API request.
        """
        return dict(
            modelId=self.model_id,
            messages=[{
                "role": "user",
                "content": [{
                    "text": query
                }]
            }],
            system=[{
                "text": self.system_prompt
            }],
            inferenceConfig=self.inference_params.to_camel_dict()
        )


    def initialize(self, model_id: str, system_prompt: str) -> None:
        """
        Sets the values for the Amazon Bedrock model identifier and the system prompt.

        Args:
            model_id: Amazon Bedrock model identifier.
        """
        self.model_id = model_id
        self.system_prompt = system_prompt


    def uninitialize(self) -> None:
        """
        Resets the values for the Amazon Bedrock model identifier and the system prompt. 
        """
        self.model_id = None
        self.system_prompt = None

    
    def is_initialized(self) -> bool:
        """
        Verifies whether the chat client mandatory properties are initialized.

        Returns:
            True if the model identifier and the system prompt have their values set otherwise False.
        """
        return self.model_id and self.system_prompt
        

    def send_query(self, query: str) -> Generator[str, None, None]:
        """
        Sends the Bedrock Converse API query request.

        Args:
            query: user query
        """
        try:
            args = self._get_converse_stream_params(query)
            response = self.bedrock_client.converse_stream(**args)
            for event in response["stream"]:
                if "contentBlockDelta" in event:
                    yield event["contentBlockDelta"]["delta"].get("text", "")
        except Exception as ex:
            yield f"Error querying Amazon Bedrock: {ex}"
