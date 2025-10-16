import logging
from .clients import GenAIClientFactory


logger = logging.getLogger(__name__)


def log_user_balance(genai_provider_name: str) -> None:

    try:
        client = GenAIClientFactory.create(genai_provider_name)
        balance, currency = client.get_user_balance()
        logger.info(f"Current user balance: {balance:.5f} {currency}")
    except Exception as e:
        logger.warning(f"Failed to retrieve user balance: {e}")
