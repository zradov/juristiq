class GenAIError(Exception):
    """Base class for all GenAI-related errors."""
    pass


class GenAIConnectionError(GenAIError):
    """Raised when there is a connection issue (network, DNS, etc.)."""
    pass


class GenAIInsufficientBalanceError(GenAIError):
    """Raised when the user has insufficient balance to perform the requested operation."""
    pass


class GenAITokenLimitError(GenAIError):
    """Raised when the request exceeds the token limit."""
    pass


class GenAIAuthError(GenAIError):
    """Raised when authentication or authorization fails."""
    pass


class GenAIRateLimitError(GenAIError):
    """Raised when rate limits are hit."""
    pass


class GenAIServerError(GenAIError):
    """Raised for 5xx server errors."""
    pass


class GenAIClientError(GenAIError):
    """Raised for 4xx client errors (invalid request, bad parameters, etc.)."""
    pass


class GenAIUnknownError(GenAIError):
    """Raised for unexpected or unmapped exceptions."""
    pass
