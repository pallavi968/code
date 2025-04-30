from llama_stack_client import LlamaStackClient
from llama_index.core.llms import ChatMessage, CompletionResponse, LLMMetadata, CustomLLM
from pydantic import BaseModel, Field
from typing import List, Any
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
LLAMA_STACK_URL = os.getenv("LLAMA_STACK_URL", "http://183.83.216.62:8321")

# Initialize LLM client
llm_client = LlamaStackClient(base_url=LLAMA_STACK_URL, timeout=200)

# Define Custom LLM
class LlamaStackLLM(CustomLLM):
    client: LlamaStackClient = Field(description="LlamaStackClient instance")
    model_id: str = Field(description="Model identifier")

    def __init__(self, client: LlamaStackClient, model_id: str):
        super().__init__(client=client, model_id=model_id)
        self.client = client
        self.model_id = model_id

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=4096, num_output=512, model_name=self.model_id)

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
        response = self.client.inference.chat_completion(model_id=self.model_id, messages=messages)
        return CompletionResponse(text=response.completion_message.content)

    def chat(self, messages: List[ChatMessage], **kwargs: Any) -> CompletionResponse:
        formatted_messages = [{"role": msg.role.value, "content": msg.content} for msg in messages]
        response = self.client.inference.chat_completion(model_id=self.model_id, messages=formatted_messages)
        return CompletionResponse(text=response.completion_message.content)

    def stream_complete(self, prompt: str, **kwargs: Any) -> Any:
        yield self.complete(prompt, **kwargs)