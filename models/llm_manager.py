import os
import logging
from typing import Dict, Any, Optional, List, Union
import torch
from langchain_huggingface import HuggingFacePipeline
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import transformers

from config import CONFIG

logger = logging.getLogger(__name__)

class LLMManager:
    """Manager for loading and using LLM models."""
    
    def __init__(self):
        """Initialize the LLM manager."""
        logger.info("Initializing LLM Manager")
        self.llm = None
        self.model_type = None
        self.model_name = None
   
    def configure_model(self,model_name: str,provider: str,api_key: Optional[str] = None) -> None:
        """
        Configure and load the model based on the provider.

        Args:
            model_name: Name of the model to load
            provider: Provider type ('local' or 'groq')
            api_key: API key for remote providers like Groq (optional for local)
        """
        logger.info(f"Configuring model: provider={provider}, model_name={model_name}")
        
        if provider == "local":
            self.load_local_model(model_name=model_name)
        elif provider == "groq":
            self.load_groq_model(model_name=model_name, api_key=api_key)
        else:
            raise ValueError(f"Unsupported model provider: {provider}")
    
    def load_local_model(self, model_name: str = CONFIG.DEFAULT_LOCAL_MODEL) -> None:
        """
        Load a local HuggingFace model with device-aware logic including GPU, CPU, Apple MPS, and offloading.
    
        Args:
        model_name: Name of the HuggingFace model to load
        """
        logger.info(f"Loading local model: {model_name}")
        

        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
    
        try:
            model_config = transformers.AutoConfig.from_pretrained(model_name)
    
            if device == "cuda":
                vram = torch.cuda.get_device_properties(0).total_memory
                if vram > 14e9:  # > 14GB VRAM
                    logger.info("Loading model in full precision on CUDA")
                    model = transformers.AutoModelForCausalLM.from_pretrained(
                        model_name,
                        config=model_config,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        trust_remote_code=True
                    )
                else:
                    logger.info("Loading model with 4-bit quantization on CUDA")
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16
                    )
                    model = transformers.AutoModelForCausalLM.from_pretrained(
                        model_name,
                        config=model_config,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True
                    )
            elif device == "mps":
                logger.info("Loading model on Apple MPS (Metal Performance Shaders)")
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    model_name,
                    config=model_config,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                ).to("mps")
            else:
                logger.info("Loading model on CPU with offloading (may be slow)")
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    model_name,
                    config=model_config,
                    device_map="auto",
                    offload_folder="offload",
                    offload_state_dict=True,
                    trust_remote_code=True
                )
    
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    
            # Text generation pipeline
            pipeline = transformers.pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=CONFIG.MAX_NEW_TOKENS,
                temperature=CONFIG.TEMPERATURE,
                repetition_penalty=1.1,
                return_full_text=False
            )
    
            self.llm = HuggingFacePipeline(pipeline=pipeline)
            self.model_type = "local"
            self.model_name = model_name
    
            logger.info(f"Successfully loaded local model: {model_name}")
    
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise




    def load_groq_model(self, model_name: str = CONFIG.DEFAULT_GROQ_MODEL, api_key: Optional[str] = None) -> None:
        """
        Load a model from Groq.
        
        Args:
            model_name: Name of the Groq model to use
            api_key: Groq API key (if None, will try to get from environment)
        """
        logger.info(f"Loading Groq model: {model_name}")
        
        if api_key is None:
            api_key = os.environ.get("GROQ_API_KEY")
            if api_key is None:
                raise ValueError("Groq API key not provided and not found in environment variables")
        
        try:
            self.llm = ChatGroq(
                groq_api_key=api_key,
                model_name=model_name,
                temperature=CONFIG.TEMPERATURE,
                max_tokens=CONFIG.MAX_NEW_TOKENS
            )
            self.model_type = "groq"
            self.model_name = model_name
            
            logger.info(f"Successfully loaded Groq model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load Groq model: {e}")
            raise

    def generate_response(
        self, 
        query: str, 
        retrieved_docs: List[Document] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a response using the loaded LLM.
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents from the vector store
            system_prompt: Optional system prompt to use
            
        Returns:
            Dictionary with the generated response and metadata
        """
        if self.llm is None:
            raise ValueError("LLM not initialized. Call load_local_model or load_groq_model first.")
        
        logger.info(f"Generating response for query: {query}")
        
        # Format retrieved documents
        context = ""
        if retrieved_docs:
            context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(retrieved_docs)])
        
        # Default system prompt if not provided
        if system_prompt is None:
            system_prompt = (
                "You are a helpful customer support assistant. Using the provided context from previous "
                "support conversations, generate a helpful, accurate, and professional response to the user's query. "
                "If the context doesn't provide enough information to answer the question properly, provide a helpful "
                "general response but be honest about what you don't know."
            )
    
        # Construct chat-style message format
        messages = [{"role": "system", "content": system_prompt}]
        
        if context:
            messages.append({
                "role": "system",
                "content": f"Here is relevant context from previous customer (not from this customer) conversations:\n{context}"
            })
            
        messages.append({"role": "user", "content": query})
    
        # Invoke the LLM
        if self.model_type == "groq":
            response = self.llm.invoke(messages)
            response_text = response.content
        else:
            # Assuming local model accepts chat-like message structure
            response = self.llm.invoke(messages)
            response_text = response.content if hasattr(response, "content") else response
    
        logger.info("Successfully generated response")
        
        return {
            "query": query,
            "response": response_text.strip(),
            "model_type": self.model_type,
            "model_name": self.model_name,
            "num_retrieved_docs": len(retrieved_docs) if retrieved_docs else 0,
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    manager = LLMManager()
    
    try:
        manager.load_local_model()
        
        query = "I need help resetting my password"
        response = manager.generate_response(query)
        
        print("\nGenerated response:")
        print(f"Query: {response['query']}")
        print(f"Response: {response['response']}")
        print(f"Model: {response['model_name']} ({response['model_type']})")
        
    except Exception as e:
        logger.warning(f"Skipping local model test: {e}")
    
    groq_api_key = os.environ.get("GROQ_API_KEY") 
    if groq_api_key:
        try:
            manager.load_groq_model(api_key=groq_api_key)
            
            # Test
            query = "I need help resetting my password"
            response = manager.generate_response(query)
            
            print("\nGenerated response:")
            print(f"Query: {response['query']}")
            print(f"Response: {response['response']}")
            print(f"Model: {response['model_name']} ({response['model_type']})")
            
        except Exception as e:
            logger.warning(f"Skipping Groq model test: {e}")
    else:
        logger.warning("Skipping Groq model test: GROQ_API_KEY not found in environment variables")
