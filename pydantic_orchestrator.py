from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from dataclasses import dataclass
from visual_rag import VisualRAG
from dotenv import load_dotenv
from typing import Optional
import glob
import os
import json
import asyncio
import requests

@dataclass
class RagDeps:
    rag: VisualRAG

# 1. Point to your local Ollama server
# Note: No API key is needed for local runs!
ollama_provider = OpenAIProvider(base_url="http://localhost:11434/v1")

# Force Ollama to ignore the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 2. Define the local model name you pulled (e.g., qwen2.5:7b)
"""
Download ollama executable from https://ollama.com/download
In cmd prompt ollama pull qwen2.5:7b
Shortlisted models: qwen2:7b | qwen2.5:7b | llama-3.1:8b 

Force model to load on cpu
ollama create llama3.1-cpu -f path/Modelfile
"""
# agent_name = "qwen2.5:7b"
# agent_name = "qwen2:7b"
agent_name ="llama3.1" # run on cpu
local_model = OpenAIChatModel(
    model_name= agent_name,
    provider=ollama_provider
)

def print_memory_usage(model_name="qwen2.5:7b", host_url="http://localhost:11434"):
    """
    Activates the model first, then checks memory directly 
    from Ollama's native API.
    """
    # ACTIVATE: Send an empty prompt to preload the model into VRAM
    try:
        # keep_alive can be "5m" (default), "1h", or -1 (permanent)
        requests.post(
            f"{host_url}/api/generate",
            json={"model": model_name, "keep_alive": "5m"} 
        )
    except requests.exceptions.ConnectionError:
        print(f"[Error] Could not connect to Ollama at {host_url}.")
        return

    # CHECK SIZE: Hit the /api/ps endpoint to read the memory footprint
    ps_endpoint = f"{host_url}/api/ps"
    try:
        response = requests.get(ps_endpoint)
        response.raise_for_status()
        
        models = response.json().get("models", [])
        
        if not models:
            print("[Ollama] No models currently loaded in memory.")
            return

        for model in models:
            name = model["name"]
            
            # Ollama returns sizes in bytes
            total_size_gb = model["size"] / (1024**3)
            vram_size_gb = model.get("size_vram", 0) / (1024**3)
            ram_size_gb = total_size_gb - vram_size_gb
            
            print(f"[Ollama] {name} | System RAM: {ram_size_gb:.2f} GB | GPU VRAM: {vram_size_gb:.2f} GB")
            
    except requests.exceptions.ConnectionError:
        print(f"[Error] Failed to fetch process status from {host_url}.")

# Define the Agent
# We use a system prompt to guide the AI on how to use the RAG tools
print("Setting up RAG agent...")
rag_agent = Agent(
    local_model,
    deps_type=RagDeps,
    # system_prompt=(
    #     "You are a visual document assistant. You have access to three main capabilities:\n"
    #     "1. 'upload_document': Use this when a user wants to upload a single file (image or PDF).\n"
    #     "2. 'batch_upload_documents': Use this when a user wants to upload all files in a folder or "
    #     "mentions a directory/pattern (e.g., 'data/*.pdf').\n"
    #     "3. 'search_documents': Use this to find information from already uploaded documents to answer questions.\n"
    #     "If a user asks to 'add' or 'index' a file, use the appropriate upload tool first."
    # ),
    system_prompt=(
        "You are a visual document assistant. You have access to three main capabilities:\n"
        "1. 'upload_document': Use this for single files. (image or PDF)\n"
        "2. 'batch_upload_documents': Use this for folders/patterns.\n"
        "3. 'search_documents': Use this to find NEW information from the database.\n\n"
        "CRITICAL INSTRUCTION ON MEMORY: You have access to the chat history. "
        "If the user asks to reformat, summarize, translate, or output the PREVIOUS response in a new way (like JSON), "
        "DO NOT CALL ANY TOOLS. Generate your response directly using the information already in the chat history."
    ),
)
print_memory_usage(model_name=agent_name)

@rag_agent.tool
async def search_documents(ctx: RunContext[RagDeps], 
                           user_query: str, 
                           date: Optional[str] = None, seller: Optional[str] = None, 
                           buyer: Optional[str] = None, 
                           invoice_id: Optional[str] = None) -> str:
    # Search for documents using a natural language query and optional metadata filters.
    """
    Search for documents using a natural language query and optional metadata filters.
    
    IMPORTANT: Do NOT use this tool if the user is asking to modify, format, 
    or output the previous response (e.g., "make it json", "summarize that"). 
    Only use this tool when looking for entirely new invoices or data.
    
    Args:
        user_query: The main topic to look for (e.g., "total amount", "items list"). If there is no specific topic, pass the user's exact original query.
        date: The date of the document if explicitly mentioned (DD-MM-YYYY format).
        seller: The name of the seller/vendor if explicitly mentioned.
        buyer: The name of the buyer if explicitly mentioned.
        invoice_id: The invoice number if explicitly mentioned.
    """
    
    # Build the filter from the extracted arguments
    filters = ctx.deps.rag.build_filters(date=date, seller=seller, buyer=buyer, invoice_id=invoice_id)
    
    # Perform the search with filters
    response = await asyncio.to_thread(
        ctx.deps.rag.search, 
        query_text=user_query, 
        k=3, 
        filter_conditions=filters
    )
    
    # if not paths:
    #     return "No documents found matching those specific criteria."
    
    # response = ctx.deps.rag.inference(user_query, paths)
    
    return json.dumps(response, indent=2)

@rag_agent.tool
async def batch_upload_documents(ctx: RunContext[RagDeps], pattern: str) -> str:
    """
    Find and upload multiple documents (Images or PDFs) matching a pattern or directory.
    
    Args:
        pattern: A glob pattern or directory path (e.g., 'data/*.pdf', './invoices/').
    """
    # Standardize path
    clean_pattern = pattern.strip().strip("'").strip('"')
    
    if os.path.isdir(clean_pattern):
        search_pattern = os.path.join(clean_pattern, "*.*")
    else:
        search_pattern = clean_pattern

    files = glob.glob(search_pattern)
    valid_extensions = ('.png', '.jpg', '.jpeg', '.pdf')
    target_files = [f for f in files if f.lower().endswith(valid_extensions)]

    if not target_files:
        return f"No valid images or PDFs found matching: {search_pattern}"

    success_count = 0
    errors = []

    print(f"Starting batch upload of {len(target_files)} files...")

    for file_path in target_files:
        try:
            # We call your refactored upload_and_index inside your VisualRAG class
            result = await asyncio.to_thread(ctx.deps.rag.upload_and_index, file_path)
            if result.startswith("Error") or result.startswith("Failed"):
                errors.append(f"{os.path.basename(file_path)}: {result}")
            else:
                success_count += 1
                print(f"[{success_count}/{len(target_files)}] {result}")
        
        except Exception as e:
            errors.append(f"{file_path}: Exception -> {str(e)}")

    status = f"Batch complete. Successfully processed {success_count} out of {len(target_files)} files."
    if errors:
        error_sample = "\n  - ".join(errors[:3]) # Show first 3 errors so the prompt doesn't overflow
        status += f"\nEncountered {len(errors)} errors. Example failures:\n  - {error_sample}"
        if len(errors) > 3:
            status += f"\n  - ... and {len(errors) - 3} more."

    return status

@rag_agent.tool
async def upload_document(ctx: RunContext[RagDeps], file_path: str) -> str:
    """
    Upload and index a single document (Image or PDF).
    
    Args:
        file_path: The local path to the file.
    """
    clean_path = file_path.strip().strip("'").strip('"')
    
    if not os.path.exists(clean_path):
        return f"Error: File not found at {clean_path}"
    
    try:
        result = await asyncio.to_thread(ctx.deps.rag.upload_and_index, clean_path)
        return result
    except Exception as e:
        return f"Upload failed with exception: {str(e)}"

async def run_chatbot():
    # Initialize class
    env_path = "credentials.env"
    load_dotenv(dotenv_path=env_path)
    rag = VisualRAG(
        qdrant_url=os.getenv("QDRANT_URL"), 
        qdrant_api_key=os.getenv("QDRANT_API_KEY")
    )
    deps = RagDeps(rag=rag)
    history = []

    # Start the CLI loop
    print(r"""
    ===========================================================================
                                "Everything is going fine...or so we wish" -TK     
                   ~ ~ ;     /  "..." - add on    
                /`"--"`;    /    ----------------------------------------------   
                \ (' ') ___/    | Visual RAG Assistant                        |
           _____c\  > '____     | --------------------------------------------|
           ||**** ),_/ **'||    | Type 'quit', 'exit', or 'q' to stop.        |
           |'* ___| |___**'|    | Type 'clear' to reset conversation memory.  | 
           |' /    ~   ,\*'|    -----------------------------------------------
           |'/           \ |      |\      _,,,---,,_           ~" _^_  "~     
       ____|/ <_ _____ >  \| __ _/,`.-'`'    -.  ;-;;,_ ________ (__") _____  
      /     '-, \    / ,-'      |,4-  ) )-,_. ,\ (  `'-'    ~"  (____")     \ 
     /         (//   \\)       '---''(_/--'  `-'\_)             ~"      "~   \
    ===========================================================================
    """ 
    )

    while True:
        # Get user input
        try:
            user_input = input("\033[1mUser:\033[0m ").strip() # \033[1m makes text bold
        except EOFError:
            break

        if not user_input:
            continue

        # Handle CLI commands
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        if user_input.lower() == "clear":
            history = []
            print("\033[93m[Memory Cleared]\033[0m")
            continue

        # Run the agent
        try:
            # We pass BOTH deps (for tool access) and history (for context)
            result = await rag_agent.run(
                user_input, 
                deps=deps, 
                message_history=history
            )
            # Print the hidden thought process
            print("\033[90m--- Agent Thought Process ---\033[0m")
            for msg in result.new_messages():
                if hasattr(msg, 'parts'):
                    for part in msg.parts:
                        # Check if the agent decided to use a tool
                        if part.part_kind == 'tool-call':
                            print(f"\033[93m🛠️ Calling Tool:\033[0m {part.tool_name}")
                            print(f"\033[93m📦 With Arguments:\033[0m {part.args}")
                        
                        # Check for the raw text response (reasoning)
                        elif part.part_kind == 'text' and part.content:
                            print(f"\033[94m🧠 Reasoning:\033[0m {part.content.strip()}")
            print("\033[90m-----------------------------\033[0m\n")
            # Print the response
            print(f"\033[92mAgent:\033[0m {result}\n")

            # Update history with the new state (User + Tool + Agent messages)
            history = result.all_messages()
            
            # OPTIONAL: Keep history manageable (last 20 messages)
            if len(history) > 50:
                history = history[-50:]

        except Exception as e:
            print(f"\033[91mError:\033[0m {e}")

if __name__ == "__main__":
    asyncio.run(run_chatbot())
