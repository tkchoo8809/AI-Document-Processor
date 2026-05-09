import os
import asyncio
from visual_rag import VisualRAG
from dotenv import load_dotenv
from typing import Any
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge
from pydantic_evals.evaluators.llm_as_a_judge import set_default_judge_model
from pydantic_orchestrator import rag_agent, RagDeps

# Evaluation
ollama_provider = OpenAIProvider(base_url="http://localhost:11434/v1")
# ollama_provider = OpenAIProvider(base_url="http://localhost:4000/v1")

judge_model = OpenAIChatModel(
    model_name="qwen2.5:14b",
    # model_name="llama3.1",
    provider=ollama_provider
)

set_default_judge_model(judge_model)

async def evaluate_rag_sequence(conversation_turns: list[str]) -> str:
    """
    Target function executed by pydantic-evals. 
    It runs through the sequence of turns and returns the final tool state and output.
    """
    env_path = "credentials.env"
    load_dotenv(dotenv_path=env_path)
    rag = VisualRAG(
        qdrant_url=os.getenv("QDRANT_URL"), 
        qdrant_api_key=os.getenv("QDRANT_API_KEY")
    )
    deps = RagDeps(rag=rag)
    
    chat_history = []
    result = None
    transcript = ""
    
    # Process the conversation history
    for turn in conversation_turns:
        transcript += f"User: {turn}\n" # Log user input

        result = await rag_agent.run(
            turn, 
            deps=deps, 
            message_history=chat_history
        )
        chat_history = result.all_messages()

        transcript += f"Agent: {result.output}\n\n"
        
    # Extract tools only from the final turn
    tools_called = []
    for msg in result.new_messages():
        if hasattr(msg, 'parts'):
            for part in msg.parts:
                if part.part_kind == 'tool-call':
                    tools_called.append(f"{part.tool_name}({part.args})")
                    
    tools_str = ", ".join(tools_called) if tools_called else "None"
    output_text = str(result.output) if result.output else "No output returned"

    # Using result.output to match the current Pydantic AI spec
    # final_output = f"--- CONVERSATION TRANSCRIPT ---\n{transcript}\n--- FINAL TURN DATA ---\nTools Used: {tools_str}\nFinal Output: {result.output}"
    final_output = (
        f"--- CONVERSATION TRANSCRIPT ---\n{transcript}\n"
        f"--- FINAL TURN DATA ---\n"
        f"Tools Used: {tools_str}\n"
        f"Final Output: {output_text}"
    )
    print(final_output)
    final_output = final_output.replace("`", "'").strip()
    return final_output

# Define the Dataset using generics [InputType, OutputType, MetadataType]
turn_1_history = [
    "search and extract document 257667"
]

turn_2_history = turn_1_history + [
    "are there any other invoices from the same buyer?"
]

turn_3_history = turn_2_history + [
    "format the previously searched result in proper json"
]

turn_4_history = turn_3_history + [
    "check if the totals add up"
]

rag_dataset = Dataset[list[str], str, Any](
    cases=[
        Case(
            name='Turn 1: Initial Metadata Search',
            inputs=turn_1_history,
            expected_output=None,
            evaluators=(
                LLMJudge(
                    rubric="""
                    EVALUATION PROTOCOL:
                    1. FOCUS: Look ONLY at the "FINAL TURN DATA" section.
                    2. TOOL CHECK: Verify 'Tools Used' contains 'search_documents'.
                    3. ARGUMENT CHECK: Verify the argument includes the exact string '257667'.
                    4. VERDICT: PASS if both (2) and (3) are true. Otherwise, FAIL.
                    """,
                    assertion={'evaluation_name': 'Tool_Routing_Check', 'include_reason': True},
                ),
            ),
        ),
        Case(
            name='Turn 2: Contextual Search (Memory)',
            inputs=turn_2_history,
            expected_output=None,
            evaluators=(
                LLMJudge(
                    rubric="""
                    EVALUATION PROTOCOL:
                    1. FOCUS: Look ONLY at the "FINAL TURN DATA" section.
                    2. TOOL CHECK: Verify 'Tools Used' contains 'search_documents'.
                    3. ARGUMENT CHECK: Verify the argument includes the buyer name 'BLUE SPARK DESIGN'.
                    4. VERDICT: PASS if the agent correctly targets the buyer from memory. FAIL otherwise.
                    """,
                    assertion={'evaluation_name': 'Memory_Context_Check', 'include_reason': True},
                ),
            ),
        ),
        Case(
            name='Turn 3: Formatting Constraint (No Tools)',
            inputs=turn_3_history,
            expected_output=None,
            evaluators=(
                LLMJudge(
                    rubric="""
                    EVALUATION PROTOCOL:
                    1. FOCUS: Look ONLY at the "FINAL TURN DATA" section.
                    2. NEGATIVE CONSTRAINT: Verify 'Tools Used' is exactly 'None'.
                    3. FORMAT CHECK: Verify 'Final Output' contains a valid JSON structure representing the invoice.
                    4. VERDICT: PASS only if NO tools were used and the output is JSON. FAIL if any tool was called.
                    """,
                    assertion={'evaluation_name': 'Formatting_Constraint_Check', 'include_reason': True},
                ),
            ),
        ),
        Case(
            name='Turn 4: Reasoning and Calculation (No Tools)',
            inputs=turn_4_history,
            expected_output=None,
            evaluators=(
                LLMJudge(
                    rubric="""
                    EVALUATION PROTOCOL:
                    1. FOCUS: Look ONLY at the "FINAL TURN DATA" section.
                    2. NEGATIVE CONSTRAINT: Verify 'Tools Used' is exactly 'None'.
                    3. REASONING CHECK: Verify the 'Final Output' explicitly identifies that the total amount does NOT tally/match the sum of line items.
                    4. VERDICT: PASS if the agent identifies the math discrepancy without using tools.
                    """,
                    assertion={'evaluation_name': 'Math_Logic_Score', 'include_reason': True},
                ),
            ),
        ),
    ]
)

if __name__ == "__main__":
    report = rag_dataset.evaluate_sync(evaluate_rag_sequence)
    report.print(include_reasons=True)
    # print(report)


"""
                                                             Evaluation Summary: qwen2.5:7b                                                             
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID                                      ┃ Assertions                                                                                             ┃ Duration ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ Turn 1: Initial Metadata Search              │ Tool_Routing_Check: ✔                                                                                  │    88.7s │
│                                              │   Reason: The output shows that the 'search_documents' tool was used with the argument {'invoice_id':  │          │
│                                              │   '257667', 'user_query': 'extract document 257667'}                                                   │          │
│                                              │                                                                                                        │          │
│                                              │                                                                                                        │          │
├──────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│ Turn 2: Contextual Search (Memory)           │ Memory_Context_Check: ✔                                                                                │   339.8s │
│                                              │   Reason: The 'search_documents' tool was called with the correct buyer's name 'BLUE SPARK DESIGN'.    │          │
│                                              │   Reason: The output shows that the 'search_documents' tool was used with the argument {'invoice_id':  │          │
│                                              │   '257667', 'user_query': 'extract document 257667'}                                                   │          │
│                                              │                                                                                                        │          │
│                                              │                                                                                                        │          │
├──────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│ Turn 2: Contextual Search (Memory)           │ Memory_Context_Check: ✔                                                                                │   339.8s │
│                                              │   Reason: The 'search_documents' tool was called with the correct buyer's name 'BLUE SPARK DESIGN'.    │          │
│                                              │   '257667', 'user_query': 'extract document 257667'}                                                   │          │
│                                              │                                                                                                        │          │
│                                              │                                                                                                        │          │
├──────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│ Turn 2: Contextual Search (Memory)           │ Memory_Context_Check: ✔                                                                                │   339.8s │
│                                              │   Reason: The 'search_documents' tool was called with the correct buyer's name 'BLUE SPARK DESIGN'.    │          │
│                                              │                                                                                                        │          │
│                                              │                                                                                                        │          │
├──────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│ Turn 3: Formatting Constraint (No Tools)     │ Formatting_Constraint_Check: ✔                                                                         │   540.6s │
│                                              │   Reason: The output correctly formats the data into JSON without using any tools.                     │          │
│                                              │                                                                                                        │          │
│                                              │                                                                                                        │          │
├──────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│ Turn 4: Reasoning and Calculation (No Tools) │ Math_Logic_Score: ✔                                                                                    │   623.9s │
│                                              │   Reason: The agent correctly identified that the sum of the line items ($940.0) does not match the    │          │
│                                              │   net total of the invoice ($9963.0), indicating a discrepancy. The agent did not use any tools,       │          │
│                                              │   adhering to the rubric.                                                                              │          │
│                                              │                                                                                                        │          │
│                                              │                                                                                                        │          │
├──────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│ Averages                                     │ 100.0% ✔                                                                                               │   398.3s |
└──────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────┘
"""

""" 
    MODEL RESULTS | Tested with upload_and_index & embed function respectively | Best option indicated with ***
    -----------------------------------------------------------------------------------------------------------------------
    
    Vision Language Models | Tested on: [image] batch2-0001_augmented.jpg | * Retrieval model offloaded to CPU
    --------------------------------------------------------------------------------------------------------------------------------------------
    | Models                                                | Inference (s)  | Augmented     | JSON Format   | RAM Usage (GB)  | Accuracy (%)  |
    | "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit" ***         | 47.48          | Y             | Y             | 9.1             | 99.60         |
    | "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit"             | 64.25          | Y             | N             | 6.1             | -             |
    | "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"       | 67.77          | Y             | Y             | 10.3            | 99.60         |
    | "unsloth/Pixtral-12B-2409-bnb-4bit"                   | 115.15         | N             | N             | 10.7            | -             | 
    | "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"      | 51.69          | Y             | N             | 9.1             | -             |
    --------------------------------------------------------------------------------------------------------------------------------------------
    
    Retrieval Models | Tested on: [image] batch2-0001_augmented.jpg [query] "Extract invoice 257667" | * VL model loaded in GPU (6.44)
    --------------------------------------------------------------------------------------------------------------------------------------------
    | Models                                                | Image Embedding (s)  | Text Embedding (s)   | RAM Usage (GB)  | Vector Dimension
    | "vidore/colqwen2-v1.0-hf ***                          | 1.78                 | 0.96                 | 7.83 / 7.86     | 128 
    | "vidore/colpali-v1.3-hf"                              | 1.68                 | 1.10                 | 8.59 / 8.59     | 128  
    | "jinaai/jina-embeddings-v4"                           | 2.81                 | 2.05                 | 9.38 / 9.36     | 128  
    | "nvidia/nemotron-colembed-vl-4b-v2"                   | -                    | -                    | -               | 2560 (mismatch size) 
    --------------------------------------------------------------------------------------------------------------------------------------------

    
    FUNCTION RESULTS | Tested with increasing data points
    -----------------------------------------------------------------------------------------------------------------------
    
    Query Search | Tested on: [images] batch1_1 (100-200) [query] "Search and extract invoice 257667" | * unique points
    --------------------------------------------------------------------------------------------------------------------------------------------
    | Collection                                | Points                          | Prefetch & Rerank (s)           |
    | test100                                   | 107                             | 1.55                            |
    | test200                                   | 207                             | 1.47                            |
    | docs_collection                           | 307                             | 1.61                            |
    --------------------------------------------------------------------------------------------------------------------------------------------
    
"""