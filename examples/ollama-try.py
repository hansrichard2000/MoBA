import torch
from moba import register_moba, MoBAConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistral")
    parser.add_argument("--moba-chunk-size", type=int, default=4096)
    parser.add_argument("--moba-topk", type=int, default=12)
    parser.add_argument(
        "--attn",
        default="moba",
        help="choose attention backend",
        choices=["flash_attention_2", "moba", "moba_naive"],
    )
    args = parser.parse_args()
    
    # Register MoBA
    register_moba(MoBAConfig(args.moba_chunk_size, args.moba_topk))
    
    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = ChatPromptTemplate.from_template(template)
    
    # Load Model
    model = OllamaLLM(
        model=args.model,
        num_gpu=2,
        topk=args.moba_topk
    )
    
    chain = prompt | model

    response = chain.invoke({"question": "What is LangChain?"})
    
    print(response)
