# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from moba import register_moba, MoBAConfig


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
#     parser.add_argument("--moba-chunk-size", type=int, default=4096)
#     parser.add_argument("--moba-topk", type=int, default=12)
#     parser.add_argument(
#         "--attn",
#         default="moba",
#         help="choose attention backend",
#         choices=["flash_attention_2", "moba", "moba_naive"],
#     )
#     args = parser.parse_args()

#     register_moba(MoBAConfig(args.moba_chunk_size, args.moba_topk))
#     model = AutoModelForCausalLM.from_pretrained(
#         args.model,
#         trust_remote_code=True,
#         device_map="auto",
#         torch_dtype=torch.float16,
#         attn_implementation=args.attn,
#     )
#     tknz = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

#     prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
#     input_tokens = tknz.encode(prompt)
#     input_ids = torch.tensor([input_tokens], device=model.device)
#     tokens = model.generate(input_ids, max_length=256, do_sample=False)
#     print(tokens)
#     print(tknz.decode(tokens.squeeze().tolist()))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from moba import register_moba, MoBAConfig

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
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

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map=None,
        torch_dtype=torch.float16,
        attn_implementation=args.attn,
    )
    model.to("cuda")
    model.eval()

    # Load tokenizer and set pad_token = eos
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Instead of calling .encode() manually, let the tokenizer produce input_ids + attention_mask
    prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    # Move to model device
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    # Now generate: pass both 'input_ids' and 'attention_mask'
    tokens = model.generate(**encoded, max_length=128, do_sample=True)

    # Print decoded output
    print(tokens)
    print(tokenizer.decode(tokens.squeeze(), skip_special_tokens=True))

