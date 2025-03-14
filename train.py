import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, DataCollatorWithPadding
from torch.optim import AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
import evaluate

accelerator = Accelerator()

raw_datasets = load_dataset("openai/gsm8k", "main")
checkpoint = "meta-llama/Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}[USER]: {{ message['content'] }}{% elif message['role'] == 'assistant' %}[ASSISTANT]: {{message['content']}}{% endif %}{% endfor %}"

def tokenize_function(samples):
    conversations = []
    for question, answer in zip(samples['question'], samples['answer']):
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        conversation = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            template=tokenizer.chat_template
        )
        conversations.append(conversation)

    tokenized_output = tokenizer(conversations, padding=True, truncation=True, max_length=1024)
    return tokenized_output

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=raw_datasets["train"].column_names)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=2, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=2, collate_fn=data_collator)

# Load pretrained base model
base_model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

# Apply LoRA via PEFT clearly
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5)

train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        
        labels = batch["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        
        outputs = model(
            input_ids=batch["input_ids"], 
            attention_mask=batch["attention_mask"], 
            labels=labels
        )
        loss = outputs.loss
        accelerator.backward(loss)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

metric = evaluate.load("accuracy")
model.eval()
for batch in eval_dl:
    batch = {k: v.to(accelerator.device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    predictions = torch.argmax(outputs.logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(metric.compute())
