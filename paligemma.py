"""
Fine-tuning PaliGemma Model on VQAv2 Dataset

This script loads the VQAv2 dataset, processes it for training, and fine-tunes 
Google's PaliGemma2-3B model for Visual Question Answering (VQA). It supports 
LoRA and QLoRA for parameter-efficient fine-tuning.

Libraries Used:
- datasets: To load the dataset
- torch: For deep learning computations
- transformers: To handle PaliGemma model and processing
- peft: For LoRA fine-tuning
"""

from datasets import load_dataset
import torch
from transformers import (
    PaliGemmaProcessor, 
    PaliGemmaForConditionalGeneration, 
    Trainer, 
    TrainingArguments, 
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig

# Configuration Flags
USE_LORA = False  # Enable LoRA fine-tuning
USE_QLORA = False  # Enable QLoRA for quantization
FREEZE_VISION = False  # Freeze vision model parameters

# Load dataset (VQAv2 small version)
ds = load_dataset('merve/vqav2-small', split="validation")
ds = ds.train_test_split(test_size=0.5)["train"]  # Split into train set

# Model and Processor Setup
model_id = "google/paligemma2-3b-pt-448"
processor = PaliGemmaProcessor.from_pretrained(model_id)

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Token ID for image placeholder
image_token = processor.tokenizer.convert_tokens_to_ids("<image>")

def collate_fn(examples):
    """
    Prepares a batch of data for training by tokenizing text and processing images.

    Args:
        examples (list): A batch of dataset samples.

    Returns:
        dict: Tokenized inputs formatted for model training.
    """
    texts = ["<image>answer en " + example["question"] for example in examples]
    labels = [example["multiple_choice_answer"] for example in examples]
    images = [example["image"].convert("RGB") for example in examples]
    
    tokens = processor(
        text=texts, 
        images=images, 
        suffix=labels,
        return_tensors="pt", 
        padding="longest"
    )
    
    tokens = tokens.to(torch.bfloat16).to(device)
    return tokens

# Model Initialization with Optional LoRA/QLoRA
if USE_LORA or USE_QLORA:
    # LoRA Configuration
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    
    # QLoRA Configuration (if enabled)
    bnb_config = None
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type=torch.bfloat16
        )

    # Load Model with LoRA or QLoRA
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, 
        device_map="auto",
        quantization_config=bnb_config if USE_QLORA else None,
        torch_dtype=torch.bfloat16
    )
    
    model = get_peft_model(model, lora_config)
    model = model.to(device)
    model.print_trainable_parameters()

else:
    # Load Model Without LoRA
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, device_map="auto").to(device)

    # Freeze vision-related parameters if enabled
    if FREEZE_VISION:
        for param in model.vision_tower.parameters():
            param.requires_grad = False

        for param in model.multi_modal_projector.parameters():
            param.requires_grad = False

# Training Arguments Configuration
args = TrainingArguments(
    num_train_epochs=3,
    remove_unused_columns=False,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    learning_rate=2e-5,
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=100,
    optim="adamw_hf",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=1,
    push_to_hub=True,
    output_dir="paligemma_vqav2",
    bf16=True,
    report_to=["tensorboard"],
    dataloader_pin_memory=False
)

# Trainer Initialization
trainer = Trainer(
    model=model,
    train_dataset=ds,
    data_collator=collate_fn,
    args=args
)

# Start Training
trainer.train()
