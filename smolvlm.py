import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics3ForConditionalGeneration
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
import os
from PIL import Image
from transformers.image_utils import load_image

# Configuration flags
USE_LORA = False  # Use LoRA (Low-Rank Adaptation) for fine-tuning
USE_QLORA = True  # Use QLoRA (Quantized LoRA) for fine-tuning
SMOL = True  # Use the smaller model (SmolVLM-Base) if True, else use Idefics3-8B-Llama3

# Model ID selection based on the SMOL flag
model_id = "HuggingFaceTB/SmolVLM-Base" if SMOL else "HuggingFaceM4/Idefics3-8B-Llama3"

# Load the processor for the model
processor = AutoProcessor.from_pretrained(model_id)

# Set CUDA device order and visible devices
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 4"  # Use GPUs 1 and 4

# Configure LoRA or QLoRA if enabled
if USE_QLORA or USE_LORA:
    # LoRA configuration
    lora_config = LoraConfig(
        r=8,  # Rank of the low-rank matrices
        lora_alpha=8,  # Scaling factor for LoRA
        lora_dropout=0.1,  # Dropout rate for LoRA
        target_modules=['down_proj', 'o_proj', 'k_proj', 'q_proj', 'gate_proj', 'up_proj', 'v_proj'],  # Target modules for LoRA
        use_dora=False if USE_QLORA else True,  # Use DoRA (Dynamic Low-Rank Adaptation) if not using QLoRA
        init_lora_weights="gaussian"  # Initialize LoRA weights with Gaussian distribution
    )
    lora_config.inference_mode = False  # Disable inference mode for training

    # QLoRA configuration (if enabled)
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Load model in 4-bit precision
            bnb_4bit_use_double_quant=True,  # Use double quantization for 4-bit
            bnb_4bit_quant_type="nf4",  # Use 4-bit NormalFloat quantization
            bnb_4bit_compute_dtype=torch.bfloat16  # Use bfloat16 for computation
        )

    # Load the model with QLoRA or LoRA configuration
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config if USE_QLORA else None,  # Apply QLoRA config if enabled
        _attn_implementation="flash_attention_2",  # Use Flash Attention 2 for efficiency
        device_map="auto"  # Automatically map model to available devices
    )
    model.add_adapter(lora_config)  # Add LoRA adapter to the model
    model.enable_adapters()  # Enable adapters for training
    model = prepare_model_for_kbit_training(model)  # Prepare the model for k-bit training
    model = get_peft_model(model, lora_config)  # Apply LoRA to the model
    print(model.get_nb_trainable_parameters())  # Print the number of trainable parameters
else:
    # Load the model without LoRA or QLoRA
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,  # Use bfloat16 precision
        _attn_implementation="flash_attention_2",  # Use Flash Attention 2
    ).to("cuda")  # Move the model to the GPU

    # Freeze the vision model parameters if only fine-tuning the LLM
    for param in model.model.vision_model.parameters():
        param.requires_grad = False

# Load the VQAv2 dataset
ds = load_dataset('merve/vqav2-small', trust_remote_code=True)

# Split the dataset into training and validation sets
split_ds = ds["validation"].train_test_split(test_size=0.8)
train_ds = split_ds["train"]

# Get the token ID for the special image token
image_token_id = processor.tokenizer.additional_special_tokens_ids[
    processor.tokenizer.additional_special_tokens.index("<image>")
]

# Collate function for batching data
def collate_fn(examples):
    texts = []
    images = []
    for example in examples:
        image = example["image"]
        if image.mode != 'RGB':  # Ensure the image is in RGB format
            image = image.convert('RGB')
        question = example["question"]
        answer = example["multiple_choice_answer"]

        # Format the input as a chat message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Answer briefly."},
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=False)
        texts.append(text.strip())
        images.append([image])

    # Process the batch of texts and images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Ignore padding tokens in loss
    labels[labels == image_token_id] = -100  # Ignore image tokens in loss
    batch["labels"] = labels

    return batch

# Extract the model name for saving and logging
model_name = model_id.split("/")[-1]

# Training arguments
training_args = TrainingArguments(
    num_train_epochs=1,  # Number of training epochs
    per_device_train_batch_size=8,  # Batch size per device
    gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
    warmup_steps=50,  # Number of warmup steps for learning rate scheduler
    learning_rate=1e-4,  # Learning rate
    weight_decay=0.01,  # Weight decay for regularization
    logging_steps=25,  # Log metrics every 25 steps
    save_strategy="steps",  # Save model checkpoint every `save_steps`
    save_steps=250,  # Save model every 250 steps
    save_total_limit=1,  # Keep only the latest checkpoint
    optim="paged_adamw_8bit",  # Use paged AdamW optimizer for 8-bit training
    bf16=True,  # Use bfloat16 precision
    output_dir=f"./{model_name}-vqav2",  # Output directory for saving checkpoints
    hub_model_id=f"{model_name}-vqav2",  # Model ID for pushing to Hugging Face Hub
    report_to="tensorboard",  # Log metrics to TensorBoard
    remove_unused_columns=False,  # Keep all columns in the dataset
    gradient_checkpointing=True  # Enable gradient checkpointing to save memory
)

# Initialize the Trainer
trainer = Trainer(
    model=model,  # The model to train
    args=training_args,  # Training arguments
    data_collator=collate_fn,  # Collate function for batching
    train_dataset=train_ds,  # Training dataset
)

# Train the model
trainer.train()

# Push the trained model to Hugging Face Hub
trainer.push_to_hub()
