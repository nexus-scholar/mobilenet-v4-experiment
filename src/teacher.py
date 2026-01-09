import torch
from transformers import CLIPModel, CLIPTokenizer


def load_clip_model(model_name: str = "openai/clip-vit-base-patch32"):
    """
    Load CLIP model and tokenizer as the teacher model.

    The teacher is frozen (eval mode, no gradients) since we never train it.

    Args:
        model_name: HuggingFace model name (default: "openai/clip-vit-base-patch32")

    Returns:
        tuple: (model, tokenizer)
    """
    # Load model and tokenizer
    model = CLIPModel.from_pretrained(model_name)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)

    # Freeze the teacher model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model, tokenizer


def get_text_features(
    model: CLIPModel,
    text_tokens: dict,
    device: torch.device
) -> torch.Tensor:
    """
    Extract and normalize text features from CLIP model.

    Args:
        model: CLIP model (teacher)
        text_tokens: Tokenized text input (dict with 'input_ids', 'attention_mask')
        device: Torch device to run inference on

    Returns:
        L2-normalized text features [B, D]
    """
    with torch.no_grad():
        # Move tokens to device
        tokens_on_device = {
            key: val.to(device) for key, val in text_tokens.items()
        }

        # Get text features from CLIP
        text_features = model.get_text_features(**tokens_on_device)

        # L2 normalize for cosine similarity (CLIP uses normalized embeddings)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features


def encode_class_prompts(
    model: CLIPModel,
    tokenizer: CLIPTokenizer,
    class_names: list,
    device: torch.device,
    prompt_template: str = "a photo of a {} plant leaf"
) -> torch.Tensor:
    """
    Encode class names into CLIP text features using a prompt template.

    Args:
        model: CLIP model (teacher)
        tokenizer: CLIP tokenizer
        class_names: List of class names
        device: Torch device
        prompt_template: Template string with {} placeholder for class name

    Returns:
        L2-normalized text features for all classes [num_classes, D]
    """
    # Create prompts for each class
    prompts = [prompt_template.format(name) for name in class_names]

    # Tokenize all prompts
    text_tokens = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # Get normalized text features
    text_features = get_text_features(model, text_tokens, device)

    return text_features

