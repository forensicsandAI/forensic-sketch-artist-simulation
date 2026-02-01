import os
import PIL.Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from typing import List, Tuple, Dict

torch.cuda.empty_cache()

# Load model
model_path = "/home/Bandana/models/Janus_prp_7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# Disable gradients globally, we'll enable selectively
for param in vl_gpt.parameters():
    param.requires_grad = False


# =============================================================================
# Selecting which features to analyze
# =============================================================================
SELECTED_FEATURES = [
    # Demographics
    "adult",
    "male",
    "Indian",
    "23",
    "28",
    
    # Face shape
    "oval",
    "rounded",
    "jawline",
    "symmetrical",
    "proportions",
    
    # Skin
    "medium brown",
    "brown",
    "smooth",
    "even-toned",
    
    # Hair
    "black",
    "straight",
    "dense",
    "short",
    "hairline",
    
    # Forehead
    "forehead",
    "convex",
    
    # Eyebrows
    "eyebrows",
    "thick",
    "dark",
    "arch",
    
    # Eyes
    "eyes",
    "almond",
    "almond-shaped",
    "dark brown",
    "irises",
    "alert",
    
    # Nose
    "nose",
    "straight",
    "dorsum",
    "bridge",
    "nostrils",
    
    # Cheeks
    "cheeks",
    "broad",
    
    # Mouth/Lips
    "lips",
    "upper lip",
    "lower lip",
    "thin",
    "fuller",
    
    # Chin
    "chin",
    "angular",
    
    # Expression
    "neutral",
    "calm",
]
# =============================================================================


def prepare_prompt(text_prompt: str) -> Tuple[str, List]:
    """Prepare conversation and prompt for generation."""
    conversation = [
        {"role": "<|User|>", "content": text_prompt},
        {"role": "<|Assistant|>", "content": ""},
    ]
    
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    
    prompt = sft_format + vl_chat_processor.image_start_tag
    return prompt, conversation


def find_feature_positions(input_ids, tokenizer, features):
    """Find token positions for specified features with exact matching."""
    tokens_text = [tokenizer.decode([tid]).lower().strip() for tid in input_ids]
    
    feature_map = {}
    for feature in features:
        feature_lower = feature.lower()
        positions = []
        
        # For multi-word features (like "medium brown")
        if ' ' in feature_lower:
            words = feature_lower.split()
            # Look for consecutive tokens that match the multi-word feature
            for i in range(len(tokens_text) - len(words) + 1):
                # Check if consecutive tokens match
                match = True
                matched_positions = []
                for j, word in enumerate(words):
                    token = tokens_text[i + j]
                    if word == token:
                        matched_positions.append(i + j)
                    else:
                        match = False
                        break
                
                if match and matched_positions:
                    positions.extend(matched_positions)
        else:
            # For single-word features, exact match only
            for i, token in enumerate(tokens_text):
                if token == feature_lower:
                    positions.append(i)
        
        if positions:
            # Remove duplicates while preserving order
            positions = sorted(list(set(positions)))
            feature_map[feature] = positions
    
    all_positions = sorted(set(p for positions in feature_map.values() for p in positions))
    
    print(f"\nFound {len(all_positions)} feature token positions (from {len(feature_map)} matched features out of {len(features)} requested)")
    print(f"Feature mapping (showing first 10):")
    for idx, (feature, positions) in enumerate(list(feature_map.items())[:10]):
        tokens = [tokens_text[p] for p in positions]
        print(f"  '{feature}' → positions {positions} (tokens: {tokens})")
    if len(feature_map) > 10:
        print(f"  ... and {len(feature_map) - 10} more features")
    
    # Print features that weren't found
    found_features = set(feature_map.keys())
    requested_features = set(features)
    missing_features = requested_features - found_features
    if missing_features:
        print(f"\nFeatures not found in prompt ({len(missing_features)}):")
        for feature in sorted(list(missing_features))[:10]:
            print(f"  - '{feature}'")
        if len(missing_features) > 10:
            print(f"  ... and {len(missing_features) - 10} more")
    
    return feature_map, all_positions


def register_layer_hooks(model, target_layers: List[int]) -> Dict:
    """Register forward hooks to capture intermediate layer outputs."""
    layer_outputs = {}
    hooks = []
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            layer_outputs[layer_idx] = output[0] if isinstance(output, tuple) else output
        return hook
    
    for layer_idx in target_layers:
        layer = model.language_model.model.layers[layer_idx]
        hook = layer.register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)
    
    return layer_outputs, hooks


def compute_selective_layerwise_saliency(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    feature_positions: List[int],
    target_layers: List[int] = [0, 14, 29],
    num_generation_steps: int = 576,
    saliency_frequency: int = 100,
    temperature: float = 1,
    cfg_weight: float = 5,
    img_size: int = 384,
    patch_size: int = 16,
):
    """Generate image and compute layer-wise saliency ONLY for selected features."""
    
    # Tokenize
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids).cuda()
    num_tokens = len(input_ids)
    
    print(f"\nTotal prompt tokens: {num_tokens}")
    print(f"Computing saliency for {len(feature_positions)} selected feature positions")
    print(f"Layers to analyze: {target_layers}")
    print(f"Saliency frequency: every {saliency_frequency} tokens")
    
    # Storage for layer-wise saliency (full size but only features will be computed)
    layerwise_importance = {layer: torch.zeros(num_tokens).cuda() for layer in target_layers}
    generated_tokens = torch.zeros((1, num_generation_steps), dtype=torch.int).cuda()
    
    # Track embeddings
    all_embeds_cond = []
    all_embeds_uncond = []
    
    # Initial prompt embeddings
    tokens_cond = input_ids.unsqueeze(0)
    tokens_uncond = input_ids.clone().unsqueeze(0)
    tokens_uncond[0, 1:-1] = vl_chat_processor.pad_id
    
    inputs_embeds_cond = mmgpt.language_model.get_input_embeddings()(tokens_cond)
    inputs_embeds_uncond = mmgpt.language_model.get_input_embeddings()(tokens_uncond)
    
    all_embeds_cond.append(inputs_embeds_cond)
    all_embeds_uncond.append(inputs_embeds_uncond)
    
    print(f"\nGenerating {num_generation_steps} image tokens...")
    
    # Convert feature_positions to set for fast lookup
    feature_positions_set = set(feature_positions)
    
    # Generate tokens and compute saliency
    for i in range(num_generation_steps):
        
        # Compute layer-wise saliency periodically
        if i % saliency_frequency == 0:
            print(f"  Step {i}/{num_generation_steps} - Computing saliency for {len(feature_positions)} features...")
            
            # Create fresh embeddings with gradient tracking ONLY for feature positions
            tokens_cond_detached = tokens_cond.clone()
            prompt_embeds_base = mmgpt.language_model.get_input_embeddings()(tokens_cond_detached).detach()
            
            # Build full sequence with image tokens
            if len(all_embeds_cond) > 1:
                image_embeds_no_grad = torch.cat(all_embeds_cond[1:], dim=1).detach()
                full_embeds_base = torch.cat([prompt_embeds_base, image_embeds_no_grad], dim=1)
            else:
                full_embeds_base = prompt_embeds_base
            
            # Register hooks to capture layer outputs
            layer_outputs, hooks = register_layer_hooks(mmgpt, target_layers)
            
            # Process each feature position
            for feat_pos in feature_positions:
                # Create a copy of embeddings
                full_embeds_with_grad = full_embeds_base.clone()
                
                # Enable gradient ONLY for this specific feature position
                feature_token = tokens_cond[:, feat_pos:feat_pos+1]
                feature_embed = mmgpt.language_model.get_input_embeddings()(feature_token)
                feature_embed.requires_grad_(True)
                
                # Replace just this one position
                full_embeds_with_grad[:, feat_pos:feat_pos+1, :] = feature_embed
                
                # Forward pass
                outputs_cond = mmgpt.language_model.model(
                    inputs_embeds=full_embeds_with_grad,
                    use_cache=False
                )
                
                hidden_states_cond = outputs_cond.last_hidden_state
                logits_cond = mmgpt.gen_head(hidden_states_cond[:, -1, :])
                
                # Compute target score
                probs = torch.softmax(logits_cond / temperature, dim=-1)
                max_prob = probs.max()
                
                # Backward to get gradient for this feature
                max_prob.backward()
                
                # Extract gradient magnitude for this feature
                if feature_embed.grad is not None:
                    grad_magnitude = feature_embed.grad.abs().sum()
                    # Store in all layers (simplified - same importance across layers for this approach)
                    for layer_idx in target_layers:
                        layerwise_importance[layer_idx][feat_pos] += grad_magnitude.detach()
                
                # Cleanup
                del feature_embed, full_embeds_with_grad, outputs_cond, hidden_states_cond, logits_cond, probs, max_prob
                torch.cuda.empty_cache()
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            print(f"    Saliency computed for all {len(feature_positions)} features")
        
        # Regular generation pass (without gradients)
        with torch.no_grad():
            full_embeds = torch.cat([
                torch.cat(all_embeds_cond, dim=1),
                torch.cat(all_embeds_uncond, dim=1)
            ], dim=0)
            
            outputs = mmgpt.language_model.model(
                inputs_embeds=full_embeds,
                use_cache=False
            )
            
            hidden_states = outputs.last_hidden_state
            logits = mmgpt.gen_head(hidden_states[:, -1, :])
            
            # CFG
            logit_cond = logits[0:1, :]
            logit_uncond = logits[1:2, :]
            logits_combined = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            
            # Sample
            probs = torch.softmax(logits_combined / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            
            # Prepare next embeddings
            next_token_single = next_token.squeeze()
            img_embed_cond = mmgpt.prepare_gen_img_embeds(next_token_single.unsqueeze(0))
            img_embed_uncond = mmgpt.prepare_gen_img_embeds(next_token_single.unsqueeze(0))
            
            all_embeds_cond.append(img_embed_cond.unsqueeze(1))
            all_embeds_uncond.append(img_embed_uncond.unsqueeze(1))
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_generation_steps} tokens")
    
    # Decode image
    print("\nDecoding image...")
    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[1, 8, img_size//patch_size, img_size//patch_size]
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)
    generated_image = dec[0].astype(np.uint8)
    
    # Normalize saliency for each layer (only non-zero values)
    for layer_idx in target_layers:
        importance = layerwise_importance[layer_idx].cpu().numpy()
        non_zero = importance[importance > 0]
        if len(non_zero) > 0 and non_zero.max() > non_zero.min():
            for i in range(len(importance)):
                if importance[i] > 0:
                    importance[i] = (importance[i] - non_zero.min()) / (non_zero.max() - non_zero.min())
        layerwise_importance[layer_idx] = importance
    
    return generated_image, layerwise_importance, input_ids.cpu().numpy()


def visualize_feature_saliency(
    layerwise_importance: Dict[int, np.ndarray],
    input_ids: np.ndarray,
    tokenizer,
    feature_positions: List[int],
    save_path: str = "feature_saliency.png"
):
    """Visualize layer-wise saliency for selected features only."""
    
    tokens_text = [tokenizer.decode([tid]) for tid in input_ids]
    layers = sorted(layerwise_importance.keys())
    
    # Extract data for feature positions only
    feature_tokens = [tokens_text[p] for p in feature_positions]
    feature_saliency = {
        layer: np.array([layerwise_importance[layer][p] for p in feature_positions])
        for layer in layers
    }
    
    num_layers = len(layers)
    num_features = len(feature_positions)
    
    # Create matrix
    saliency_matrix = np.zeros((num_layers, num_features))
    for i, layer in enumerate(layers):
        saliency_matrix[i, :] = feature_saliency[layer]
    
    # Create figure
    fig = plt.figure(figsize=(max(20, num_features * 0.4), 12))
    
    # Heatmap
    ax1 = plt.subplot(2, 1, 1)
    im = ax1.imshow(saliency_matrix, cmap='hot', aspect='auto', interpolation='nearest')
    
    ax1.set_xticks(range(num_features))
    ax1.set_xticklabels(feature_tokens, rotation=90, ha='right', fontsize=8)
    ax1.set_yticks(range(num_layers))
    ax1.set_yticklabels([f"Layer {l}" for l in layers], fontsize=10)
    
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Importance Score', rotation=270, labelpad=20, fontsize=12)
    
    ax1.set_title(f'Selected Feature Saliency ({num_features} features across {num_layers} layers)', 
                  fontsize=14, pad=20)
    ax1.set_xlabel('Selected Features', fontsize=12)
    ax1.set_ylabel('Model Layers', fontsize=12)
    
    ax1.set_xticks(np.arange(num_features) - 0.5, minor=True)
    ax1.set_yticks(np.arange(num_layers) - 0.5, minor=True)
    ax1.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Line plot
    ax2 = plt.subplot(2, 1, 2)
    for i, layer in enumerate(layers):
        ax2.plot(range(num_features), feature_saliency[layer], 
                label=f'Layer {layer}', linewidth=2, alpha=0.7, marker='o', markersize=3)
    
    ax2.set_xticks(range(num_features))
    ax2.set_xticklabels(feature_tokens, rotation=90, ha='right', fontsize=8)
    ax2.set_ylabel('Importance Score', fontsize=12)
    ax2.set_xlabel('Selected Features', fontsize=12)
    ax2.set_title('Feature Importance Across Layers', fontsize=14, pad=15)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nFeature saliency visualization saved to {save_path}")
    
    # Print rankings
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE RANKINGS")
    print("="*80)
    
    for layer in layers:
        scores = feature_saliency[layer]
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        
        print(f"\nLayer {layer} (Top 15):")
        for rank, (idx, score) in enumerate(ranked[:15], 1):
            token = feature_tokens[idx]
            pos = feature_positions[idx]
            print(f"  {rank:2d}. '{token:20s}' (position {pos:3d}) - {score:.4f}")


# Main execution
if __name__ == "__main__":
    
    prompt_text = """An adult male aged approximately 23–28 years, most likely of Indian Origin, exhibiting an oval facial shape with slightly rounded jawline and balanced vertical proportions. The complexion is medium brown with smooth, even-toned skin and fine natural texture. The hair is black, straight, and dense, trimmed short at the sides with moderate volume on the crown, forming a natural, symmetrical frontal hairline with slight tapering at the temples. The forehead is medium in height and width, smooth and slightly convex, showing no visible lines or irregularities. The eyebrows are thick, dark, and evenly distributed, extending slightly beyond the medial canthi with a gentle natural arch following the supraorbital ridge. The eyes are medium-sized, almond-shaped, and horizontally aligned, with dark brown irises and clear sclerae, indicating an alert and composed gaze. The upper eyelids display moderate creasing while the lower eyelids appear smooth and natural, consistent with relaxed facial expression. The nose exhibits a straight dorsum with medium-width bridge and a softly rounded nasal tip. The nostrils are symmetrical with mild alar flare and a moderately wide nasal base. The cheeks are broad with subtle zygomatic definition, producing a gentle midfacial curvature. The mouth is of moderate width with lips closed in a neutral expression. The upper lip is thin to medium, while the lower lip is slightly fuller, maintaining smooth vermilion definition. The philtrum is shallow and vertically aligned with the nasal septum and chin. The chin is moderately broad and rounded with mild anterior projection, aligning smoothly with a symmetrical jawline of soft angular contour. Ears are moderately sized, symmetrically placed, and partially visible. The neck is proportionate to head width and of medium length. The expression is neutral, calm, and direct. Bilateral features, eyes, eyebrows, and nostrils show good symmetry. Facial ratios including interocular distance, nasal width, and mouth width fall within normal proportional limits. Stable morphological features such as bone structure and nose shape dominate, while hairstyle and expression are transient attributes."""
    
    full_prompt, _ = prepare_prompt(prompt_text)
    
    print("=" * 80)
    print("SELECTIVE FEATURE SALIENCY ANALYSIS")
    print("=" * 80)
    print(f"\nSelected features to analyze: {len(SELECTED_FEATURES)}")
    print(f"Features: {', '.join(SELECTED_FEATURES[:15])}...")
    
    # Tokenize to find feature positions
    input_ids = vl_chat_processor.tokenizer.encode(full_prompt)
    input_ids_np = np.array(input_ids)
    
    # Find feature positions
    print("\n" + "=" * 80)
    print("STEP 1: Locating Selected Features in Prompt")
    print("=" * 80)
    
    feature_map, all_feature_positions = find_feature_positions(
        input_ids_np, tokenizer, SELECTED_FEATURES
    )
    
    if not all_feature_positions:
        print("\nERROR: No selected features found in prompt!")
        print("Check that feature names match words in your prompt.")
        exit(1)
    
    # Generate image and compute saliency
    print("\n" + "=" * 80)
    print("STEP 2: Generating Image & Computing Feature Saliency")
    print("=" * 80)
    
    target_layers = [0, 14, 29]  # Early, middle, late
    
    generated_img, layerwise_sal, token_ids = compute_selective_layerwise_saliency(
        vl_gpt,
        vl_chat_processor,
        full_prompt,
        all_feature_positions,
        target_layers=target_layers,
        num_generation_steps=576,
        saliency_frequency=100,
    )
    
    # Save outputs
    os.makedirs('selective_saliency_output', exist_ok=True)
    
    img_path = 'selective_saliency_output/generated_image.jpg'
    PIL.Image.fromarray(generated_img).save(img_path)
    print(f"\nGenerated image saved to {img_path}")
    
    # Visualize
    print("\n" + "=" * 80)
    print("STEP 3: Creating Visualizations")
    print("=" * 80)
    
    visualize_feature_saliency(
        layerwise_sal,
        token_ids,
        tokenizer,
        all_feature_positions,
        save_path='selective_saliency_output/selected_feature_saliency.png'
    )
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nOutputs:")
    print(f"  - selective_saliency_output/generated_image.jpg")
    print(f"  - selective_saliency_output/selected_feature_saliency.png")
    
    torch.cuda.empty_cache()