"""
LoRA (Low-Rank Adaptation) utilities for parameter-efficient fine-tuning.

This module provides functionality to apply LoRA to PyTorch models, allowing
efficient adaptation of large models by training only small low-rank matrices
instead of all parameters.
"""

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer wrapper.
    
    Wraps a linear layer to add trainable low-rank adaptation matrices.
    The original layer weights are frozen, and only the LoRA matrices are trained.
    
    Args:
        original_layer: The original nn.Linear layer to wrap
        rank: Rank of the low-rank matrices (default: 4)
        alpha: Scaling factor for LoRA output (default: 1.0)
    
    The forward pass computes: original_output + (x @ lora_A @ lora_B) * (alpha / rank)
    """
    def __init__(self, original_layer, rank=4, alpha=1.0):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Freeze original layer parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # Initialize LoRA matrices only for Linear layers
        if isinstance(original_layer, nn.Linear):
            # lora_A: (in_features, rank) - initialized with small random values
            self.lora_A = nn.Parameter(
                torch.randn(original_layer.in_features, rank) * 0.01
            )
            # lora_B: (rank, out_features) - initialized to zeros
            self.lora_B = nn.Parameter(torch.zeros(rank, original_layer.out_features))
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x):
        """Forward pass: original output + LoRA adaptation."""
        original_output = self.original_layer(x)
        if self.lora_A is not None and self.lora_B is not None:
            lora_output = (x @ self.lora_A) @ self.lora_B * (self.alpha / self.rank)
            return original_output + lora_output
        return original_output


def apply_lora_to_model(model, rank=4, alpha=1.0, target_modules=None):
    """
    Apply LoRA to specified linear layers in a model.
    
    Args:
        model: PyTorch model to apply LoRA to
        rank: Rank of LoRA matrices (default: 4)
        alpha: Scaling factor for LoRA output (default: 1.0)
        target_modules: List of module name patterns to apply LoRA to.
                       If None or empty list, applies to all Linear layers.
    
    Returns:
        The model with LoRA layers applied (modified in-place)
    """
    if target_modules is None:
        target_modules = [""]

    def should_apply_lora(name):
        """Check if LoRA should be applied to a module based on its name."""
        if not target_modules:
            return True
        return any(pattern in name for pattern in target_modules)

    # Iterate through all modules and wrap matching Linear layers
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and should_apply_lora(name):
            # Navigate to the parent module to replace the layer
            *parent_path, attr_name = name.split(".")
            parent = model
            for part in parent_path:
                parent = getattr(parent, part)
            
            # Replace the original layer with LoRALayer wrapper
            lora_layer = LoRALayer(module, rank=rank, alpha=alpha)
            setattr(parent, attr_name, lora_layer)
    
    return model


def get_lora_parameters(model):
    """
    Extract all LoRA parameters from a model.
    
    Args:
        model: PyTorch model containing LoRALayer modules
    
    Returns:
        List of LoRA parameter tensors (lora_A and lora_B from all LoRALayers)
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALayer):
            if module.lora_A is not None:
                lora_params.append(module.lora_A)
            if module.lora_B is not None:
                lora_params.append(module.lora_B)
    return lora_params


def get_all_optimizers(agent):
    """
    Find all optimizers in an agent object.
    
    Searches for optimizers in common locations:
    - Direct attributes (optimizer, actor_optimizer, etc.)
    - agent.policy attributes
    - agent.model attributes
    
    Args:
        agent: Agent object that may contain optimizers
    
    Returns:
        List of unique optimizer objects found
    """
    optimizers = []
    optimizers_attrs = [
        "optimizer",
        "optimizers",
        "actor_optimizer",
        "critic_optimizer",
        "policy_optimizer",
        "q_optimizer",
        "value_optimizer",
    ]
    
    # Search in agent itself
    for attr in optimizers_attrs:
        if hasattr(agent, attr):
            opt = getattr(agent, attr)
            if opt is not None:
                if isinstance(opt, (list, tuple)):
                    optimizers.extend(opt)
                else:
                    optimizers.append(opt)
    
    # Search in agent.policy
    if hasattr(agent, "policy"):
        for attr in optimizers_attrs:
            if hasattr(agent.policy, attr):
                opt = getattr(agent.policy, attr)
                if opt is not None:
                    if isinstance(opt, (list, tuple)):
                        optimizers.extend(opt)
                    else:
                        optimizers.append(opt)
    
    # Search in agent.model
    if hasattr(agent, "model"):
        for attr in optimizers_attrs:
            if hasattr(agent.model, attr):
                opt = getattr(agent.model, attr)
                if opt is not None:
                    if isinstance(opt, (list, tuple)):
                        optimizers.extend(opt)
                    else:
                        optimizers.append(opt)
    
    return list(set(optimizers))


def replace_optimizer_params_with_lora(agent, lora_params, base_lr=3e-4):
    """
    Replace optimizer parameters to train only LoRA parameters.
    
    Finds existing optimizers in the agent and replaces them with new optimizers
    that only train the LoRA parameters.
    
    Args:
        agent: Agent object containing optimizers
        lora_params: List of LoRA parameter tensors to optimize
        base_lr: Base learning rate if original optimizer doesn't have one
    
    Returns:
        True if any optimizer was replaced, False otherwise
    """
    if not lora_params:
        return False
    
    optimizers = get_all_optimizers(agent)
    replaced_any = False
    
    for opt in optimizers:
        try:
            # Get learning rate from existing optimizer
            lr = opt.defaults.get("lr", base_lr)
            
            # Create new optimizer with only LoRA parameters
            new_optimizer = torch.optim.Adam(lora_params, lr=lr)
            
            # Find and replace optimizer references in agent
            # Search in agent itself
            for attr in dir(agent):
                if getattr(agent, attr, None) is opt:
                    setattr(agent, attr, new_optimizer)
                    replaced_any = True
                    break
            
            # Search in agent.policy
            if hasattr(agent, "policy"):
                for attr in dir(agent.policy):
                    if getattr(agent.policy, attr, None) is opt:
                        setattr(agent.policy, attr, new_optimizer)
                        replaced_any = True
                        break
            
            # Search in agent.model
            if hasattr(agent, "model"):
                for attr in dir(agent.model):
                    if getattr(agent.model, attr, None) is opt:
                        setattr(agent.model, attr, new_optimizer)
                        replaced_any = True
                        break
        except Exception:
            continue
    
    return replaced_any

