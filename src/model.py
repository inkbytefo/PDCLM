## Developer: inkbytefo
## Modified: 2025-11-16

import torch
import torch.nn as nn
from .pse import PatternStreamEncoder
from .hmr import HierarchicalMemoryRouter

class PDCLMBase(nn.Module):
    def __init__(self, embed_dim: int = 256, num_layers: int = 4, heads: int = 4, window_size: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.heads = heads
        self.window_size = window_size
        
        # PSE as input encoder with adapted embed_dim
        self.pse = PatternStreamEncoder(embed_dim=embed_dim, window_size=window_size)
        self.hmr = HierarchicalMemoryRouter(embed_dim=embed_dim)
        
        # Transformer decoder setup
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=heads, 
            batch_first=True,
            dim_feedforward=embed_dim * 4  # Standard transformer dimension
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection for next-pattern prediction
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # Loss function for continuous pattern prediction
        self.loss_fn = nn.MSELoss()

    def forward(self, raw_text: str, target_shift: int = 1):
        """Forward pass with next-pattern prediction"""
        # Get pattern stream from PSE
        stream = self.pse(raw_text)  # [seq_windows, embed_dim]
        
        # Handle insufficient sequence length
        if stream.size(0) <= target_shift:
            # Return zero loss for very short sequences
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=False)
        
        # Create input-target pairs for autoregressive training
        input_stream = stream[:-target_shift]
        target_stream = stream[target_shift:]
        
        # Create memory for transformer (initially zeros, can be improved)
        memory, mem_weights = self.hmr(input_stream)
        mem_pad_mask = (mem_weights < 0.2).unsqueeze(0)
        output = self.transformer(
            input_stream.unsqueeze(0),
            memory.unsqueeze(0),
            memory_key_padding_mask=mem_pad_mask
        )[0]
        
        # Project to pattern space
        logits = self.output_proj(output)
        
        # Compute loss
        loss = self.loss_fn(logits, target_stream)
        
        return loss

    def generate(self, raw_text: str, max_new_tokens: int = 100):
        """Generate new patterns from input text"""
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            stream = self.pse(raw_text)
            if stream.size(0) == 0:
                return ""
            generated = stream.clone()
            for _ in range(max_new_tokens):
                if generated.size(0) < 2:
                    break
                input_stream = generated[-2:]
                memory, mem_weights = self.hmr(input_stream)
                mem_pad_mask = (mem_weights < 0.2).unsqueeze(0)
                output = self.transformer(
                    input_stream.unsqueeze(0),
                    memory.unsqueeze(0),
                    memory_key_padding_mask=mem_pad_mask
                )[0]
                next_pattern = self.output_proj(output[-1:])
                generated = torch.cat([generated, next_pattern], dim=0)
            return generated
        
    def get_attention_weights(self, raw_text: str):
        """Get attention weights for analysis (requires hooks)"""
        self.eval()
        with torch.no_grad():
            stream = self.pse(raw_text)
            if stream.size(0) <= 1:
                return None
            
            input_stream = stream[:-1]
            memory, mem_weights = self.hmr(input_stream)
            mem_pad_mask = (mem_weights < 0.2).unsqueeze(0)
            output = self.transformer(
                input_stream.unsqueeze(0),
                memory.unsqueeze(0),
                memory_key_padding_mask=mem_pad_mask
            )[0]
            return output

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self):
        """Get model architecture information"""
        return {
            'embed_dim': self.embed_dim,
            'num_layers': self.num_layers,
            'heads': self.heads,
            'window_size': self.window_size,
            'total_params': self.count_parameters(),
            'pse_params': sum(p.numel() for p in self.pse.parameters() if p.requires_grad),
            'transformer_params': sum(p.numel() for p in self.transformer.parameters() if p.requires_grad),
            'output_proj_params': sum(p.numel() for p in self.output_proj.parameters() if p.requires_grad)
        }


def pretrain_step(model: nn.Module, raw_text_batch: str, optimizer: torch.optim.Optimizer, device: torch.device):
    """
    Single pretraining step with mixed precision and gradient clipping.
    
    Args:
        model: PDCLMBase model
        raw_text_batch: Input text batch
        optimizer: Optimizer instance
        device: Training device
        
    Returns:
        loss.item(): Loss value for logging
    """
    model.train()
    optimizer.zero_grad()
    
    # Mixed precision training
    with torch.autocast(device.type if device.type == 'cuda' else 'cpu'):
        loss = model(raw_text_batch)
    
    # Backward pass with gradient scaling for mixed precision
    loss.backward()
    
    # Gradient clipping to prevent NaN
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss.item()


def create_batches(text: str, batch_size: int = 10000, stride: int = None):
    """
    Create sliding window batches from text.
    
    Args:
        text: Input text
        batch_size: Size of each batch (characters)
        stride: Stride between batches (default: batch_size // 2)
        
    Yields:
        Text batches for training
    """
    if stride is None:
        stride = batch_size // 2
    
    for i in range(0, len(text) - batch_size + 1, stride):
        yield text[i:i + batch_size]


if __name__ == "__main__":
    # Test the model
    model = PDCLMBase()
    print("Model created successfully!")
    print(f"Model info: {model.get_model_info()}")
    
    # Test forward pass
    test_text = "This is a test string for pattern stream encoding. " * 50  # ~2000 chars
    loss = model(test_text)
    print(f"Test loss: {loss.item():.4f}")
    
    # Test parameter counting
    total_params = model.count_parameters()
    print(f"Total parameters: {total_params:,}")
    
    # Test on GPU if available
    if torch.cuda.is_available():
        model_gpu = PDCLMBase().cuda()
        loss_gpu = model_gpu(test_text)
        print(f"GPU test loss: {loss_gpu.item():.4f}")
