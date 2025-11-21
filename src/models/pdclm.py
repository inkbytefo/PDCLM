## Developer: inkbytefo
## Modified: 2025-11-21

import torch
import torch.nn as nn
from ..layers.pse import PatternStreamEncoder
from ..memory.hmr import HierarchicalMemoryRouter

class PDCLMBase(nn.Module):
    def __init__(self, embed_dim: int = 256, num_layers: int = 4, heads: int = 4, window_size: int = 512, max_windows: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.heads = heads
        self.window_size = window_size
        self.max_windows = max_windows
        
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
        
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.lm_head = nn.Linear(embed_dim, 256)
        self.value_head = nn.Linear(embed_dim, 1)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, raw_text: str):
        min_len = getattr(self.pse, "window_size", 32)
        text_in = raw_text
        if len(text_in) < min_len:
            repeat = (min_len // max(len(text_in), 1)) + 1
            text_in = (text_in + " ") * repeat
        stream = self.pse(text_in)
        if stream.size(0) <= 1:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device, requires_grad=False)
        input_stream = stream[:-1]
        if input_stream.size(0) > self.max_windows:
            input_stream = input_stream[-self.max_windows:]
        memory, mem_weights = self.hmr(input_stream)
        mem_pad_mask = (mem_weights < 0.2).unsqueeze(0)
        def _tf(inp, mem, pad):
            return self.transformer(inp, mem, memory_key_padding_mask=pad)[0]
        output = torch.utils.checkpoint.checkpoint(_tf, input_stream.unsqueeze(0), memory.unsqueeze(0), mem_pad_mask, use_reentrant=False)
        logits = self.lm_head(output)
        device = logits.device
        bytes_seq = torch.tensor([ord(c) % 256 for c in text_in], dtype=torch.long, device=device)
        W = input_stream.size(0)
        stride = getattr(self.pse, "stride", 1)
        window_size = getattr(self.pse, "window_size", 1)
        starts = torch.arange(W, device=device) * stride
        pos = starts + window_size
        pos = torch.clamp(pos, max=bytes_seq.size(0) - 1)
        targets = bytes_seq.index_select(0, pos)
        loss = self.loss_fn(logits.view(-1, 256), targets.view(-1))
        if torch.isnan(loss):
            loss = torch.nan_to_num(loss, nan=10.0)
        return loss

    def generate_text(self, prompt: str, max_new_bytes: int = 256, temperature: float = 1.0, top_k: int = 0, top_p: float = 0.9, return_traces: bool = False):
        self.eval()
        device = next(self.parameters()).device
        text = prompt
        with torch.no_grad():
            actions = []
            logprobs = []
            vals = []
            for _ in range(max_new_bytes):
                min_len = getattr(self.pse, "window_size", 32)
                txt = text
                if len(txt) < min_len:
                    repeat = (min_len // max(len(txt), 1)) + 1
                    txt = (txt + " ") * repeat
                stream = self.pse(txt)
                if stream.size(0) < 2:
                    break
                input_stream = stream[-2:]
                memory, mem_weights = self.hmr(input_stream)
                mem_pad_mask = (mem_weights < 0.2).unsqueeze(0)
                output = self.transformer(
                    input_stream.unsqueeze(0),
                    memory.unsqueeze(0),
                    memory_key_padding_mask=mem_pad_mask
                )[0]
                logits = self.lm_head(output[-1])
                val = self.value_head(output[-1]).squeeze(-1)
                probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
                if top_k > 0:
                    topk_vals, indices = torch.topk(probs, k=min(top_k, probs.size(0)))
                    mask = torch.zeros_like(probs)
                    mask[indices] = 1.0
                    probs = probs * mask
                    denom = probs.sum()
                    probs = probs / (denom + 1e-12)
                if 0.0 < top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumulative = torch.cumsum(sorted_probs, dim=0)
                    cutoff = (cumulative <= top_p).float()
                    cutoff[0] = 1.0
                    filtered = sorted_probs * cutoff
                    denom = filtered.sum()
                    filtered = filtered / (denom + 1e-12)
                    choice = torch.multinomial(filtered, num_samples=1)
                    byte_idx = sorted_idx[choice].item()
                else:
                    byte_idx = torch.multinomial(probs, num_samples=1).item()
                actions.append(byte_idx)
                lp = torch.log_softmax(logits, dim=-1)[byte_idx]
                logprobs.append(float(lp.item()))
                vals.append(float(val.item()))
                try:
                    text += chr(byte_idx)
                except Exception:
                    text += " "
                if len(text) > 2 and "Final answer:" in text:
                    break
        if return_traces:
            return text, {"actions": actions, "logprobs": logprobs, "values": vals}
        return text

    def evaluate_sequence_logprobs(self, prompt: str, actions: list[str | int]):
        self.train()
        device = next(self.parameters()).device
        text = prompt
        logprobs = []
        values = []
        for a in actions:
            min_len = getattr(self.pse, "window_size", 32)
            txt = text
            if len(txt) < min_len:
                repeat = (min_len // max(len(txt), 1)) + 1
                txt = (txt + " ") * repeat
            stream = self.pse(txt)
            if stream.size(0) < 2:
                break
            input_stream = stream[-2:]
            memory, mem_weights = self.hmr(input_stream)
            mem_pad_mask = (mem_weights < 0.2).unsqueeze(0)
            output = self.transformer(
                input_stream.unsqueeze(0),
                memory.unsqueeze(0),
                memory_key_padding_mask=mem_pad_mask
            )[0]
            logits = self.lm_head(output[-1])
            val = self.value_head(output[-1]).squeeze(-1)
            ai = int(a) if isinstance(a, int) else ord(str(a)[0]) % 256
            lp = torch.log_softmax(logits, dim=-1)[ai]
            logprobs.append(lp)
            values.append(val)
            try:
                text += chr(ai)
            except Exception:
                text += " "
        return torch.stack(logprobs), torch.stack(values)
        
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


def pretrain_step(model: nn.Module, raw_text_batch: str, optimizer: torch.optim.Optimizer, device: torch.device, mini_batch_chars: int | None = None, do_step: bool = True, scale_loss: float = 1.0):
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
    if do_step:
        optimizer.zero_grad()
    
    # Mixed precision training
    if not mini_batch_chars or mini_batch_chars >= len(raw_text_batch):
        if device.type == 'cuda':
            with torch.autocast('cuda', dtype=torch.float16):
                loss = model(raw_text_batch)
        else:
            loss = model(raw_text_batch)
        (loss * scale_loss).backward()
        if do_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        return loss.item()
    total_loss = 0.0
    count = 0
    for i in range(0, len(raw_text_batch), mini_batch_chars):
        sub = raw_text_batch[i:i + mini_batch_chars]
        if not sub:
            continue
        if device.type == 'cuda':
            with torch.autocast('cuda', dtype=torch.float16):
                loss = model(sub)
        else:
            loss = model(sub)
        (loss * scale_loss).backward()
        total_loss += float(loss.item())
        count += 1
    if do_step:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    return total_loss / max(count, 1)


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
