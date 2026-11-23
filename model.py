import torch
from torch import nn

BASE_MODEL = 'answerdotai/ModernBERT-base'

class SystemOneV3(nn.Module):
    """Encode context once, and each question/option once. Attend jointly per question."""
    def __init__(self, base_model=BASE_MODEL, encoder=None, hidden=None, layers=2, heads=4):
        super().__init__()
        if encoder is None:
            from transformers import AutoModel
            encoder = AutoModel.from_pretrained(base_model)
        self.encoder = encoder
        h = hidden or self.encoder.config.hidden_size
        self.type_embedding = nn.Embedding(3, h)
        self.decision = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(h, heads, dim_feedforward=h*2, dropout=.1,
                                       batch_first=True, activation='gelu', norm_first=True),
            num_layers=layers, enable_nested_tensor=False)
        self.scorer = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h//2), nn.GELU(), nn.Linear(h//2, 1))

    def encode(self, ids, mask):
        # ModernBERT uses a CLS token at index 0.
        return self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]

    def forward(self, batch):
        # Batch consists of separately tokenized contexts, questions and options.
        c = self.encode(batch['context_ids'], batch['context_attention'])
        q = self.encode(batch['question_ids'], batch['question_attention'])
        o = self.encode(batch['option_ids'], batch['option_attention'])
        owners = batch['question_owner']
        qidx = batch['option_question']
        # One context + one question + all candidate options, per question.
        # Padding is excluded from attention and logits.
        groups = []
        lengths = []
        for j in range(q.shape[0]):
            options = o[qidx == j]
            tokens = torch.cat([c[owners[j]:owners[j]+1], q[j:j+1], options], dim=0)
            kinds = torch.cat([torch.zeros(1,device=c.device,dtype=torch.long),
                               torch.ones(1,device=c.device,dtype=torch.long),
                               torch.full((len(options),),2,device=c.device,dtype=torch.long)])
            groups.append(tokens + self.type_embedding(kinds))
            lengths.append(len(options))
        width = max(len(g) for g in groups)
        padded = torch.zeros(len(groups), width, c.shape[-1], device=c.device, dtype=c.dtype)
        padding = torch.ones(len(groups), width, device=c.device, dtype=torch.bool)
        for j,g in enumerate(groups):
            padded[j,:len(g)] = g
            padding[j,:len(g)] = False
        mixed = self.decision(padded, src_key_padding_mask=padding)
        logits = self.scorer(mixed[:,2:]).squeeze(-1)
        option_mask = ~padding[:,2:]
        logits = logits.masked_fill(~option_mask, torch.finfo(logits.dtype).min)
        return logits, option_mask
