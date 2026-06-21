import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config

class ARModel(nn.Module):
    """
    Autoregressive model for CAD generation based on GPT2
    """
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=8,
                 dim_feedforward=512, dropout=0.1, max_seq_len=1024, pad_token_id=None,
                 use_image_prefix=False, image_feature_dim=1024, num_image_prefix_tokens=0):
        super(ARModel, self).__init__()
        self.use_image_prefix = use_image_prefix
        self.image_feature_dim = image_feature_dim
        self.num_image_prefix_tokens = num_image_prefix_tokens
        total_positions = max_seq_len + (num_image_prefix_tokens if use_image_prefix else 0)

        config_kwargs = dict(
            vocab_size=vocab_size,
            n_positions=total_positions,
            n_embd=d_model,
            n_layer=num_layers,
            n_head=nhead,
            n_inner=dim_feedforward,
            activation_function="gelu_new",
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            bos_token_id=0,
            eos_token_id=0,
            loss_type=None
        )
        if pad_token_id is not None:
            config_kwargs['pad_token_id'] = pad_token_id
        self.config = GPT2Config(**config_kwargs)

        self.model = GPT2LMHeadModel(self.config)
        self.transformer = self.model.transformer
        self.lm_head = self.model.lm_head

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.total_positions = total_positions

        if self.use_image_prefix:
            self.image_prefix_proj = nn.Linear(image_feature_dim, d_model)
            self.image_prefix_norm = nn.LayerNorm(d_model)

        rank = int(os.environ.get('RANK', '0')) if 'RANK' in os.environ else 0
        if rank == 0:
            print(f"Initialized ARModel with vocab_size={vocab_size}, d_model={d_model}, "
                  f"layers={num_layers}, heads={nhead}, max_seq_len={max_seq_len}, "
                  f"use_image_prefix={use_image_prefix}, num_image_prefix_tokens={num_image_prefix_tokens}")

    def _build_image_prefix(self, image_features):
        if image_features is None:
            raise ValueError("image_features must be provided when image prefix is enabled")
        if image_features.dim() != 3:
            raise ValueError(f"image_features must be 3D [B, P, C], got shape={tuple(image_features.shape)}")

        pooled = F.adaptive_avg_pool1d(
            image_features.transpose(1, 2),
            self.num_image_prefix_tokens
        ).transpose(1, 2)
        prefix = self.image_prefix_proj(pooled)
        prefix = self.image_prefix_norm(prefix)
        return prefix

    def forward(self, input_ids, attention_mask=None, labels=None, image_features=None):
        """
        Forward pass through the model

        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            labels (torch.Tensor): Labels for language modeling [batch_size, seq_len]
            image_features (torch.Tensor): Frozen image patch features [batch_size, num_patches, feat_dim]

        Returns:
            outputs: Model outputs including loss and logits
        """
        if self.use_image_prefix and image_features is not None:
            token_embeds = self.transformer.wte(input_ids)
            prefix_embeds = self._build_image_prefix(image_features).to(dtype=token_embeds.dtype)
            inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

            batch_size = input_ids.shape[0]
            prefix_mask = torch.ones(
                batch_size,
                self.num_image_prefix_tokens,
                dtype=attention_mask.dtype if attention_mask is not None else torch.long,
                device=input_ids.device
            )
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=prefix_mask.dtype)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            if labels is not None:
                prefix_labels = torch.full(
                    (batch_size, self.num_image_prefix_tokens),
                    -100,
                    dtype=labels.dtype,
                    device=labels.device
                )
                labels = torch.cat([prefix_labels, labels], dim=1)

            return self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

    def generate(self, input_ids, attention_mask=None, image_features=None, **kwargs):
        """
        Generate sequences using the model

        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            **kwargs: Additional arguments for generation

        Returns:
            torch.Tensor: Generated token IDs
        """

        generation_kwargs = {
            'early_stopping': False,
            'do_sample': True,
            'temperature': 0.6,
            'top_p': 0.6,
            'top_k': 0,
            'repetition_penalty': 1,
            'pad_token_id': self.config.pad_token_id,
            'eos_token_id': self.config.eos_token_id,
            'bos_token_id': self.config.bos_token_id,
        }

        generation_kwargs.update(kwargs)

        if self.use_image_prefix and image_features is not None:
            token_embeds = self.transformer.wte(input_ids)
            prefix_embeds = self._build_image_prefix(image_features).to(dtype=token_embeds.dtype)
            inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

            batch_size = input_ids.shape[0]
            prefix_mask = torch.ones(
                batch_size,
                self.num_image_prefix_tokens,
                dtype=attention_mask.dtype if attention_mask is not None else torch.long,
                device=input_ids.device
            )
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=prefix_mask.dtype)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            return self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **generation_kwargs
            )

        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs
        )

    def save_pretrained(self, save_directory):
        self.model.save_pretrained(save_directory)

    def load_state_dict(self, state_dict, strict=True):
        normalized_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.') or key.startswith('image_prefix_'):
                normalized_state_dict[key] = value
            elif key.startswith('transformer.') or key.startswith('lm_head.'):
                normalized_state_dict[f'model.{key}'] = value
            else:
                normalized_state_dict[key] = value

        has_prefix_module = any(
            key.startswith('image_prefix_') or key.startswith('model.image_prefix_')
            for key in normalized_state_dict.keys()
        )

        if not has_prefix_module and all(
            not key.startswith('model.') and not key.startswith('image_prefix_')
            for key in state_dict.keys()
        ):
            return self.model.load_state_dict(state_dict, strict=strict)

        return super().load_state_dict(normalized_state_dict, strict=strict)
