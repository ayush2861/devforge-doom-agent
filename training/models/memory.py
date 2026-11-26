"""GRU + Differentiable Episodic Slot Memory for long-horizon DOOM navigation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentMemory(nn.Module):
    """GRU core augmented with a gated episodic key-value slot bank.

    Packs both GRU hidden state (hidden_dim) and episodic slots (n_slots * slot_dim)
    into a single flat state tensor (1, B, state_dim) for seamless BPTT and rollout storage.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 384, n_slots: int = 8, slot_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.state_dim = hidden_dim + n_slots * slot_dim

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)

        # Episodic slot attention read/write
        self.read_query = nn.Linear(hidden_dim, slot_dim)
        self.read_proj = nn.Linear(slot_dim, hidden_dim)
        self.write_gate = nn.Linear(hidden_dim, 1)
        self.write_addr = nn.Linear(hidden_dim, n_slots)
        self.write_val = nn.Linear(hidden_dim, slot_dim)

        self.out_norm = nn.LayerNorm(hidden_dim)

    def init_hidden(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.zeros(1, batch_size, self.state_dim, device=device, dtype=torch.float32)

    def _step_core(
        self, x_step: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = x_step.shape[0]
        gru_h = state[:, :, : self.hidden_dim].contiguous()  # (1, B, H)
        slots = state[:, :, self.hidden_dim :].squeeze(0).view(b, self.n_slots, self.slot_dim)

        gru_out, next_gru_h = self.gru(x_step.unsqueeze(1), gru_h)
        h_vec = gru_out.squeeze(1)  # (B, H)

        # Attention read from episodic slots
        q = self.read_query(h_vec).unsqueeze(1)  # (B, 1, D_s)
        attn_logits = torch.bmm(q, slots.transpose(1, 2)) / (self.slot_dim ** 0.5)  # (B, 1, K)
        attn_w = F.softmax(attn_logits, dim=-1)
        mem_read = torch.bmm(attn_w, slots).squeeze(1)  # (B, D_s)

        # Gated write into episodic slots (e.g., doors, keys, landmarks)
        gate = torch.sigmoid(self.write_gate(h_vec)).unsqueeze(-1)  # (B, 1, 1)
        addr = F.softmax(self.write_addr(h_vec), dim=-1).unsqueeze(-1)  # (B, K, 1)
        val = torch.tanh(self.write_val(h_vec)).unsqueeze(1)  # (B, 1, D_s)
        alpha = gate * addr  # (B, K, 1)
        next_slots = slots * (1.0 - alpha) + alpha * val

        out = self.out_norm(h_vec + self.read_proj(mem_read))
        next_state = torch.cat([next_gru_h, next_slots.reshape(1, b, -1)], dim=-1)
        return out, next_state

    def forward_step(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step forward: x (B, D), hidden (1, B, state_dim) -> (out (B, H), next_hidden)."""
        if hidden is None:
            hidden = self.init_hidden(x.shape[0], x.device)
        return self._step_core(x, hidden)

    def forward_sequence(
        self,
        x_seq: torch.Tensor,
        hidden: torch.Tensor | None = None,
        dones: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sequence forward for BPTT: x_seq (B, T, D), dones (B, T) -> (out_seq (B, T, H), next_hidden)."""
        b, t, _ = x_seq.shape
        if hidden is None:
            hidden = self.init_hidden(b, x_seq.device)

        outputs = []
        state = hidden
        for step in range(t):
            if step > 0 and dones is not None:
                keep = (1.0 - dones[:, step - 1].float()).view(1, b, 1)
                state = state * keep
            out_t, state = self._step_core(x_seq[:, step, :], state)
            outputs.append(out_t)
        return torch.stack(outputs, dim=1), state
