from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP


class Qwen2MLPV3(Qwen2MLP):
    def forward(self, hidden_state):
        print("Qwen2MLPV3")

        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))