import torch
import torch.nn as nn

class MultiLoraLinear(nn.Module):
    def __init__(self, base: nn.Linear, num_adapters: int, r=8, lora_alpha=16, lora_dropout=0.05):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = base.weight
        self.bias = base.bias

        self.num_adapters = num_adapters
        self.active_adapter = 0
        self.scaling = lora_alpha / r
        self.dropout = nn.Dropout(p=lora_dropout)

        self.lora_As = nn.ParameterList([
            nn.Parameter(torch.randn(r, self.in_features) * 0.01) for _ in range(num_adapters)
        ])
        self.lora_Bs = nn.ParameterList([
            nn.Parameter(torch.randn(self.out_features, r) * 0.01) for _ in range(num_adapters)
        ])

    def set_active_adapter(self, idx: int):
        self.active_adapter = idx

    def forward(self, x):
        result = nn.functional.linear(x, self.weight, self.bias)
        A = self.lora_As[self.active_adapter]
        B = self.lora_Bs[self.active_adapter]
        lora_out = self.dropout(x) @ A.T @ B.T * self.scaling
        return result + lora_out


def replace_with_multi_lora(model, target_keywords, num_adapters, r=8, alpha=16, dropout=0.05):
    for name, module in model.named_modules():
        for child_name, child_module in module.named_children():
            if isinstance(child_module, nn.Linear) and any(key in child_name for key in target_keywords):
                multi_lora = MultiLoraLinear(child_module, num_adapters, r, alpha, dropout)
                setattr(module, child_name, multi_lora)


def set_active_adapter(model, adapter_idx):
    for module in model.modules():
        if isinstance(module, MultiLoraLinear):
            module.set_active_adapter(adapter_idx)
