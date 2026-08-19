from deepmd.pt.model.atomic_model.base_atomic_model import (
    BaseAtomicModel,
)


class SomeAtomicModel(BaseAtomicModel, torch.nn.Module):
    def __init__(self, arg1: bool, arg2: float) -> None:
        pass

    def forward_atomic(self):
        pass
