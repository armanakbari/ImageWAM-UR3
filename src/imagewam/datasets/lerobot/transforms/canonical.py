from __future__ import annotations

import torch


class RobotWin14ToCanonicalEE16:
    """Pad RoboTwin 14D vectors into the 16D pretrain space.

    This does not convert RoboTwin's joint/action representation into end-effector
    poses. The two 7D RoboTwin groups are copied unchanged; dimensions 7 and 15
    are zero-filled placeholders matching the InternData-A1 checkpoint layout.
    """

    def __init__(self, key: str = "default"):
        self.key = str(key)

    @staticmethod
    def _to_canonical(x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 14:
            raise ValueError(f"Expected 14D RoboTwin vector, got {x.shape[-1]}D.")
        out = x.new_zeros(*x.shape[:-1], 16)
        out[..., 0:7] = x[..., 0:7]
        out[..., 8:15] = x[..., 7:14]
        return out

    @staticmethod
    def _from_canonical(x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 16:
            raise ValueError(f"Expected 16D canonical EE vector, got {x.shape[-1]}D.")
        return torch.cat([x[..., 0:7], x[..., 8:15]], dim=-1)

    @staticmethod
    def _canonical_dim_mask(device: torch.device | None = None) -> torch.Tensor:
        return torch.tensor(
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
            ],
            dtype=torch.bool,
            device=device,
        )

    def forward(self, batch: dict) -> dict:
        if "action" in batch and self.key in batch["action"]:
            batch["action"][self.key] = self._to_canonical(batch["action"][self.key])
            batch["action_dim_is_pad"] = self._canonical_dim_mask(batch["action"][self.key].device)

        if "state" in batch and self.key in batch["state"]:
            batch["state"][self.key] = self._to_canonical(batch["state"][self.key])
            batch["state_dim_is_pad"] = self._canonical_dim_mask(batch["state"][self.key].device)

        batch["embodiment"] = batch.get("embodiment", "robotwin_ee16")
        return batch

    def backward(self, batch: dict) -> dict:
        if "action" in batch and self.key in batch["action"]:
            batch["action"][self.key] = self._from_canonical(batch["action"][self.key])
        if "state" in batch and self.key in batch["state"]:
            batch["state"][self.key] = self._from_canonical(batch["state"][self.key])
        return batch
