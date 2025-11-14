"""バッチ処理モジュール"""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor

from .data.data import OutputData
from .utility.pytorch_utility import to_device


@dataclass
class BatchOutput:
    """バッチ処理後のデータ構造"""

    input_wave_list: list[Tensor]  # [(L, 1)]
    target_wave_list: list[Tensor]  # [(L, 1)]
    noise_wave_list: list[Tensor]  # [(L, 1)]
    lf0_list: list[Tensor]  # [(L, 1)]
    t: Tensor  # (B,)
    r: Tensor  # (B,)
    speaker_id: Tensor  # (B,)

    @property
    def data_num(self) -> int:
        """バッチサイズを返す"""
        return self.t.shape[0]

    def to_device(self, device: str, non_blocking: bool) -> Self:
        """データを指定されたデバイスに移動"""
        self.input_wave_list = to_device(
            self.input_wave_list, device, non_blocking=non_blocking
        )
        self.target_wave_list = to_device(
            self.target_wave_list, device, non_blocking=non_blocking
        )
        self.noise_wave_list = to_device(
            self.noise_wave_list, device, non_blocking=non_blocking
        )
        self.lf0_list = to_device(self.lf0_list, device, non_blocking=non_blocking)
        self.t = to_device(self.t, device, non_blocking=non_blocking)
        self.r = to_device(self.r, device, non_blocking=non_blocking)
        self.speaker_id = to_device(self.speaker_id, device, non_blocking=non_blocking)
        return self


def collate_stack(values: list[Tensor]) -> Tensor:
    """Tensorのリストをスタックする"""
    return torch.stack(values)


def collate_dataset_output(data_list: list[OutputData]) -> BatchOutput:
    """DatasetOutputのリストをBatchOutputに変換"""
    if len(data_list) == 0:
        raise ValueError("batch is empty")

    return BatchOutput(
        input_wave_list=[d.input_wave for d in data_list],
        target_wave_list=[d.target_wave for d in data_list],
        noise_wave_list=[d.noise_wave for d in data_list],
        lf0_list=[d.lf0 for d in data_list],
        t=collate_stack([d.t for d in data_list]),
        r=collate_stack([d.r for d in data_list]),
        speaker_id=collate_stack([d.speaker_id for d in data_list]),
    )
