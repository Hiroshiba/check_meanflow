"""評価値計算モジュール"""

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor, nn
from torch.nn.functional import mse_loss

from .batch import BatchOutput
from .generator import Generator
from .utility.pytorch_utility import detach_cpu
from .utility.train_utility import DataNumProtocol


@dataclass
class EvaluatorOutput(DataNumProtocol):
    """評価値"""

    mse_loss: Tensor

    def detach_cpu(self) -> Self:
        """全てのTensorをdetachしてCPUに移動"""
        self.mse_loss = detach_cpu(self.mse_loss)
        return self


def calculate_value(output: EvaluatorOutput) -> Tensor:
    """評価値の良し悪しを計算する関数。高いほど良い。"""
    return -output.mse_loss


class Evaluator(nn.Module):
    """評価値を計算するクラス"""

    def __init__(self, generator: Generator, step_num: int):
        super().__init__()
        self.generator = generator
        self.step_num = step_num

    @torch.no_grad()
    def forward(self, batch: BatchOutput) -> EvaluatorOutput:
        """データをネットワークに入力して評価値を計算する"""
        output = self.generator(
            noise_wave_list=batch.noise_wave_list,
            lf0_list=batch.lf0_list,
            speaker_id=batch.speaker_id,
            step_num=self.step_num,
        )

        generated_wave = torch.cat(output.wave_list)
        target_wave = torch.cat([tw.squeeze(1) for tw in batch.target_wave_list])

        mse = mse_loss(generated_wave, target_wave)

        return EvaluatorOutput(
            mse_loss=mse,
            data_num=batch.data_num,
        )
