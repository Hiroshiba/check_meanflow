"""モデルのモジュール。ネットワークの出力から損失を計算する。"""

from dataclasses import dataclass
from typing import Self, assert_never

import torch
from torch import Tensor, nn
from torch.nn.functional import mse_loss

from .batch import BatchOutput
from .config import ModelConfig
from .network.predictor import (
    Predictor,
    create_padding_mask,
    get_lengths,
    pad_tensor_list,
)
from .utility.pytorch_utility import detach_cpu
from .utility.train_utility import DataNumProtocol


@dataclass
class ModelOutput(DataNumProtocol):
    """学習時のモデルの出力。損失と、イテレーション毎に計算したい値を含む"""

    loss: Tensor
    """逆伝播させる損失"""

    mse_loss: Tensor

    def detach_cpu(self) -> Self:
        """全てのTensorをdetachしてCPUに移動"""
        self.loss = detach_cpu(self.loss)
        self.mse_loss = detach_cpu(self.mse_loss)
        return self


class Model(nn.Module):
    """学習モデルクラス"""

    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(self, batch: BatchOutput) -> ModelOutput:
        """データをネットワークに入力して損失などを計算する"""
        if self.model_config.flow_type == "rectified_flow":
            return self._forward_rectified_flow(batch)
        elif self.model_config.flow_type == "meanflow":
            return self._forward_meanflow(batch)
        else:
            assert_never(self.model_config.flow_type)

    def _forward_rectified_flow(self, batch: BatchOutput) -> ModelOutput:
        """RectifiedFlowの損失を計算"""
        h = torch.zeros_like(batch.t)

        predicted_v_list = self.predictor.forward_list(
            wave_list=batch.input_wave_list,
            lf0_list=batch.lf0_list,
            t=batch.t,
            h=h,
            speaker_id=batch.speaker_id,
        )

        predicted_v = torch.cat(predicted_v_list)
        target_v = torch.cat(
            [
                (target_wave - noise_wave).squeeze(1)
                for target_wave, noise_wave in zip(
                    batch.target_wave_list, batch.noise_wave_list, strict=True
                )
            ]
        )

        mse = mse_loss(predicted_v, target_v)

        return ModelOutput(
            loss=mse,
            mse_loss=mse,
            data_num=batch.data_num,
        )

    def _forward_meanflow(self, batch: BatchOutput) -> ModelOutput:
        """MeanFlowの損失を計算"""
        lengths = get_lengths(batch.input_wave_list)  # (B,)
        mask = create_padding_mask(lengths)  # (B, 1, L)

        padded_input_wave = pad_tensor_list(batch.input_wave_list)  # (B, L, 1)
        padded_lf0 = pad_tensor_list(batch.lf0_list)  # (B, L, 1)
        padded_target_wave = pad_tensor_list(batch.target_wave_list)  # (B, L, 1)
        padded_noise_wave = pad_tensor_list(batch.noise_wave_list)  # (B, L, 1)
        padded_target_v = padded_noise_wave - padded_target_wave  # (B, L, 1)

        def u_func(wave: Tensor, t: Tensor, r: Tensor) -> Tensor:
            """JVP計算用のラッパー関数"""
            h = t - r
            return self.predictor(
                padded_wave=wave.unsqueeze(-1),
                padded_lf0=padded_lf0,
                mask=mask,
                t=t,
                h=h,
                speaker_id=batch.speaker_id,
            ).squeeze(-1)

        jvp_result: tuple[Tensor, Tensor] = torch.func.jvp(
            func=u_func,
            primals=(padded_input_wave.squeeze(-1), batch.t, batch.r),
            tangents=(
                padded_target_v.squeeze(-1),
                torch.ones_like(batch.t),
                torch.zeros_like(batch.r),
            ),
        )  # type: ignore
        u_pred, du_dt = jvp_result  # (B, L), (B, L)

        batch_size = batch.t.shape[0]
        max_length = padded_input_wave.size(1)
        h_expanded = (
            (batch.t - batch.r)
            .unsqueeze(1)
            .expand(batch_size, max_length)  # FIXME: .view()に変えられそう
        )  # (B, L)

        u_tgt = padded_target_v.squeeze(-1) - h_expanded * du_dt  # (B, L)
        mse_per_element = (u_pred - u_tgt.detach()) ** 2  # (B, L)

        mask_2d = mask.squeeze(1)  # (B, L)
        masked_mse = mse_per_element * mask_2d  # (B, L)
        mse = masked_mse.sum() / mask_2d.sum()

        loss_per_sample = masked_mse.sum(dim=1) / mask_2d.sum(dim=1)  # (B,)
        adp_wt = (
            loss_per_sample.detach() + self.model_config.adaptive_weighting_eps
        ) ** self.model_config.adaptive_weighting_p  # (B,)
        loss_per_sample = loss_per_sample / adp_wt  # (B,)

        loss = loss_per_sample.mean()

        return ModelOutput(
            loss=loss,
            mse_loss=mse,
            data_num=batch.data_num,
        )
