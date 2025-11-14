"""学習済みモデルからの推論モジュール"""

from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

import numpy
import torch
from torch import Tensor, nn

from .config import Config
from .network.predictor import Predictor, create_predictor

TensorLike = Tensor | numpy.ndarray


@dataclass
class GeneratorOutput:
    """生成したデータ"""

    wave_list: list[Tensor]  # [(L,)]


def to_tensor(array: TensorLike, device: torch.device) -> Tensor:
    """データをTensorに変換する"""
    if not isinstance(array, Tensor | numpy.ndarray):
        array = numpy.asarray(array)
    if isinstance(array, numpy.ndarray):
        tensor = torch.from_numpy(array)
    else:
        tensor = array

    tensor = tensor.to(device)
    return tensor


class Generator(nn.Module):
    """生成経路で推論するクラス"""

    def __init__(
        self,
        config: Config,
        predictor: Predictor | Path,
        use_gpu: bool,
    ):
        super().__init__()

        self.config = config
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")

        if isinstance(predictor, Path):
            state_dict = torch.load(predictor, map_location=self.device)
            predictor = create_predictor(config.network)
            predictor.load_state_dict(state_dict)
        self.predictor = predictor.eval().to(self.device)

    @torch.no_grad()
    def forward(
        self,
        *,
        noise_wave_list: list[TensorLike],  # [(L, 1)]
        lf0_list: list[TensorLike],  # [(L, 1)]
        speaker_id: TensorLike,  # (B,)
        step_num: int,
    ) -> GeneratorOutput:
        """生成経路で推論する"""
        noise_wave_list_tensor = [
            to_tensor(item, self.device) for item in noise_wave_list
        ]
        lf0_list_tensor = [to_tensor(item, self.device) for item in lf0_list]
        speaker_id_tensor = to_tensor(speaker_id, self.device)

        if self.config.model.flow_type == "rectified_flow":
            return self._generate_rectified_flow(
                noise_wave_list_tensor,
                lf0_list_tensor,
                speaker_id_tensor,
                step_num,
            )
        elif self.config.model.flow_type == "meanflow":
            return self._generate_meanflow(
                noise_wave_list_tensor,
                lf0_list_tensor,
                speaker_id_tensor,
                step_num,
            )
        else:
            assert_never(self.config.model.flow_type)

    def _generate_rectified_flow(
        self,
        noise_wave_list: list[Tensor],  # [(L, 1)]
        lf0_list: list[Tensor],  # [(L, 1)]
        speaker_id: Tensor,  # (B,)
        step_num: int,
    ) -> GeneratorOutput:
        """RectifiedFlowで生成"""
        wave_list = [noise_wave.clone() for noise_wave in noise_wave_list]

        t_array = torch.linspace(0, 1, steps=step_num + 1, device=self.device)[:-1]
        delta_t_step = 1.0 / step_num

        for i in range(step_num):
            t = t_array[i].expand(len(wave_list))
            h = torch.zeros_like(t)

            velocity_list = self.predictor.forward_list(
                wave_list=wave_list,
                lf0_list=lf0_list,
                t=t,
                h=h,
                speaker_id=speaker_id,
            )

            for wave, velocity in zip(wave_list, velocity_list, strict=True):
                wave[:, 0] += velocity * delta_t_step

        output_wave_list = [wave[:, 0] for wave in wave_list]

        return GeneratorOutput(wave_list=output_wave_list)

    def _generate_meanflow(
        self,
        noise_wave_list: list[Tensor],  # [(L, 1)]
        lf0_list: list[Tensor],  # [(L, 1)]
        speaker_id: Tensor,  # (B,)
        step_num: int,
    ) -> GeneratorOutput:
        """MeanFlowで生成"""
        if step_num == 1:
            t = torch.ones(len(noise_wave_list), device=self.device)
            h = t

            average_velocity_list = self.predictor.forward_list(
                wave_list=noise_wave_list,
                lf0_list=lf0_list,
                t=t,
                h=h,
                speaker_id=speaker_id,
            )

            output_wave_list = []
            for noise_wave, avg_velocity in zip(
                noise_wave_list, average_velocity_list, strict=True
            ):
                output_wave = noise_wave[:, 0] - avg_velocity
                output_wave_list.append(output_wave)

            return GeneratorOutput(wave_list=output_wave_list)
        else:
            wave_list = [noise_wave.clone() for noise_wave in noise_wave_list]

            t_array = torch.linspace(1, 0, steps=step_num + 1, device=self.device)
            delta_t_step = 1.0 / step_num

            for i in range(step_num):
                t_start = t_array[i]
                t_end = t_array[i + 1]
                t = t_start.expand(len(wave_list))
                h = (t_start - t_end).expand(len(wave_list))

                average_velocity_list = self.predictor.forward_list(
                    wave_list=wave_list,
                    lf0_list=lf0_list,
                    t=t,
                    h=h,
                    speaker_id=speaker_id,
                )

                for wave, avg_velocity in zip(
                    wave_list, average_velocity_list, strict=True
                ):
                    wave[:, 0] -= avg_velocity * delta_t_step

            output_wave_list = [wave[:, 0] for wave in wave_list]

            return GeneratorOutput(wave_list=output_wave_list)
