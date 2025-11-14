"""データ処理モジュール"""

from dataclasses import dataclass
from typing import Literal, assert_never

import numpy
import torch
from torch import Tensor


@dataclass
class InputData:
    """データ処理前のデータ構造"""

    lf0: float
    sampling_length: int
    speaker_id: int


@dataclass
class OutputData:
    """データ処理後のデータ構造"""

    input_wave: Tensor
    target_wave: Tensor
    noise_wave: Tensor
    lf0: Tensor
    t: Tensor
    r: Tensor
    h: Tensor
    speaker_id: Tensor


def generate_sin_wave(
    lf0: float,
    phase: float,
    length: int,
    sampling_rate: float,
) -> numpy.ndarray:
    """サイン波を生成"""
    f0 = numpy.exp(lf0)
    wave = numpy.sin(
        2 * numpy.pi * f0 * numpy.arange(length) / sampling_rate + 2 * numpy.pi * phase
    )
    return wave


def sigmoid(a: float | numpy.ndarray) -> float | numpy.ndarray:
    """シグモイド関数"""
    return 1 / (1 + numpy.exp(-a))


def sample_time_meanflow(data_proportion: float) -> tuple[float, float]:
    """MeanFlow用の時間サンプリング (t, r)"""
    rng = numpy.random.default_rng()
    t_sample = float(sigmoid(rng.standard_normal() * 1.0 + (-0.4)))
    r_sample = float(sigmoid(rng.standard_normal() * 1.0 + (-0.4)))

    t = max(t_sample, r_sample)
    r = min(t_sample, r_sample)

    if rng.random() < data_proportion:
        r = t

    return t, r


def preprocess(
    d: InputData,
    *,
    sampling_rate: float,
    data_proportion: float,
    flow_type: Literal["rectified_flow", "meanflow"],
    is_eval: bool,
) -> OutputData:
    """データ処理"""
    rng = numpy.random.default_rng()
    target_wave = generate_sin_wave(
        lf0=d.lf0,
        phase=rng.random(),
        length=d.sampling_length,
        sampling_rate=sampling_rate,
    ).reshape(-1, 1)
    target_wave *= numpy.sqrt(2)

    if is_eval:
        t, r = 1.0, 0.0
    else:
        match flow_type:
            case "meanflow":
                t, r = sample_time_meanflow(data_proportion=data_proportion)
            case "rectified_flow":
                t = float(sigmoid(rng.standard_normal() * 1.0 + (-0.4)))
                r = 0.0
            case _:
                assert_never(flow_type)

    h = t - r

    noise_wave = rng.standard_normal(target_wave.shape)
    input_wave = noise_wave + t * (target_wave - noise_wave)

    lf0_array = numpy.full((d.sampling_length, 1), d.lf0, dtype=numpy.float32)

    return OutputData(
        input_wave=torch.from_numpy(input_wave.astype(numpy.float32)),
        target_wave=torch.from_numpy(target_wave.astype(numpy.float32)),
        noise_wave=torch.from_numpy(noise_wave.astype(numpy.float32)),
        lf0=torch.from_numpy(lf0_array),
        t=torch.tensor(t, dtype=torch.float32),
        r=torch.tensor(r, dtype=torch.float32),
        h=torch.tensor(h, dtype=torch.float32),
        speaker_id=torch.tensor(d.speaker_id, dtype=torch.long),
    )
