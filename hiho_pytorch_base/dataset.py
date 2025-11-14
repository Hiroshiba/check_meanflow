"""データセットモジュール"""

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import assert_never

import numpy
from torch.utils.data import Dataset as BaseDataset

from .config import DatasetConfig
from .data.data import InputData, OutputData, preprocess


@dataclass
class LazyInputData:
    """遅延読み込み対応の入力データ構造"""

    lf0_low: float
    lf0_high: float
    min_sampling_length: int
    max_sampling_length: int
    speaker_id: int

    def fetch(self) -> InputData:
        """ファイルからデータを読み込んでInputDataを生成"""
        rng = numpy.random.default_rng()
        lf0 = rng.uniform(self.lf0_low, self.lf0_high)
        if self.min_sampling_length == self.max_sampling_length:
            sampling_length = self.max_sampling_length
        else:
            sampling_length = rng.integers(
                self.min_sampling_length, self.max_sampling_length
            )

        return InputData(
            lf0=float(lf0),
            sampling_length=int(sampling_length),
            speaker_id=self.speaker_id,
        )


def prefetch_datas(
    train_datas: list[LazyInputData],
    test_datas: list[LazyInputData],
    valid_datas: list[LazyInputData] | None,
    train_indices: list[int],
    train_batch_size: int,
    num_prefetch: int,
) -> None:
    """データセットを学習順序に従って前もって読み込む"""
    if num_prefetch <= 0:
        return

    prefetch_order: list[LazyInputData] = []
    prefetch_order += [train_datas[i] for i in train_indices[:train_batch_size]]
    prefetch_order += test_datas
    prefetch_order += [train_datas[i] for i in train_indices[train_batch_size:]]
    if valid_datas is not None:
        prefetch_order += valid_datas

    with ThreadPoolExecutor(max_workers=num_prefetch) as executor:
        for data in prefetch_order:
            executor.submit(data.fetch)


class Dataset(BaseDataset[OutputData]):
    """メインのデータセット"""

    def __init__(
        self,
        datas: list[LazyInputData],
        config: DatasetConfig,
        is_eval: bool,
    ):
        self.datas = datas
        self.config = config
        self.is_eval = is_eval

    def __len__(self):
        """データセットのサイズ"""
        return len(self.datas)

    def __getitem__(self, i: int) -> OutputData:
        """指定されたインデックスのデータを前処理して返す"""
        try:
            return preprocess(
                self.datas[i].fetch(),
                sampling_rate=self.config.sampling_rate,
                data_proportion=self.config.data_proportion,
                is_eval=self.is_eval,
            )
        except Exception as e:
            raise RuntimeError(
                f"データ処理に失敗しました: index={i} data={self.datas[i]}"
            ) from e


class DatasetType(str, Enum):
    """データセットタイプ"""

    TRAIN = "train"
    TEST = "test"
    EVAL = "eval"
    VALID = "valid"


@dataclass
class DatasetCollection:
    """データセットコレクション"""

    train: Dataset
    """重みの更新に用いる"""

    test: Dataset
    """trainと同じドメインでモデルの過適合確認に用いる"""

    eval: Dataset | None
    """testと同じデータを評価に用いる"""

    valid: Dataset | None
    """trainやtestと異なり、評価専用に用いる"""

    def get(self, type: DatasetType) -> Dataset:
        """指定されたタイプのデータセットを返す"""
        match type:
            case DatasetType.TRAIN:
                return self.train
            case DatasetType.TEST:
                return self.test
            case DatasetType.EVAL:
                if self.eval is None:
                    raise ValueError("evalデータセットが設定されていません")
                return self.eval
            case DatasetType.VALID:
                if self.valid is None:
                    raise ValueError("validデータセットが設定されていません")
                return self.valid
            case _:
                assert_never(type)


def create_dataset(config: DatasetConfig) -> DatasetCollection:
    """データセットを作成"""
    assert config.train_num is not None
    datas = [
        LazyInputData(
            lf0_low=config.lf0_low,
            lf0_high=config.lf0_high,
            min_sampling_length=config.min_sampling_length,
            max_sampling_length=config.max_sampling_length,
            speaker_id=0,
        )
        for _ in range(config.train_num + config.test_num)
    ]

    if config.seed is not None:
        random.Random(config.seed).shuffle(datas)

    tests, trains = datas[: config.test_num], datas[config.test_num :]
    if config.train_num is not None:
        trains = trains[: config.train_num]

    def _wrapper(datas: list[LazyInputData], is_eval: bool) -> Dataset:
        if is_eval:
            datas = datas * config.eval_times_num
        dataset = Dataset(datas=datas, config=config, is_eval=is_eval)
        return dataset

    return DatasetCollection(
        train=_wrapper(trains, is_eval=False),
        test=_wrapper(tests, is_eval=False),
        eval=(_wrapper(tests, is_eval=True) if config.eval_for_test else None),
        valid=None,
    )
