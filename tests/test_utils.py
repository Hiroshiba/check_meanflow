"""テストの便利モジュール"""

from pathlib import Path

import yaml
from upath import UPath

from hiho_pytorch_base.config import Config


def setup_data_and_config(base_config_path: Path, data_dir: UPath) -> Config:
    """テストデータをセットアップし、設定を作る"""
    with base_config_path.open() as f:
        config_dict = yaml.safe_load(f)

    config = Config.from_dict(config_dict)
    return config
