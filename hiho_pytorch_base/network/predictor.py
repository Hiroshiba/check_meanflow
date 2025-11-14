"""メインのネットワークモジュール"""

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from ..config import NetworkConfig
from .conformer.encoder import Encoder
from .transformer.utility import make_non_pad_mask


def get_lengths(
    tensor_list: list[Tensor],  # [(L, ?)]
) -> Tensor:  # (B,)
    """テンソルリストからlengthsを取得"""
    device = tensor_list[0].device
    lengths = torch.tensor([t.shape[0] for t in tensor_list], device=device)
    return lengths


def pad_tensor_list(
    tensor_list: list[Tensor],  # [(L, ?)]
) -> Tensor:  # (B, L, ?)
    """テンソルリストをパディング"""
    batch_size = len(tensor_list)
    if batch_size == 1:
        # NOTE: ONNX化の際にpad_sequenceがエラーになるため迂回
        padded = tensor_list[0].unsqueeze(0)
    else:
        padded = pad_sequence(tensor_list, batch_first=True)
    return padded


def create_padding_mask(
    lengths: Tensor,  # (B,)
) -> Tensor:  # (B, 1, L)
    """lengthsからパディングマスクを生成"""
    mask = make_non_pad_mask(lengths).unsqueeze(-2).to(lengths.device)
    return mask


class Predictor(nn.Module):
    """メインのネットワーク"""

    def __init__(
        self,
        hidden_size: int,
        speaker_size: int,
        speaker_embedding_size: int,
        encoder: Encoder,
    ):
        super().__init__()

        self.speaker_embedder = nn.Embedding(speaker_size, speaker_embedding_size)

        input_size = 1 + 1 + 1 + 1 + speaker_embedding_size
        self.pre_conformer = nn.Linear(input_size, hidden_size)
        self.encoder = encoder
        self.post = nn.Linear(hidden_size, 1)

    def forward(  # noqa: D102
        self,
        *,
        padded_wave: Tensor,  # (B, L, 1)
        padded_lf0: Tensor,  # (B, L, 1)
        mask: Tensor,  # (B, 1, L)
        t: Tensor,  # (B,)
        h: Tensor,  # (B,)
        speaker_id: Tensor,  # (B,)
    ) -> Tensor:  # (B, L, 1)
        batch_size = t.shape[0]
        speaker_embedding = self.speaker_embedder(speaker_id)

        max_length = padded_wave.size(1)
        speaker_expanded = speaker_embedding.unsqueeze(1).expand(
            batch_size, max_length, -1
        )

        t_expanded = t.unsqueeze(1).unsqueeze(2).expand(batch_size, max_length, 1)
        h_expanded = h.unsqueeze(1).unsqueeze(2).expand(batch_size, max_length, 1)

        combined = torch.cat(
            [padded_wave, padded_lf0, t_expanded, h_expanded, speaker_expanded],
            dim=2,
        )

        h = self.pre_conformer(combined)

        encoded, _ = self.encoder(x=h, cond=None, mask=mask)

        output = self.post(encoded)

        return output

    def forward_list(  # noqa: D102
        self,
        *,
        wave_list: list[Tensor],  # [(L, 1)]
        lf0_list: list[Tensor],  # [(L, 1)]
        t: Tensor,  # (B,)
        h: Tensor,  # (B,)
        speaker_id: Tensor,  # (B,)
    ) -> list[Tensor]:  # [(L,)]
        lengths = get_lengths(wave_list)
        padded_wave = pad_tensor_list(wave_list)
        padded_lf0 = pad_tensor_list(lf0_list)
        mask = create_padding_mask(lengths)

        output = self(
            padded_wave=padded_wave,
            padded_lf0=padded_lf0,
            mask=mask,
            t=t,
            h=h,
            speaker_id=speaker_id,
        )

        output_list = [output[i, :length, 0] for i, length in enumerate(lengths)]

        return output_list


def create_predictor(config: NetworkConfig) -> Predictor:
    """設定からPredictorを作成"""
    encoder = Encoder(
        hidden_size=config.hidden_size,
        condition_size=0,
        block_num=config.conformer_block_num,
        dropout_rate=config.conformer_dropout_rate,
        positional_dropout_rate=config.conformer_dropout_rate,
        attention_head_size=8,
        attention_dropout_rate=config.conformer_dropout_rate,
        use_macaron_style=True,
        use_conv_glu_module=True,
        conv_glu_module_kernel_size=31,
        feed_forward_hidden_size=config.hidden_size * 4,
        feed_forward_kernel_size=3,
    )
    return Predictor(
        hidden_size=config.hidden_size,
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
        encoder=encoder,
    )
