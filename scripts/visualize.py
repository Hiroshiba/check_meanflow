"""
機械学習データセットの可視化ツール

設定ファイルからDatasetCollectionを読み込み、データタイプごとにGradio UIで表示する。
各データタイプの表示形式（プロット、テーブル等）は機械学習タスクに応じてカスタマイズする。
データタイプに応じた可視化ロジックを_update_*_plotや_get_data_infoで調整する。
"""

import argparse
from dataclasses import dataclass
from typing import Any

import gradio as gr
import japanize_matplotlib  # noqa: F401 日本語フォントに必須
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from upath import UPath

from hiho_pytorch_base.config import Config
from hiho_pytorch_base.data.data import OutputData
from hiho_pytorch_base.dataset import (
    DatasetCollection,
    DatasetType,
    LazyInputData,
    create_dataset,
)


@dataclass
class DataInfo:
    """データ情報"""

    input_wave: np.ndarray
    target_wave: np.ndarray
    noise_wave: np.ndarray
    lf0: np.ndarray
    t: float
    r: float
    speaker_id: int
    details: str


@dataclass
class FigureState:
    """図の状態"""

    input_wave_fig: Figure | None = None
    target_wave_fig: Figure | None = None
    noise_wave_fig: Figure | None = None
    lf0_fig: Figure | None = None
    input_wave_line: Line2D | None = None
    target_wave_line: Line2D | None = None
    noise_wave_line: Line2D | None = None
    lf0_line: Line2D | None = None


class VisualizationApp:
    """可視化アプリケーション"""

    def __init__(self, config_path: UPath, initial_dataset_type: DatasetType):
        self.config_path = config_path
        self.initial_dataset_type = initial_dataset_type

        self.dataset_collection = self._create_dataset()
        self.figure_state = FigureState()

    def _create_dataset(self) -> DatasetCollection:
        """データセットを作成"""
        config = Config.from_dict(yaml.safe_load(self.config_path.read_text()))
        return create_dataset(config.dataset)

    def _get_output_data(self, index: int, dataset_type: DatasetType) -> OutputData:
        """前処理済みのOutputDataを取得"""
        dataset = self.dataset_collection.get(dataset_type)
        return dataset[index]

    def _get_lazy_data(self, index: int, dataset_type: DatasetType) -> LazyInputData:
        """遅延読み込み用のLazyInputDataを取得"""
        dataset = self.dataset_collection.get(dataset_type)
        return dataset.datas[index]

    def _create_details_text(
        self, output_data: OutputData, lazy_data: LazyInputData
    ) -> str:
        """詳細情報テキストを作成"""
        return f"""
設定ファイル: {self.config_path}

入力波形
shape: {tuple(output_data.input_wave.shape)}

ターゲット波形
shape: {tuple(output_data.target_wave.shape)}

ノイズ波形
shape: {tuple(output_data.noise_wave.shape)}

log F0
shape: {tuple(output_data.lf0.shape)}

時間パラメータ t: {output_data.t.item():.6f}
比率パラメータ r: {output_data.r.item():.6f}

話者ID: {output_data.speaker_id.item()}

データ生成パラメータ:
lf0範囲: [{lazy_data.lf0_low:.2f}, {lazy_data.lf0_high:.2f}]
サンプリング長範囲: [{lazy_data.min_sampling_length}, {lazy_data.max_sampling_length}]
"""

    def _setup_input_wave_plot(self, data: np.ndarray) -> Figure:
        if (
            self.figure_state.input_wave_fig is None
            or self.figure_state.input_wave_line is None
        ):
            self.figure_state.input_wave_fig, ax = plt.subplots(figsize=(10, 4))
            x_data = range(len(data))
            (self.figure_state.input_wave_line,) = ax.plot(x_data, data)
            ax.set_title("入力波形")
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Amplitude")
            ax.grid(True)
        else:
            x_data = range(len(data))
            self.figure_state.input_wave_line.set_data(x_data, data)
            ax = self.figure_state.input_wave_fig.gca()
            ax.relim()
            ax.autoscale_view()
            self.figure_state.input_wave_fig.canvas.draw()

        return self.figure_state.input_wave_fig

    def _setup_target_wave_plot(self, data: np.ndarray) -> Figure:
        if (
            self.figure_state.target_wave_fig is None
            or self.figure_state.target_wave_line is None
        ):
            self.figure_state.target_wave_fig, ax = plt.subplots(figsize=(10, 4))
            x_data = range(len(data))
            (self.figure_state.target_wave_line,) = ax.plot(x_data, data)
            ax.set_title("ターゲット波形")
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Amplitude")
            ax.grid(True)
        else:
            x_data = range(len(data))
            self.figure_state.target_wave_line.set_data(x_data, data)
            ax = self.figure_state.target_wave_fig.gca()
            ax.relim()
            ax.autoscale_view()
            self.figure_state.target_wave_fig.canvas.draw()

        return self.figure_state.target_wave_fig

    def _setup_noise_wave_plot(self, data: np.ndarray) -> Figure:
        if (
            self.figure_state.noise_wave_fig is None
            or self.figure_state.noise_wave_line is None
        ):
            self.figure_state.noise_wave_fig, ax = plt.subplots(figsize=(10, 4))
            x_data = range(len(data))
            (self.figure_state.noise_wave_line,) = ax.plot(x_data, data)
            ax.set_title("ノイズ波形")
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Amplitude")
            ax.grid(True)
        else:
            x_data = range(len(data))
            self.figure_state.noise_wave_line.set_data(x_data, data)
            ax = self.figure_state.noise_wave_fig.gca()
            ax.relim()
            ax.autoscale_view()
            self.figure_state.noise_wave_fig.canvas.draw()

        return self.figure_state.noise_wave_fig

    def _setup_lf0_plot(self, data: np.ndarray) -> Figure:
        if (
            self.figure_state.lf0_fig is None
            or self.figure_state.lf0_line is None
        ):
            self.figure_state.lf0_fig, ax = plt.subplots(figsize=(10, 4))
            x_data = range(len(data))
            (self.figure_state.lf0_line,) = ax.plot(x_data, data)
            ax.set_title("log F0")
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("log F0")
            ax.grid(True)
        else:
            x_data = range(len(data))
            self.figure_state.lf0_line.set_data(x_data, data)
            ax = self.figure_state.lf0_fig.gca()
            ax.relim()
            ax.autoscale_view()
            self.figure_state.lf0_fig.canvas.draw()

        return self.figure_state.lf0_fig

    def _setup_plots(
        self, output_data: OutputData
    ) -> tuple[Figure, Figure, Figure, Figure]:
        """プロットを作成または更新"""
        # データの取得と整形
        input_wave_data = output_data.input_wave.cpu().numpy().flatten()
        target_wave_data = output_data.target_wave.cpu().numpy().flatten()
        noise_wave_data = output_data.noise_wave.cpu().numpy().flatten()
        lf0_data = output_data.lf0.cpu().numpy().flatten()

        # figureの更新または作成
        input_wave_plot = self._setup_input_wave_plot(input_wave_data)
        target_wave_plot = self._setup_target_wave_plot(target_wave_data)
        noise_wave_plot = self._setup_noise_wave_plot(noise_wave_data)
        lf0_plot = self._setup_lf0_plot(lf0_data)

        return (input_wave_plot, target_wave_plot, noise_wave_plot, lf0_plot)

    def _create_data_info(
        self, output_data: OutputData, lazy_data: LazyInputData
    ) -> DataInfo:
        """データ情報を作成"""
        input_wave = output_data.input_wave.cpu().numpy()
        target_wave = output_data.target_wave.cpu().numpy()
        noise_wave = output_data.noise_wave.cpu().numpy()
        lf0 = output_data.lf0.cpu().numpy()
        t = float(output_data.t.item())
        r = float(output_data.r.item())
        speaker_id = int(output_data.speaker_id.item())
        details = self._create_details_text(output_data, lazy_data)

        return DataInfo(
            input_wave=input_wave,
            target_wave=target_wave,
            noise_wave=noise_wave,
            lf0=lf0,
            t=t,
            r=r,
            speaker_id=speaker_id,
            details=details,
        )

    def launch(self) -> None:
        """Gradio UIを起動"""
        initial_dataset = self.dataset_collection.get(self.initial_dataset_type)
        initial_max_index = len(initial_dataset) - 1

        with gr.Blocks() as demo:
            # 状態管理
            current_index = gr.State(0)
            current_dataset_type = gr.State(self.initial_dataset_type)

            # UI コンポーネント
            with gr.Row():
                dataset_type_dropdown = gr.Dropdown(
                    choices=list(DatasetType),
                    value=self.initial_dataset_type,
                    label="データセットタイプ",
                    scale=1,
                )
                index_slider = gr.Slider(
                    minimum=0,
                    maximum=initial_max_index,
                    value=0,
                    step=1,
                    label="サンプルインデックス",
                    scale=3,
                )

            @gr.render(inputs=[current_index, current_dataset_type])
            def render_content(index: int, dataset_type: DatasetType) -> None:
                output_data = self._get_output_data(index, dataset_type)
                lazy_data = self._get_lazy_data(index, dataset_type)

                (
                    input_wave_plot,
                    target_wave_plot,
                    noise_wave_plot,
                    lf0_plot,
                ) = self._setup_plots(output_data)
                data_info = self._create_data_info(output_data, lazy_data)

                with gr.Row():
                    gr.Textbox(
                        value=f"t = {data_info.t:.6f}",
                        label="時間パラメータ",
                        interactive=False,
                        scale=1,
                    )
                    gr.Textbox(
                        value=f"r = {data_info.r:.6f}",
                        label="比率パラメータ",
                        interactive=False,
                        scale=1,
                    )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 入力波形")
                        gr.Plot(value=input_wave_plot, label="input_wave")

                    with gr.Column():
                        gr.Markdown("### ターゲット波形")
                        gr.Plot(value=target_wave_plot, label="target_wave")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### ノイズ波形")
                        gr.Plot(value=noise_wave_plot, label="noise_wave")

                    with gr.Column():
                        gr.Markdown("### log F0")
                        gr.Plot(value=lf0_plot, label="lf0")

                with gr.Row():
                    gr.Textbox(
                        value=str(data_info.speaker_id),
                        label="話者ID",
                        interactive=False,
                    )

                gr.Markdown("---")
                gr.Textbox(
                    value=data_info.details,
                    label="詳細情報",
                    lines=20,
                    max_lines=30,
                    interactive=False,
                )

            # 状態変更によるUI同期
            def sync_slider_from_state(
                index: int, dataset_type: DatasetType
            ) -> tuple[int, Any]:
                dataset = self.dataset_collection.get(dataset_type)
                max_index = len(dataset) - 1

                return (
                    index,  # index_slider value
                    gr.update(maximum=max_index),  # index_slider max
                )

            current_index.change(
                sync_slider_from_state,
                inputs=[current_index, current_dataset_type],
                outputs=[index_slider, index_slider],
            )

            current_dataset_type.change(
                sync_slider_from_state,
                inputs=[current_index, current_dataset_type],
                outputs=[index_slider, index_slider],
            )

            # UI操作から状態への更新
            index_slider.change(
                lambda new_index: new_index,
                inputs=[index_slider],
                outputs=[current_index],
            )

            dataset_type_dropdown.change(
                lambda new_type: (0, new_type),
                inputs=[dataset_type_dropdown],
                outputs=[current_index, current_dataset_type],
            )

            # 初期化
            demo.load(
                lambda: (0, self.initial_dataset_type),
                outputs=[current_index, current_dataset_type],
            )

        demo.launch()


def visualize(config_path: UPath, dataset_type: DatasetType) -> None:
    """指定されたデータセットをGradio UIで可視化する"""
    app = VisualizationApp(config_path, dataset_type)
    app.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="データセットのビジュアライゼーション")
    parser.add_argument("config_path", type=UPath, help="設定ファイルのパス")
    parser.add_argument(
        "--dataset_type", type=DatasetType, required=True, help="データセットタイプ"
    )

    args = parser.parse_args()
    visualize(config_path=args.config_path, dataset_type=args.dataset_type)
