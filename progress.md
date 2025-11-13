# MeanFlow実装プロジェクト進捗

## 目標

このプロジェクトの目標は以下の通り：

1. **MeanFlowの理解**: MeanFlowの理論と実装を深く理解する
2. **差分の明確化**: 通常のRectifiedFlowとMeanFlowの実装上の差分を明確にする
3. **実装**: サイン波生成タスクでMeanFlowを実装する
4. **比較検証**: RectifiedFlowとMeanFlowの性能・挙動を比較し、MeanFlowの効果を検証する

### 参考資料

- **2505.13447v1.pdf**: MeanFlow元論文（references/2505.13447v1.pdf）

### 参考リポジトリ

- check_diffusion_sine: 以前実装したRectifiedFlow（比較基準）
- meanflow-jax: 公式JAX実装
- py-meanflow: 公式PyTorch実装
- MeanFlow-zhuyu: 再現報告つき実装
- MeanFlow-PyTorch: ImageNet訓練コード
- MeanFlow-haidog: 最小構成実装

---

## 2025-11-13

### 完了タスク

参考リポジトリのclone作業を完了しました。

#### cloneしたリポジトリ

以下のリポジトリを `references/` ディレクトリ配下にcloneしました：

1. **check_diffusion_sine** - 以前実装したRectifiedFlowの実装
   - パス: `references/check_diffusion_sine/`
   - 用途: 通常のRectifiedFlowとMeanFlowの差を確認する基準

2. **meanflow-jax** (Gsunshine/meanflow) - JAX実装・公式
   - パス: `references/meanflow-jax/`
   - 用途: 公式実装の参照

3. **py-meanflow** (Gsunshine/py-meanflow) - PyTorch実装・公式
   - パス: `references/py-meanflow/`
   - 用途: PyTorchでの公式実装の参照

4. **MeanFlow-zhuyu** (zhuyu-cs/MeanFlow) - PyTorch・再現報告つき
   - パス: `references/MeanFlow-zhuyu/`
   - 用途: 再現実験の確認

5. **MeanFlow-PyTorch** (HaoyiZhu/MeanFlow-PyTorch) - PyTorch・ImageNet訓練コード
   - パス: `references/MeanFlow-PyTorch/`
   - 用途: 大規模データセットでの実装例

6. **MeanFlow-haidog** (haidog-yaqub/MeanFlow) - PyTorch・最小構成
   - パス: `references/MeanFlow-haidog/`
   - 用途: 最小限の実装の理解

### 次のステップ

元論文と5つのMeanFlow実装を比較調査し、レポートをここにまとめる。
実装間で差異がある箇所を特定し、その内容を記録する。
タスク依存の実装は対象外。CFGは除く。
