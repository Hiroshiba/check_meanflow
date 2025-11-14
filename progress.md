# MeanFlow実装プロジェクト進捗

## 目標

このプロジェクトの目標は以下の通り：

1. **MeanFlowの理解**: MeanFlowの理論と実装を深く理解する
2. **差分の明確化**: 通常のRectifiedFlowとMeanFlowの実装上の差分を明確にする
3. **実装**: サイン波生成タスクでMeanFlowを実装する
4. **比較検証**: RectifiedFlowとMeanFlowの性能・挙動を比較し、MeanFlowの効果を検証する

### 参考資料

- **2505.13447v1.pdf**: MeanFlow元論文（./references/2505.13447v1.pdf）
- **2209.03003v1.pdf**: RectifiedFlow元論文（./references/2209.03003v1.pdf）

### 参考リポジトリ

- ./references/check_diffusion_sine: 以前実装したRectifiedFlow（比較基準）
- ./references/meanflow-jax: 公式JAX実装
- ./references/py-meanflow: 公式PyTorch実装
- ./references/MeanFlow-zhuyu: 再現報告つき実装
- ./references/MeanFlow-PyTorch: ImageNet訓練コード
- ./references/MeanFlow-haidog: 最小構成実装

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
実装間で元論文と不整合がある怪しい差異がある箇所を特定し、その内容を記録する。
タスク依存の実装は対象外。CFGは除く。

---

## 2025-11-14

### 完了タスク

元論文と5つのMeanFlow実装を徹底的に調査し、理論と実装の差異を明確化しました。

### MeanFlow理論概要

#### 核心的な概念

MeanFlowは **平均速度（average velocity）** を導入することで、従来のFlow Matchingの **瞬間速度（instantaneous velocity）** とは異なるアプローチを取ります。

#### 数学的定義

**平均速度の定義（Eq. 3）:**
```
u(z_t, r, t) = 1/(t-r) ∫[r to t] v(z_τ, τ)dτ
```

**MeanFlow Identity（Eq. 6）- 最重要:**
```
u(z_t, r, t) = v(z_t, t) - (t-r) * d/dt u(z_t, r, t)
```

**時間微分の展開（Eq. 8）- JVPで計算:**
```
d/dt u(z_t, r, t) = v(z_t, t) * ∂_z u + ∂_t u
```
これはJacobian-vector product (JVP)として計算: `[∂_z u, ∂_r u, ∂_t u]` と `[v, 0, 1]`

**学習目標（Eq. 11）:**
```
u_tgt = v_t - (t-r) * (v_t * ∂_z u_θ + ∂_t u_θ)
loss = ||u_θ(z_t, r, t) - stopgrad(u_tgt)||^2
```

#### RectifiedFlowとの主な違い

1. **2つの時間変数**: `(r, t)` を使用（RectifiedFlowは `t` のみ）
2. **JVP計算**: 時間微分項を計算するためにJVPが必須
3. **r=tの割合制御**: `data_proportion` パラメータで制御（通常75%で `r=t`）
4. **1-step生成**: `z_0 = z_1 - u(z_1, 0, 1)` で直接サンプリング可能

### 各実装の分析

#### A. meanflow-jax（公式JAX実装）

**ファイル:** `references/meanflow-jax/meanflow.py`

**特徴:**
- 論文著者による実装、最も信頼性が高い
- logit_normal(-0.4, 1.0) による時間サンプリング
- `data_proportion=0.75` で75%が `r=t`
- κパラメータ実装済み（改善版CFG）
- Adaptive weighting: `p=1.0`, `eps=0.01`
- 時間変数形式: `(t, h)` where `h = t - r`

**核心コード:** Lines 206-256

#### B. py-meanflow（公式PyTorch実装）

**ファイル:** `references/py-meanflow/meanflow/models/meanflow.py`

**特徴:**
- CIFAR-10に特化した実装
- **重要**: `torch.amp.autocast("cuda", enabled=False)` でJVP時にautocastを明示的に無効化
- READMEで「JVPはFlash Attentionと互換性がない」と明記
- `ratio_r_not_equal_t` パラメータで制御（デフォルト0.25 = 75%が `r=t`）
- 時間変数形式: `(t, h)` をタプルで渡す

**核心コード:** Lines 36-79

#### C. MeanFlow-zhuyu（再現報告つき実装）

**ファイル:** `references/MeanFlow-zhuyu/loss.py`

**特徴:**
- 詳細な再現実験報告あり
- CFG実装が最も複雑（バッチ分割、時間範囲制御）
- `cfg_min_t`～`cfg_max_t` で適用範囲を制御
- **注意**: `meanflow_sampler.py` で推論時に明示的CFGを適用（論文と矛盾の可能性）
- 時間変数形式: `(r, t)` を直接渡す

**核心コード:** Lines 84-227

#### D. MeanFlow-PyTorch（ImageNet訓練コード）

**ファイル:** `references/MeanFlow-PyTorch/meanflow.py`

**特徴:**
- JAX公式実装に最も忠実
- κパラメータ実装済み
- `torch.func.jvp` または `torch.autograd.functional.jvp` を選択可能
- CFGガイダンスの実装が詳細
- 時間変数形式: `(t, h)` where `h = t - r`

**核心コード:** Lines 125-188

#### E. MeanFlow-haidog（最小構成実装）

**ファイル:** `references/MeanFlow-haidog/meanflow.py`

**特徴:**
- 最もシンプルな実装、学習目的に最適
- **注意**: `flow_ratio=0.50`（50%が `r=t`）は論文推奨値（75%）と異なる
- CFGは基本版のみ（κパラメータなし）
- Adaptive weighting: `p=0.5`（他は `p=1.0`）
- 時間変数形式: `(t, r)` を直接渡す

**核心コード:** Lines 127-179

### 実装間の主な差異

#### 1. 時間サンプリング（r=t割合）

| 実装 | r=tの割合 | 論文推奨との差 |
|------|----------|--------------|
| JAX公式 | 75% | ✓ 一致 |
| PyTorch公式 | 75% | ✓ 一致 |
| zhuyu | 75% | ✓ 一致 |
| PyTorch(HaoyiZhu) | 75% | ✓ 一致 |
| **haidog** | **50%** | **⚠ 不一致** |

#### 2. JVP実装とautocast

| 実装 | autocast無効化 | 備考 |
|------|--------------|------|
| JAX公式 | N/A | JAXには不要 |
| **PyTorch公式** | **✓ 明示的に無効化** | Flash Attention互換性問題を回避 |
| zhuyu | 記述なし | 潜在的な問題の可能性 |
| PyTorch(HaoyiZhu) | 記述なし | 潜在的な問題の可能性 |
| haidog | 記述なし | 潜在的な問題の可能性 |

#### 3. ネットワーク入力形式

| 実装 | 時間変数形式 | 論文Table 1c |
|------|------------|-------------|
| JAX公式 | `(t, h)` where `h = t - r` | 最良（FID 61.06） |
| PyTorch公式 | `(t, h)` タプル | 最良 |
| zhuyu | `(r, t)` | ほぼ同等（FID 61.75） |
| PyTorch(HaoyiZhu) | `(t, h)` where `h = t - r` | 最良 |
| haidog | `(t, r)` | ほぼ同等 |

#### 4. CFG実装

| 実装 | κパラメータ | 時間範囲制御 | 推論時CFG |
|------|-----------|------------|----------|
| JAX公式 | ✓ | ✓ | 不要（学習時統合済み） |
| PyTorch公式 | - | - | - |
| zhuyu | ✓ | ✓（最も詳細） | **⚠ 明示的適用（論文と矛盾）** |
| PyTorch(HaoyiZhu) | ✓ | ✓ | 不要 |
| haidog | - | - | 不要 |

#### 5. Adaptive Weighting

| 実装 | `p` 値 | `eps` 値 | 論文推奨との差 |
|------|-------|---------|--------------|
| JAX公式 | 1.0 | 0.01 | ✓ 最良 |
| PyTorch公式 | 1.0 | 0.01 | ✓ 最良 |
| zhuyu | 1.0 | 0.001 | ほぼ一致 |
| PyTorch(HaoyiZhu) | 1.0 | 0.01 | ✓ 最良 |
| **haidog** | **0.5** | - | **競争力あり（FID 63.98）** |

### 注意すべき実装上の違い

#### 🚨 クリティカルな差異

**1. zhuyuの推論時CFG（最重要）**
- 学習時CFG統合と推論時CFGの両方を実装している
- `meanflow_sampler.py` で推論時に明示的CFGを適用（実質2-NFE相当）
- 論文の主張（1-NFE with CFG）と矛盾する可能性
- ただし、READMEでは「cfg-scale must be set to 1.0 during inference」と記載あり

**2. JVPのautocast扱い**
- py-meanflowのみが `torch.amp.autocast("cuda", enabled=False)` で明示的に無効化
- READMEで「JVPはFlash Attentionと互換性がない」と警告
- 他の実装では `torch.compile` やFlash Attention使用時に問題が起こる可能性
- **PyTorch実装時には必ず考慮すべき**

**3. haidogのr=t割合（50% vs 75%）**
- 論文の最適値（75%が `r=t`）から大きく外れている
- MNIST/CIFAR-10では問題ないかもしれないが、ImageNetでは性能が落ちる可能性
- 論文Table 1aで検証済み

#### ⚠️ 中程度の差異

**4. ネットワーク入力の時間変数形式**
- `(t, r)` vs `(t, h)` where `h = t - r`
- 論文では両方機能するが、`(t, h)` が最良（FID: 61.06 vs 61.75）
- JVP計算時の微妙な違いがある可能性

**5. Adaptive weightingのデフォルト値**
- haidogのみ `p=0.5`、他は全て `p=1.0`
- 論文Table 1eでは `p=1.0` が最良だが、`p=0.5` も競争力あり

### 推奨実装とベストプラクティス

#### 最も信頼できる実装

1. **JAX公式（meanflow-jax）**: 論文著者による実装、最も信頼性が高い
2. **PyTorch(HaoyiZhu)**: JAX実装に最も忠実、κパラメータ実装済み
3. **py-meanflow**: CIFAR-10に特化、autocast処理が適切

#### 実装時のベストプラクティス

1. **時間サンプリング**: logit_normal(-0.4, 1.0) with **75%を `r=t`**
2. **JVP**: `torch.amp.autocast("cuda", enabled=False)` で**autocastを無効化**
3. **CFG**: 学習時統合（κパラメータ使用）、推論時は1-NFE維持
4. **Adaptive weighting**: `p=1.0`, `eps=0.01`
5. **時間変数**: `(t, t-r)` 形式がベスト
6. **ネットワーク**: `model(z, t, h, y)` where `h = t - r`

### 次のステップ

`references/check_diffusion_sine` のRectifiedFlow実装を比較し、差があるか確認し、レポートをここにまとめる。
元論文と不整合がある怪しい差異がある箇所を特定し、その内容を記録する。
今回はMeanFlow実装との比較は一切不要。

---

### 完了タスク

`references/check_diffusion_sine` のRectifiedFlow実装を元論文（2209.03003v1.pdf）と比較調査しました。

#### ファイル構成

**主要ファイル:**
- `check_diffusion_sine/model.py` - 損失計算とモデル定義
- `check_diffusion_sine/network/predictor.py` - Conformerベースの速度予測ネットワーク
- `check_diffusion_sine/generator.py` - ODEサンプリング（推論）
- `check_diffusion_sine/dataset.py` - データ生成・前処理
- `train.py` - 学習ループ

#### 元論文との整合性チェック

##### ✅ 一致している点

| 項目 | 説明 |
|------|------|
| **基本アルゴリズム** | RectifiedFlowのアルゴリズム1に準拠 |
| **Linear interpolation** | z_t = z_0 + t(z_1 - z_0) を正しく実装 |
| **速度の定義** | v = z_1 - z_0 |
| **損失関数** | \|\|v_θ(z_t, t) - v\|\|^2 のMSE loss |
| **時間サンプリング** | logit-normal分布（`t = sigmoid(randn())`） |
| **ODEソルバー** | Euler法を使用 |

##### 📝 元論文と異なる点（許容範囲）

| 項目 | 論文 | 実装 | 影響 |
|------|------|------|------|
| **ネットワーク** | U-Net/Transformer | Conformer + Postnet | 音声向け設計、問題なし |
| **時間埋め込み** | 明示的記載なし | 直接特徴量結合 | シンプル、問題なし |
| **条件付け** | 条件yの詳細なし | lf0（対数基本周波数） | タスク依存、問題なし |
| **Reflow** | 提案されている | 未実装 | 性能向上の機会損失 |

##### 📝 その他の実装の工夫

- **分散調整**: sin波を√2倍して分散を1に調整（`dataset.py:70`）
  - sin波の分散は0.5なので√2倍して1に正規化
  - 論文に記載なし、データ正規化の実装上の工夫

##### ⚠️ 非標準的な実装

- **ODEサンプリングでt=1.0を評価**（`generator.py:56, 66-73`）
  - `torch.linspace(0, 1, steps=N)`で最終点t=1を含む
  - 標準的な左端点Euler法では終点を評価しない
  - RectifiedFlowでは速度が定数のため実用上の影響は小さい

#### 実装の特徴

- logit-normal時間サンプリングを正しく実装
- 基本的なRectifiedFlowアルゴリズムを理解した実装
- 音声向けの工夫（Conformer、lf0条件付け）
- Reflowは未実装（論文で提案されている性能向上手法）

#### 結論

この実装はRectifiedFlowの基本アルゴリズムを正しく理解し、実装しています。

### 次のステップ

MeanFlow実装をサイン波生成タスクで開始する。
JAX公式実装（meanflow-jax）を参照しながらゼロから実装し、RectifiedFlowとの切り替えを可能にする。
check_diffusion_sineを参考にしつつ、現在のリポジトリのテンプレート構造（hiho_pytorch_base）に合わせて実装し、Conformerネットワークを使用する。
MeanFlowは公式実装に完全準拠し、`data_proportion=0.75`、時間変数形式`(t, h)`、adaptive weighting `p=1.0, eps=0.01`を適用する。
学習モード切り替え（MeanFlow/RectifiedFlow）により同一条件での比較評価を可能にし、生成品質の定量的比較指標を策定する。
