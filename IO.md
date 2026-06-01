# Skin Severity Scoring Pipeline: Input/Output (I/O) Specifications

This document outlines the strict tensor shapes, data types, value ranges, and architectural expectations for all inputs and outputs across the various components of the **Skin Severity Scoring Pipeline**.

---

## 📖 Table of Contents
1. [Top-Level Model (SkinSeverityModel)](#1-top-level-model-skinseveritymodel)
2. [Stage 1: Preprocessing & Crop (ROIExtractor)](#2-stage-1-preprocessing--crop-roiextractor)
3. [Stage 2: Shared Backbone (SharedBackbone)](#3-stage-2-shared-backbone-sharedbackbone)
4. [Stage 3: Individual Feature Gates](#4-stage-3-individual-feature-gates)
5. [Stage 4: Attention Fusion Layer (GatedFusion)](#5-stage-4-attention-fusion-layer-gatedfusion)

---

## 1. Top-Level Model (`SkinSeverityModel`)

* **Target File**: `skin_severity/model.py`
* **Entry Method**: `forward(self, image, metadata=None, location=None)`

### 📥 Inputs

The model consumes three primary inputs. Patient metadata and body location are optional and support robust graceful degradation if missing.

| Tensor Name | Shape | Data Type | Value Range / Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`image`** | `[B, 3, H, W]` | `torch.float32` | `[0.0, 1.0]` | The primary RGB dermatological image. |
| **`metadata`** | `[B, M]` *(Optional)*| `torch.float32` | Real numbers (usually standardized or normalized) | Patient surveys, symptom indicators (e.g. itch, pain, duration). |
| **`location`** | `[B]` *(Optional)* | `torch.int64` (Long) | `[-1, L-1]` where `L` is total location categories | Discrete anatomical body location index. `-1` denotes missing data. |

---

### 📤 Outputs

The `forward` pass returns a python dictionary containing the continuous predictions, categorical clinical bands, attention explanations, and latent representations.

```python
output_dict = model(image, metadata, location)
```

| Dict Key | Return Type | Shape | Value Range / Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`"severity_score"`** | `torch.Tensor` | `[B, 1]` | `[0.0, 3.0]` | The continuous severity score regressed by the network. |
| **`"severity_bands"`** | `list[str]` | Length: `B` | `['very_mild', 'mild', 'moderate', 'severe', 'critical']` | Human-interpretable categorical clinical severity band assigned per sample. |
| **`"gate_weights"`** | `torch.Tensor` | `[B, 8]` | `[0.0, 1.0]` (sums to `1.0` per sample) | Sample-dependent dynamic self-attention coefficients indicating how much each gate contributed. |
| **`"fused_embedding"`**| `torch.Tensor` | `[B, 128]` | Real numbers | The aggregated, unified multi-modal latent representation before score projection. |
| **`"explanations"`** | `dict[str, Tensor]`| 8 keys, each `[B, 1]` | `[0.0, 1.0]` | A helper mapping that binds gate names (e.g. `'color'`) directly to their attention coefficient tensor. |

---

## 2. Stage 1: Preprocessing & Crop (`ROIExtractor`)

* **Target File**: `skin_severity/roi_extractor.py`
* **Entry Method**: `forward(self, x)`

### 📥 Inputs
* **`x`**: `[B, 3, H, W]` (`torch.float32`, range `[0.0, 1.0]`) - The original full-sized RGB input image.

### 📤 Outputs
Returns a three-element tuple: `(roi_crop, mask, box_coords)`

| Output Element | Shape | Data Type | Value Range / Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`roi_crop`** | `[B, 3, 224, 224]` | `torch.float32` | `[0.0, 1.0]` | The bilinearly cropped and resized lesion region. |
| **`mask`** | `[B, 1, 224, 224]` | `torch.float32` | `[0.0, 1.0]` (soft probability) | Soft segmentor mask indicating lesion vs. healthy skin boundary. |
| **`box_coords`** | `[B, 4]` | `torch.float32` | `[0.0, 1.0]` (normalized) | Normalized coordinates $[x_{min}, y_{min}, x_{max}, y_{max}]$ of the lesion boundary. |

---

## 3. Stage 2: Shared Backbone (`SharedBackbone`)

* **Target File**: `skin_severity/backbone.py`
* **Entry Method**: `forward(self, x)`

### 📥 Inputs
* **`x`**: `[B, 3, 224, 224]` (`torch.float32`) - The processed `roi_crop` from the ROIExtractor.

### 📤 Outputs
Returns a three-element tuple containing multi-scale hierarchical feature maps: `(low, mid, high)`

| Output Element | Shape | Channels | Spatial Resolution | Clinical Representation |
| :--- | :--- | :--- | :--- | :--- |
| **`low`** | `[B, 64, 56, 56]` | 64 | $56 \times 56$ | Edges, local color gradients, and fine surface texture details. |
| **`mid`** | `[B, 128, 28, 28]` | 128 | $28 \times 28$ | Crustiness, scaling, and roughness features. |
| **`high`** | `[B, 256, 14, 14]`| 256 | $14 \times 14$ | Semantic structures, macro lesion shape, and tissue damage patterns. |

---

## 4. Stage 3: Individual Feature Gates

Every gate inherits from the base class `FeatureGate` ([base_gate.py](file:///d:/OneDrive/Documents/My_Proj/skin_severity/gates/base_gate.py)) and returns a tuple: `(embedding, score)`.
* **`embedding`**: `[B, 128]` (`torch.float32`) - The gate's unified multi-modal latent representation.
* **`score`**: `[B, 1]` (`torch.float32` in range `[0.0, 1.0]`) - The internal activation/confidence score computed using detached features.

### 1. `ColorGate`
* **Inputs**:
  - `roi_image`: `[B, 3, 224, 224]` (`torch.float32`)
  - `low`: `[B, 64, 56, 56]` (`torch.float32`)
  - `mid`: `[B, 128, 28, 28]` (`torch.float32`)

### 2. `TextureGate`
* **Inputs**:
  - `low`: `[B, 64, 56, 56]` (`torch.float32`)
  - `mid`: `[B, 128, 28, 28]` (`torch.float32`)

### 3. `ShapeBorderGate`
* **Inputs**:
  - `mid`: `[B, 128, 28, 28]` (`torch.float32`)
  - `high`: `[B, 256, 14, 14]` (`torch.float32`)
  - `mask`: `[B, 1, 224, 224]` (`torch.float32` - handles empty mask gracefully)
  - `box`: `[B, 4]` (`torch.float32`)

### 4. `MorphologyGate`
* **Inputs**:
  - `mid`: `[B, 128, 28, 28]` (`torch.float32`)
  - `high`: `[B, 256, 14, 14]` (`torch.float32`)

### 5. `DamageGate`
* **Inputs**:
  - `high`: `[B, 256, 14, 14]` (`torch.float32`)
  - `roi_image`: `[B, 3, 224, 224]` (`torch.float32`)

### 6. `AreaSpreadGate`
* **Inputs**:
  - `mid`: `[B, 128, 28, 28]` (`torch.float32`)
  - `mask`: `[B, 1, 224, 224]` (`torch.float32`)
  - `box`: `[B, 4]` (`torch.float32`)

### 7. `LocationRiskGate`
* **Inputs**:
  - `location`: `[B]` (`torch.int64`, supports discrete location indices where `-1` represents missing metadata)
  - `_batch_size`: `int` (passed to fallback if `location` tensor is `None`)

### 8. `MetadataGate`
* **Inputs**:
  - `metadata`: `[B, M]` (`torch.float32`, handles all-zeros or `None` gracefully)
  - `_batch_size`: `int` (passed to fallback if `metadata` is `None`)

---

## 5. Stage 4: Attention Fusion Layer (`GatedFusion`)

* **Target File**: `skin_severity/fusion.py`
* **Entry Method**: `forward(self, gate_embeddings, gate_scores)`

### 📥 Inputs

| Input Parameter | Data Type | Contains | Expected Shape per Item |
| :--- | :--- | :--- | :--- |
| **`gate_embeddings`**| `list[Tensor]` | 8 tensors of projected features | Each tensor is `[B, 128]` |
| **`gate_scores`** | `list[Tensor]` | 8 tensors of gate-level confidence | Each tensor is `[B, 1]` |

---

### 📤 Outputs

The fusion block produces a dictionary containing the final aggregated predictions:

| Dict Key | Return Type | Shape | Value Range / Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`"severity"`** | `torch.Tensor` | `[B, 1]` | `[0.0, 3.0]` | The final predicted continuous severity score. |
| **`"gate_weights"`** | `torch.Tensor` | `[B, 8]` | `[0.0, 1.0]` (sums to `1.0` per sample) | Sample-dependent dynamic self-attention coefficients indicating how much each gate contributed. |
| **`"fused"`** | `torch.Tensor` | `[B, 128]` | Real numbers | The aggregated, unified multi-modal latent representation. |
