# Neural Content Intelligence

**Predict how the human brain responds to video content — before you publish it.**

NCI uses [Meta's TRIBE v2](https://github.com/facebookresearch/tribev2) brain encoding model to predict neural activation patterns from video content entirely computationally. No lab, no subjects, no $150k studies — just a GPU and your video file.

<p align="center">
  <img src="paper/figures/comparison_dashboard.png" alt="NCI Comparison Dashboard — neural profiles across 5 video content types" width="100%">
</p>

---

## Why NCI?

The content industry is **reactive**: publish, measure, iterate. Traditional neuromarketing is **predictive** but costs $15,000–$150,000 per study and takes weeks. NCI bridges the gap:

| | A/B Testing | Platform Analytics | AI Scoring Tools | Neuromarketing | **NCI** |
|---|---|---|---|---|---|
| **Timing** | Post-publication | Post-publication | Pre-pub (metadata only) | Pre-publication | **Pre-publication** |
| **Evaluates actual video** | Indirectly | Indirectly | No | Yes | **Yes** |
| **Cost per evaluation** | Ad spend required | Free (post-hoc) | $10–50/mo | $15k–$150k/study | **~$0.10–$1/video** |
| **Requires audience** | Yes | Yes | No | Yes (20–40 subjects) | **No** |
| **Explains *why*** | No | Limited | No | Yes | **Yes** |

<p align="center">
  <img src="paper/figures/fig5_method_comparison.png" alt="Method comparison matrix" width="80%">
</p>

---

## Key Numbers

- **1,000+ hours** of fMRI training data from ~720 participants
- **20,484** cortical vertices predicted per timestep
- **~2 minutes** per 30–60s video on a standard GPU
- **7** interpretable neural engagement signals
- **5** composite engagement metrics

---

## The 7 Neural Engagement Signals

NCI maps raw voxel predictions onto the **Yeo 7-network brain parcellation** — a standard neuroscience atlas — to produce interpretable cognitive signals:

<p align="center">
  <img src="paper/figures/yeo_parcellation.png" alt="Yeo 7-network brain parcellation" width="50%">
</p>

| Network | Signal | What It Captures |
|---|---|---|
| Visual | **Visual Salience** | Scene complexity, motion, color contrast |
| Somatomotor | **Embodied Response** | Sensory/motor resonance, "feeling" the content |
| Dorsal Attention | **Sustained Attention** | Top-down focus, visual tracking |
| Ventral Attention | **Surprise Detection** | Novelty, salience breaks, hook moments |
| Limbic | **Emotional Resonance** | Affective processing, emotional engagement |
| Frontoparietal | **Decision Activation** | Critical thinking, CTA readiness |
| Default Mode | **Narrative Engagement** | Story immersion, self-referential processing |

---

## 5 Engagement Metrics

From the 7 network signals, NCI computes five composite metrics:

- **Attention Retention Score (ARS)** — Does the content sustain focus across its duration?
- **Emotional Impact Index (EII)** — How strongly does it engage emotional processing?
- **Hook Strength Score (HSS)** — How effectively do the first 1–3 seconds capture attention?
- **CTA Activation Score (CAS)** — When are viewers in a decision-making neural state?
- **Neural Engagement Score (NES)** — Single composite ranking across all dimensions

<p align="center">
  <img src="paper/figures/fig3_radar_neural_profile.png" alt="Radar neural engagement profile" width="55%">
</p>

---

## Proof of Concept: 5 Video Archetypes

We analyzed five diverse short-form videos to demonstrate that **different content formats engage qualitatively different brain systems** — not just different degrees of the same response, but entirely different neural architectures.

<p align="center">
  <img src="paper/figures/fig4a_comparative_radar_overlay.png" alt="Comparative radar overlay — all 5 videos" width="60%">
</p>

<p align="center">
  <img src="paper/figures/fig4b_comparative_radar_grid.png" alt="Comparative radar grid — individual profiles" width="80%">
</p>

### Results by Content Type

<table>
<tr>
<td align="center" width="20%"><img src="paper/figures/thumb_BusinessEdLeilaHarmozi.jpg" alt="Business Education" width="120"><br><b>Business Education</b><br><sub>Somatomotor + Default Mode dominant — voice delivery matters more than visuals</sub></td>
<td align="center" width="20%"><img src="paper/figures/thumb_ElonAI.jpg" alt="Tech/AI News" width="120"><br><b>Tech/AI News</b><br><sub>Elevated Frontoparietal — viewers in critical-thinking mode, prime for CTAs</sub></td>
<td align="center" width="20%"><img src="paper/figures/thumb_PerfumeUGCInterview.jpg" alt="UGC Product Review" width="120"><br><b>UGC Review</b><br><sub>Highest Somatomotor (27%) — embodied/sensory processing drives authenticity</sub></td>
<td align="center" width="20%"><img src="paper/figures/thumb_sanitaryPadProductDemo.jpg" alt="Product Demo" width="120"><br><b>Product Demo</b><br><sub>Visual + Dorsal Attention dominant — pure "show don't tell" engagement</sub></td>
<td align="center" width="20%"><img src="paper/figures/thumb_viralJapaneseIceCutter.jpg" alt="Viral Satisfying" width="120"><br><b>Viral/Satisfying</b><br><sub>Highest Ventral Attention (17%) — surprise peaks create addictive looping</sub></td>
</tr>
</table>

---

## How It Works

```
Video → TRIBE v2 Brain Encoding → 20,484 Voxel Predictions → Yeo 7-Network Mapping → Engagement Metrics
```

The temporal dynamics reveal *when* each brain network activates across the video timeline:

<p align="center">
  <img src="paper/figures/fig1_network_timecourse.png" alt="Network activation timecourse" width="80%">
</p>

Salience peaks in the Ventral Attention network identify **hook moments** — the frames that neurally "grab" viewers:

<p align="center">
  <img src="paper/figures/fig2_ventral_attention_peaks.png" alt="Ventral attention peaks — hook detection" width="80%">
</p>

Mean brain activation across the cortical surface:

<p align="center">
  <img src="paper/figures/mean_activation_map.png" alt="Mean cortical activation map" width="70%">
</p>

---

## Deep-Dive Analysis Examples

Each video gets a full neural analysis — network timecourses, activation percentages, hook detection, and engagement scoring:

<details>
<summary><b>Business Education (Leila Hormozi)</b></summary>
<p align="center"><img src="paper/figures/analysis_BusinessEdLeilaHarmozi.png" alt="Analysis — Business Education" width="100%"></p>
</details>

<details>
<summary><b>Tech/AI News (Elon AI Commentary)</b></summary>
<p align="center"><img src="paper/figures/analysis_ElonAI.png" alt="Analysis — Tech AI News" width="100%"></p>
</details>

<details>
<summary><b>UGC Product Review (Perfume Interview)</b></summary>
<p align="center"><img src="paper/figures/analysis_PerfumeUGCInterview.png" alt="Analysis — UGC Review" width="100%"></p>
</details>

<details>
<summary><b>Product Demo (Sanitary Pad)</b></summary>
<p align="center"><img src="paper/figures/analysis_sanitaryPadProductDemo.png" alt="Analysis — Product Demo" width="100%"></p>
</details>

<details>
<summary><b>Viral/Satisfying (Japanese Ice Cutter)</b></summary>
<p align="center"><img src="paper/figures/analysis_viralJapaneseIceCutter.png" alt="Analysis — Viral Satisfying" width="100%"></p>
</details>

---

## Setup

### Prerequisites

- Python 3.11+
- Hugging Face account with access to `meta-llama/Llama-3.2-3B` (gated model)

### Installation

```bash
chmod +x setup.sh
./setup.sh
```

Or manually:

```bash
git clone https://github.com/facebookresearch/tribev2.git
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e ./tribev2
huggingface-cli login
```

## Usage

```bash
source .venv/bin/activate

# Text input (easiest smoke test, no video decode needed)
python run_inference.py --text sample_prompt.txt

# Video input
python run_inference.py --video sample.mp4

# Audio input
python run_inference.py --audio sample.wav

# Custom output path
python run_inference.py --video sample.mp4 --output results.npy
```

Output is a `.npy` file with shape `(n_timesteps, n_vertices)` — predicted brain responses on the fsaverage5 cortical mesh, shifted 5s back for hemodynamic lag.

---

## Paper

The full research paper is available at [`paper/neural_content_intelligence.pdf`](paper/neural_content_intelligence.pdf):

> **Neural Content Intelligence: Using Brain Encoding Models to Predict Social Media Engagement Before Publication**
>
> *Josh W. — Independent Researcher*

---

## Limitations

NCI is a proof-of-concept. Key limitations include:
- No validation against real-world engagement metrics yet (the critical next step)
- Population-level predictions only (~720 training subjects)
- Training data from controlled lab settings (WEIRD populations)
- Current implementation is visual-only (audio support planned)

See the paper for full discussion.

---

## License

This project uses Meta's TRIBE v2 model. See the [TRIBE v2 repository](https://github.com/facebookresearch/tribev2) for model licensing terms.
