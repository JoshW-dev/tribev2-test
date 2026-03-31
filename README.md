# TRIBE v2 Test

Test harness for Meta's [TRIBE v2](https://github.com/facebookresearch/tribev2) brain encoding model.

## Prerequisites

- Python 3.11+
- Hugging Face account with access to `meta-llama/Llama-3.2-3B` (gated model)

## Setup

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
