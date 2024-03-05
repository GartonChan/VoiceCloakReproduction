# Project: VoiceCloak Reproduction

## Directory Structure

We apologize for any disorganization in the file structure. Below is a detailed description of our project's directory, which should assist in evaluating our reproduction efforts.

**Platform:** Linux

- `/`
  - `README.md` - Provides a helpful description of our reproduction process.
  - `CVAE.py` - Implements the CVAE as described in the paper, including the model's definition and training process. This script requires the preprocessed dataset to run.
  - `data.py` - Data loader script for `CVAE.py`.
  - `cvae_weights.pth` - Contains the trained weights for our CVAE model. The dataset for training CVAE is large, so only the model weights are included.
  - `optimizer_with_STOI.py` - Executes the project on the test set, outputs metric results to `results.txt`, and saves the convolved audio files in their respective directories.
  - `optimizer_with_STOI_self.py` - Similar to `optimizer_with_STOI.py`, but for self-testing with our samples. Outputs results to `results_self.txt`.
  - `WAD.py` - Compares each processed audio file in `text-clean` with its original version in `processed-text-clean`, saving results in `WAD_results.txt`.
  - `WAD_self.py` - Compares each processed audio file in `selfSampling` with its original version, saving results in `WAD_results_self.txt`.
  - `/logAnalyse/` - Processes log data, generates figures using `matplotlib.pyplot`, and outputs statistical data.
    - `extractData.py` - Extracts useful data from log files (e.g., `results.txt` and `WAD_results.txt`) for statistical analysis.
    - `*.json` - Results of `extractData.py`.
    - `TestSetAnalysis.ipynb` - Analyzes data and plots figures to visualize the results.
    - `SelfSampleAnalysis.ipynb` - Analyzes data of self-samples and plots figures to visualize the results.
  - `/text-clean/` - Contains original audio from the test set.
  - `/afterRIR-text-clean/` - Contains audio files from the test set convolved with the RIR signal, without optimization.
  - `/processed-text-clean/` - Contains audio files from the test set processed by VoiceCloak (convolved with optimized delta).
  - `/selfSampling/` - Contains original self-recorded samples.
  - `/afterRIRSelfSampling/` - Contains self-recorded samples convolved with the RIR signal, without optimization.
  - `/processedSelfSampling/` - Contains self-recorded samples processed by VoiceCloak (convolved with optimized delta).
  - `*.txt` - Contains other necessary log files.

## Our Work

### 1. Implemented CVAE as per the appendix:

- Preprocessed the audio data to extract embeddings using mainstream ASR tools.
- Appended a one-hot label to the embeddings.
- Defined the model (layers, hyperparameters).
- Conducted training (initial settings, optimal hyperparameters).

### 2. Implemented the Optimizer based on the pseudo code:

- Selected an effective RIR as a template for delta.
- Optimized delta using the Gradient Descent Method.

### 3. Evaluation and Analysis:

- **Data Source:** (1) Test Set; (2) Self-samples.
- **Metrics (completed):** (1) DSR; (2) STOI; (3) WAD.
- Generated figures to display statistical information.

## How to Run Our Project

**Step 1:** Set up the required environment, including `packages` and the `data` necessary for execution.

**Step 2:** Execute `optimizer_with_STOI.py` to generate all audio files and calculate the `STOI`. Results are saved in `results.txt`.

**Step 3:** Run `WAD.py` to compare each processed audio file with its original version. Results are saved in `WAD_results.txt`.

**Step 4:** Use the `logAnalyse` script in `./logAnalyse` to obtain additional statistical information (including `DSR`, `STOI`, and `WAD`). Please ensure to `copy` the log files or modify the path as needed.

**NOTE:**

- We employ `os.getcwd()` combined with specific folder and file names. Ensure that your terminal's working directory is set to our project's root directory (`/`).
- Files with the `_self` suffix pertain to our self-samples for extended testing.
