# CUT&Tag Agent

An AI-powered epigenomic pipeline assistant that converts the CUT&Tag data processing
tutorial into an interactive agentic workflow. Users provide sample metadata and
background information; the agent automatically generates a customised peak-calling
pipeline and supports natural-language conversation for result interpretation, QC
debugging, and live pipeline adjustment — no programming knowledge required. 

Demo video: [Watch here](https://drive.google.com/file/d/19rU1KYLlXunBvZUn90XhOp6QbaWI0UFG/view?usp=sharing)

## Motivation

CUT&Tag (Cleavage Under Targets and Tagmentation) is a powerful technique for profiling
histone modifications and transcription factor binding across the genome. However, the
standard data processing tutorial requires researchers to:

- Manually write shell scripts for every sample
- Choose between peak callers (MACS2 vs SEACR) and understand their parameters
- Debug QC failures by reading documentation and forum posts
- Re-run analysis whenever parameters need adjustment

This project addresses these barriers by wrapping the peak-calling step in an
**agentic workflow**:

1. Users upload a simple CSV describing their samples — no code required
2. The agent reads the metadata and automatically generates a customised pipeline
3. Users can ask questions in plain English: *"Why did you choose SEACR?"*,
   *"My peak counts are very low, what is wrong?"*, *"Switch to hg19"*
4. The agent reasons about the question and either explains, debugs, or adjusts the
   pipeline autonomously

The knowledge base is seeded from CUT&Tag protocol documentation and tutorial comments,
mirroring the NotebookLM-style design where domain knowledge is embedded into the
agent context via RAG retrieval.



## Agent Engineering Design

| Component | Technology | Role |
|---|---|---|
| **LLM backbone** | Local Ollama model (OpenAI-compatible API) | Reasoning, interpretation, decision-making — fully offline |
| **Agentic output** | Structured JSON output (replaces Function Calling) | LLM autonomously decides whether and how to modify the pipeline |
| **RAG** | Keyword-overlap retrieval over 12 KB entries | Grounds answers in CUT&Tag protocol knowledge |
| **Memory** | Sliding-window conversation history (last 10 turns) | Enables cross-turn debugging and follow-up questions |
| **Pipeline state** | PipelinePlan dataclass serialised to JSON | Single source of truth; every agent action mutates it |
| **UI** | Gradio Blocks | No-code web interface accessible from any browser |

### Agentic Loop

```
User message
     |
     v
RAG retrieval
(keyword search over KB -> top-3 entries injected into system prompt)
     |
     v
Local LLM receives:
  system prompt + pipeline state JSON + conversation history + user message
     |
     v
LLM outputs structured JSON:
  {"action": "none|adjust_peak_caller|change_genome|change_mark_type|add_debug_note",
   "params": {...},
   "message": "reply to user"}
     |
     v
Action executor mutates PipelinePlan (if action is not none)
     |
     v
UI updates: chat reply + pipeline state panel + regenerated shell commands
```

### Available Agent Actions

| Action | Triggered when |
|---|---|
| `adjust_peak_caller` | User explicitly requests a different caller |
| `change_genome` | User specifies a different reference genome |
| `change_mark_type` | User clarifies broad vs sharp mark assumption |
| `add_debug_note` | User reports a QC problem; agent records diagnosis in the log |
| `none` | User asks a conceptual or interpretive question |



## Focus Step: Peak Calling

This prototype focuses on the **peak calling** step of the CUT&Tag workflow, which is
the most consequential and error-prone step:

- Choosing narrow vs broad mode incorrectly produces near-zero peaks
- A genome build mismatch silently corrupts all downstream analysis
- Threshold settings directly control the false-positive rate

The full CUT&Tag workflow also includes alignment, QC, signal track generation, and
differential enrichment analysis. These steps follow the same agentic design pattern
and can be added as additional pipeline modules.



## Requirements

- Python 3.9 or later
- [Ollama](https://ollama.ai) installed and running
- A GPU with enough VRAM to run your chosen model (e.g. 20 GB VRAM for a 20B model)
- No internet connection required after initial setup



## Quick Start

### Step 1 — Install Ollama

Download from [https://ollama.ai](https://ollama.ai) and install for your OS.

### Step 2 — Check available models

```bash
ollama list
```

This shows all models already loaded. The model name format is `name:tag`,
for example `llama3.1:latest` or `gpt-oss:20b`.

If you have a model file on disk not yet registered with Ollama:

```bash
ollama create my-model-name -f /path/to/Modelfile
```

### Step 3 — Start the Ollama server

```bash
ollama serve
```

Keep this terminal open. You should see log output confirming the server is running
on port 11434.

### Step 4 — Install Python dependencies

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
openai>=1.0.0
gradio>=4.0.0
```

The `openai` library is used only to call Ollama's local OpenAI-compatible endpoint.
No OpenAI account or API key is required.

### Step 5 — Run the application

```bash
python cuttag_agent.py
```

Open your browser at `http://localhost:7860`.



## File Structure

```
cuttag_agent_local/
├── cuttag_agent.py       # Main application (Gradio UI + agent logic)
├── requirements.txt      # Python dependencies
├── metadata_demo.csv     # Example sample metadata (4 H3K27me3 samples)
├── kb_demo.json          # Additional knowledge base entries (optional)
└── README.md             # This file
```



## Input Format

### Sample Metadata CSV

Upload a CSV file with the following columns:

| Column | Required | Description | Example |
|---|---|---|---|
| `sample_id` | Yes | Unique sample identifier | `K27me3_ctrl_rep1` |
| `condition` | Yes | Experimental condition | `ctrl` or `treat` |
| `replicate` | Yes | Replicate number | `1` or `2` |
| `target` | Yes | Histone mark or TF target | `H3K27me3`, `CTCF` |
| `mark_type` | Yes | `broad`, `sharp`, or `auto` | `broad` |
| `genome` | Yes | Reference genome build | `hg38`, `hg19`, `mm10`, `mm39` |
| `bam_path` | Yes | Path to aligned BAM file | `data/sample.bam` |
| `control_bam` | No | Path to IgG control BAM | `data/IgG.bam` |

Example (`metadata_demo.csv`):

```csv
sample_id,condition,replicate,target,mark_type,genome,bam_path,control_bam
K27me3_ctrl_rep1,ctrl,1,H3K27me3,broad,hg38,data/K27me3_ctrl_rep1.bam,
K27me3_ctrl_rep2,ctrl,2,H3K27me3,broad,hg38,data/K27me3_ctrl_rep2.bam,
K27me3_treat_rep1,treat,1,H3K27me3,broad,hg38,data/K27me3_treat_rep1.bam,
K27me3_treat_rep2,treat,2,H3K27me3,broad,hg38,data/K27me3_treat_rep2.bam,
```

**Mark type auto-inference:** If `mark_type` is set to `auto`, the agent infers
broad or sharp from the target name:

- Broad marks: H3K27me3, H3K36me3, H3K9me3, H3K9me2
- Sharp marks: H3K4me3, H3K27ac, H3K4me1, CTCF, H3K9ac



## Output

### Generated Shell Commands

For each sample, the agent produces a ready-to-run shell command:

**MACS2 narrow** — sharp marks with or without control:
```bash
mkdir -p results/SAMPLE/peaks && \
  macs2 callpeak -t sample.bam -c control.bam \
  -f BAMPE -g hs -n SAMPLE --outdir results/SAMPLE/peaks -q 0.01
```

**MACS2 broad** — broad marks with control:
```bash
mkdir -p results/SAMPLE/peaks && \
  macs2 callpeak -t sample.bam -c control.bam \
  -f BAMPE -g hs -n SAMPLE --outdir results/SAMPLE/peaks \
  --broad --broad-cutoff 0.1
```

**SEACR relaxed** — broad marks without control:
```bash
mkdir -p results/SAMPLE/peaks && \
  bedtools genomecov -bg -ibam sample.bam > \
    results/SAMPLE/peaks/signal.bedgraph && \
  SEACR_1.3.sh results/SAMPLE/peaks/signal.bedgraph \
    0.01 non stringent results/SAMPLE/peaks/SAMPLE_seacr
```

### Output Files (produced after running the commands)

| File extension | Description |
|---|---|
| `*.narrowPeak` | Narrow peak calls from MACS2 |
| `*.broadPeak` | Broad peak calls from MACS2 |
| `*_seacr.stringent.bed` | Peak calls from SEACR |
| `signal.bedgraph` | Genome-wide signal coverage (SEACR input) |

### Pipeline State JSON

The full pipeline state is available under the **Pipeline JSON** accordion in the
Chat tab. It contains all sample metadata, generated commands, agent action log,
and QC checklist, and updates live whenever the agent modifies the pipeline.



## How to Test — Step by Step

### Test 1: Setup and pipeline generation

1. Enter your model name in the **Model name** field
   (run `ollama list` to confirm the exact name including tag)
2. Leave Ollama URL as `http://localhost:11434`
3. Click **Test Connection**
   - Pass: `Connected successfully. Model <name> is available.`
   - Fail: confirm `ollama serve` is running in another terminal
4. Upload `metadata_demo.csv`
5. Click **Generate Pipeline**

Expected result on the right panel:
- Pipeline Overview table: H3K27me3, broad, hg38, SEACR_relaxed, 4 samples
- Design Rationale explaining why SEACR and broad were chosen
- QC Checklist with 6 items (no checkboxes)
- Sample-Level Shell Commands for all 4 samples, split across multiple lines



### Test 2: Conceptual question — no pipeline change expected

Switch to **Chat with Agent** tab. Send:

```
Why did you choose SEACR_relaxed instead of MACS2 for H3K27me3?
```

Expected:
- Agent explains SEACR is designed for CUT&Tag's low background and H3K27me3 is a
  broad mark with no IgG control
- Pipeline state panel on the right does NOT change
- Agent Log does NOT show any new `[Agent]` entries



### Test 3: Genome switch — pipeline update expected

```
Please switch the reference genome to hg19.
```

Expected:
- Agent confirms the change
- Pipeline state: **Genome: hg19**
- Agent Log: `[Agent] Genome changed to hg19: ...`
- Updated commands accordion shows all 4 commands regenerated



### Test 4: Mark type update — pipeline update expected

```
I want to use broad peak calling mode for H3K27me3. Please update the pipeline.
```

Expected:
- Pipeline state: **Mark type: broad**, **Caller: SEACR_relaxed**
- All sample commands regenerated



### Test 5: QC debugging — too few peaks

```
After running the pipeline, each sample only has a few hundred peaks. That seems too low. What could be wrong?
```

Expected:
- Agent lists diagnostic causes: genome mismatch, low alignment rate, wrong mark
  type, missing control, stringent thresholds, low library complexity
- Agent Log: `[Debug] Possible causes for low peak counts: ...`
- Pipeline caller and genome do NOT change (no explicit change requested)



### Test 6: QC debugging — noisy signal

```
The signal looks very noisy and there are too many peaks. How should I fix this?
```

Expected:
- Agent explains over-calling causes and remedies
- Agent Log records a debug note
- No structural pipeline changes



### Test 7: Cross-turn memory

Send first:

```
What is FRiP score and why does it matter?
```

Then send:

```
My FRiP score is only 0.03. What should I do?
```

Expected:
- First reply explains FRiP definition and acceptable thresholds
- Second reply uses context from the first turn and gives specific advice
  for a value of 0.03 without re-explaining what FRiP is



### Test 8: Agent Design tab

Click the **Agent Design** tab.

Expected: Architecture diagram in ASCII art, component summary table, explanation of
structured JSON output vs function calling, and a table of example interactions.



## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Connection refused` when testing | Ollama server not running | Run `ollama serve` in a separate terminal |
| `Model not found` when testing | Wrong model name or tag | Run `ollama list` and copy the exact name |
| Chatbot shows `Error` on send | Gradio version mismatch | Run `pip install gradio --upgrade` |
| Pipeline does not update after chat | Agent chose `action=none` | Rephrase as an explicit request, e.g. `Please switch to hg19` |
| Long wait before first response | Model loading into GPU | Normal on first request; subsequent responses are faster |
| `ollama serve` shows 404 on POST | Model name typo | Double-check with `ollama list` |



## Knowledge Base

The agent RAG knowledge base (`DEFAULT_KB` in `cuttag_agent.py`) contains 12 entries:

| Entry | Topic |
|---|---|
| kb01 | CUT&Tag workflow overview |
| kb02 | SEACR preferred over MACS2 for CUT&Tag |
| kb03 | Broad vs narrow mark classification |
| kb04 | IgG control BAM usage |
| kb05 | FRiP score interpretation |
| kb06 | Too few peaks — debugging checklist |
| kb07 | Too many peaks / noisy signal |
| kb08 | Replicate concordance (IDR) |
| kb09 | Genome build mismatch detection |
| kb10 | MACS2 BAMPE mode requirement |
| kb11 | Differential enrichment analysis |
| kb12 | Spike-in normalisation |

To extend the knowledge base, add entries to the `DEFAULT_KB` list in
`cuttag_agent.py` following the format:

```python
{"id": "kb13", "title": "Your topic", "text": "Your knowledge text here."}
```



## Limitations

- Peak calling only: This prototype automates the peak-calling step. Alignment,
  QC metrics computation, bigwig generation, and differential analysis are not yet
  included but follow the same design pattern.
- No real execution: The agent generates shell commands but does not run them.
  Copy commands from the UI and submit them to your HPC scheduler.
- Local model quality: Response quality depends on the model loaded in Ollama.
  Larger models produce more accurate and nuanced CUT&Tag-specific advice.
- KB coverage: The knowledge base covers common scenarios. Highly specialised
  experimental designs may require manual KB extension.



## Research and Citation

If you use this project in a paper, report, thesis, or study, please cite this repository:

```bibtex
@misc{Chen2026CUTTagAgent,
  author = {Yuyan Chen},
  title  = {CUT\&Tag Agent: An AI-Powered Epigenomic Pipeline Assistant for Interactive Peak-Calling Workflow Generation and Result Interpretation},
  year   = {2026},
  url    = {https://github.com/Yukyin/cuttag-agent}
}
```

## License

Noncommercial use is governed by `LICENSE` (PolyForm Noncommercial 1.0.0).  
Commercial use requires a separate agreement — see `COMMERCIAL_LICENSE.md`.

📨 Commercial inquiries: yolandachen0313@gmail.com

