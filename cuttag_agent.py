"""
CUT&Tag Agent  -  Local LLM Version (Ollama)
=============================================

Agent Engineering Design
-------------------------
1. LLM Backbone    : Local Ollama model via OpenAI-compatible endpoint
                     (localhost:11434). No API key required.

2. Agentic Structured Output (replaces Tool Use)
                   : Local models have unstable function-calling support,
                     so we use a structured JSON output approach instead:
                       - System prompt instructs the model to always return
                         {"action": "...", "params": {...}, "message": "..."}
                       - Agent parses JSON -> executes action -> updates plan
                       - Falls back to plain text if JSON parsing fails
                     This is still an observe -> reason -> act -> respond loop.

3. RAG (Retrieval-Augmented Generation)
                   : Before each LLM call, keyword-overlap search retrieves
                     the top-3 most relevant knowledge-base entries and injects
                     them into the system prompt.

4. Conversation Memory
                   : Full conversation history (sliding window, last 10 turns)
                     is passed to the LLM on every call for cross-turn reasoning.

5. Focus Step      : Peak Calling - caller selection, command generation, QC.

Usage:
  1. Install Ollama: https://ollama.ai
  2. Pull a model:  ollama pull <model-name>
  3. Start server:  ollama serve
  4. pip install -r requirements.txt
  5. python cuttag_agent.py
"""

import csv
import json
import os
import re
import shlex
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import gradio as gr
from openai import OpenAI


# ======================================================================
#  DATA MODELS
# ======================================================================

@dataclass
class SampleRecord:
    sample_id: str
    condition: str
    replicate: str
    target: str
    mark_type: str
    genome: str
    bam_path: str
    control_bam: str = ""


@dataclass
class PipelinePlan:
    step_name: str
    target: str
    mark_type: str
    genome: str
    peak_caller: str
    rationale: List[str]
    commands: List[Dict[str, str]]
    qc_checks: List[str]
    next_outputs: List[str]


# ======================================================================
#  KNOWLEDGE BASE  (RAG source - tutorial comments + protocol notes)
# ======================================================================

DEFAULT_KB: List[Dict[str, str]] = [
    {"id": "kb01", "title": "Workflow overview",
     "text": "CUT&Tag analysis includes alignment (Bowtie2), QC (samtools flagstat), "
             "peak calling (MACS2 or SEACR), signal visualisation (deepTools bigwig), "
             "and optional differential enrichment analysis (DiffBind/DESeq2)."},
    {"id": "kb02", "title": "SEACR preferred for CUT&Tag",
     "text": "SEACR is specifically designed for CUT&Tag data and often outperforms MACS2 "
             "because CUT&Tag has lower background. SEACR uses top 1% of signal by default "
             "when no IgG control is available."},
    {"id": "kb03", "title": "Broad vs narrow peak calling",
     "text": "H3K27me3, H3K36me3, and H3K9me3 are broad/repressive marks requiring broad "
             "peak calling. H3K4me3, H3K27ac, and CTCF are sharp marks requiring narrow "
             "peak calling. Incorrect mode selection results in near-zero peaks."},
    {"id": "kb04", "title": "IgG control BAM",
     "text": "An IgG control BAM enables control-aware peak calling (MACS2 -c flag or "
             "SEACR with control bedgraph). Without a control, SEACR uses a self-threshold "
             "in relaxed mode."},
    {"id": "kb05", "title": "FRiP score QC",
     "text": "FRiP (Fraction of Reads in Peaks) above 0.1 is generally acceptable; "
             "above 0.2 is high quality. FRiP below 0.05 suggests poor antibody efficiency, "
             "wrong peak-calling mode, or alignment issues."},
    {"id": "kb06", "title": "Too few peaks - debugging",
     "text": "Low peak counts: (1) verify genome build matches BAM header, "
             "(2) check BAM indexing (.bai present), (3) consider broad-mark settings, "
             "(4) inspect alignment rate and duplication rate."},
    {"id": "kb07", "title": "Too many peaks / noisy signal",
     "text": "Excess peaks: use IgG control, switch to SEACR stringent mode, "
             "raise MACS2 q-value cutoff (e.g. 0.05 -> 0.01), or verify antibody lot."},
    {"id": "kb08", "title": "Replicate concordance",
     "text": "Use IDR or bedtools intersect to assess replicate concordance. "
             "Low concordance may indicate antibody variability or experimental inconsistency. "
             "At least 2 biological replicates per condition are required."},
    {"id": "kb09", "title": "Genome build mismatch",
     "text": "Genome mismatch causes silent downstream failures. Check BAM @SQ headers "
             "with: samtools view -H sample.bam | grep '@SQ'"},
    {"id": "kb10", "title": "MACS2 BAMPE mode",
     "text": "For paired-end CUT&Tag BAMs always use -f BAMPE in MACS2. This uses the "
             "actual fragment length from read pairs instead of estimating it."},
    {"id": "kb11", "title": "Differential enrichment analysis",
     "text": "DiffBind uses DESeq2/edgeR on read counts within a consensus peak set. "
             "Requires at least 2 biological replicates per condition for statistical testing."},
    {"id": "kb12", "title": "Spike-in normalisation",
     "text": "E. coli or Drosophila spike-in DNA enables absolute quantification. "
             "Use deepTools bamCompare --scalingFactors for spike-in-normalised bigwigs."},
]


def kb_search(query: str, kb_items: List[Dict], k: int = 3) -> List[Dict]:
    """Keyword-overlap RAG retrieval. Returns top-k most relevant KB entries."""
    q_tokens = set(re.findall(r"[a-zA-Z0-9_]+", query.lower()))
    scored = []
    for item in kb_items:
        item_tokens = set(re.findall(
            r"[a-zA-Z0-9_]+", (item["title"] + " " + item["text"]).lower()))
        scored.append((len(q_tokens & item_tokens), item))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [item for score, item in scored[:k] if score > 0]
    return top if top else kb_items[:k]


# ======================================================================
#  PIPELINE LOGIC
# ======================================================================

BROAD_TARGETS = {"h3k27me3", "h3k36me3", "h3k9me3", "h3k9me2", "broad"}
SHARP_TARGETS = {"h3k4me3", "h3k27ac", "h3k4me1", "ctcf", "sharp", "tf",
                 "transcription_factor", "h3k9ac"}


def load_metadata(file_obj: Any) -> Tuple[List[SampleRecord], str]:
    if file_obj is None:
        return [], "No metadata file uploaded."
    rows: List[SampleRecord] = []
    with open(file_obj.name, "r", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            rows.append(SampleRecord(
                sample_id=(raw.get("sample_id") or "").strip(),
                condition=(raw.get("condition") or "").strip(),
                replicate=(raw.get("replicate") or "").strip(),
                target=(raw.get("target") or "").strip(),
                mark_type=(raw.get("mark_type") or "auto").strip().lower(),
                genome=(raw.get("genome") or "hg38").strip(),
                bam_path=(raw.get("bam_path") or "").strip(),
                control_bam=(raw.get("control_bam") or "").strip(),
            ))
    valid = [r for r in rows if r.sample_id and r.target and r.bam_path]
    return valid, f"Loaded {len(valid)} valid samples."


def infer_mark_type(target: str, user_hint: str) -> str:
    if user_hint in {"broad", "sharp"}:
        return user_hint
    t = target.lower().strip()
    if t in BROAD_TARGETS:
        return "broad"
    if t in SHARP_TARGETS:
        return "sharp"
    return "sharp"


def choose_caller(mark_type: str, has_control: bool) -> str:
    if mark_type == "broad":
        return "MACS2_broad" if has_control else "SEACR_relaxed"
    return "MACS2_narrow"


def gsize(genome: str) -> str:
    return "mm" if genome.lower() in {"mm10", "mm39", "grcm38"} else "hs"


def build_command(sample: SampleRecord, mark_type: str, caller: str) -> str:
    bam  = shlex.quote(sample.bam_path)
    ctrl = shlex.quote(sample.control_bam) if sample.control_bam else ""
    name = shlex.quote(sample.sample_id)
    outd = shlex.quote(f"results/{sample.sample_id}/peaks")
    gs   = gsize(sample.genome)

    if caller == "MACS2_narrow":
        cmd = (f"mkdir -p {outd} && "
               f"macs2 callpeak -t {bam} "
               f"{'-c ' + ctrl + ' ' if ctrl else ''}"
               f"-f BAMPE -g {gs} -n {name} --outdir {outd} -q 0.01")
    elif caller == "MACS2_broad":
        cmd = (f"mkdir -p {outd} && "
               f"macs2 callpeak -t {bam} "
               f"{'-c ' + ctrl + ' ' if ctrl else ''}"
               f"-f BAMPE -g {gs} -n {name} --outdir {outd} "
               f"--broad --broad-cutoff 0.1")
    else:  # SEACR_relaxed
        cmd = (f"mkdir -p {outd} && "
               f"bedtools genomecov -bg -ibam {bam} > {outd}/signal.bedgraph && "
               f"SEACR_1.3.sh {outd}/signal.bedgraph 0.01 non stringent "
               f"{outd}/{sample.sample_id}_seacr")

    return re.sub(r"\s+", " ", cmd).strip()


def _wrap_command(cmd: str) -> str:
    """Split long commands at && and > for readability."""
    # Split at &&
    parts = cmd.split(" && ")
    # Also split bedgraph redirect line at >
    wrapped = []
    for part in parts:
        if " > " in part and "bedgraph" in part:
            part = part.replace(" > ", " > \n    ", 1)
        wrapped.append(part)
    return " && \\\n  ".join(wrapped)


def create_plan(samples: List[SampleRecord]) -> PipelinePlan:
    if not samples:
        return PipelinePlan(
            step_name="peak_calling", target="N/A", mark_type="N/A",
            genome="N/A", peak_caller="N/A",
            rationale=["Upload a metadata CSV and click Generate Pipeline."],
            commands=[], qc_checks=[], next_outputs=[],
        )

    first    = samples[0]
    inferred = infer_mark_type(first.target, first.mark_type)
    has_ctrl = any(bool(s.control_bam) for s in samples)
    caller   = choose_caller(inferred, has_ctrl)

    commands = []
    for s in samples:
        mt = infer_mark_type(s.target, s.mark_type)
        c  = choose_caller(mt, bool(s.control_bam))
        commands.append({
            "sample_id":   s.sample_id, "condition":  s.condition,
            "replicate":   s.replicate, "target":     s.target,
            "mark_type":   mt,          "peak_caller": c,
            "command":     build_command(s, mt, c),
        })

    rationale = [
        f"Target '{first.target}' classified as **{inferred}** mark -> "
        f"{'broad' if inferred == 'broad' else 'narrow'} peak calling selected.",
        f"Reference genome: **{first.genome}**.",
        ("Control BAM detected -> control-aware peak calling enabled."
         if has_ctrl else
         "No control BAM -> threshold-based (self-normalised) peak calling."),
        "Pipeline generated automatically from metadata - no manual scripting required.",
    ]

    qc_checks = [
        "Verify BAM files exist and are indexed (.bai files in same directory as .bam).",
        "Check alignment rate via samtools flagstat - expect >70% for CUT&Tag.",
        "Confirm >= 2 biological replicates per condition.",
        "Review peak count per sample and compare to published benchmarks for this target.",
        "Calculate FRiP score (>0.1 acceptable, >0.2 high quality).",
        "Assess replicate concordance via IDR or bedtools intersect overlap.",
    ]

    return PipelinePlan(
        step_name="peak_calling", target=first.target,
        mark_type=inferred, genome=first.genome, peak_caller=caller,
        rationale=rationale, commands=commands, qc_checks=qc_checks,
        next_outputs=[
            "Peak files (.narrowPeak / .broadPeak / _seacr.stringent.bed)",
            "Signal tracks (.bedgraph -> bigwig via bedGraphToBigWig)",
            "QC summary report",
        ],
    )


# ======================================================================
#  STRUCTURED JSON ACTION EXECUTOR
# ======================================================================

VALID_ACTIONS = {
    "none", "adjust_peak_caller", "change_genome",
    "change_mark_type", "add_debug_note",
}
VALID_CALLERS = {"MACS2_narrow", "MACS2_broad", "SEACR_relaxed"}
VALID_GENOMES = {"hg38", "hg19", "mm10", "mm39"}
VALID_MARKS   = {"broad", "sharp"}


def _bam_from_cmd(cmd: str) -> str:
    m = re.search(r"-ibam\s+'([^']+)'|-t\s+'([^']+)'|-ibam\s+(\S+)|-t\s+(\S+)", cmd)
    if not m:
        return "data/example.bam"
    return next(g for g in m.groups() if g).strip("'\"")


def execute_action(plan: PipelinePlan, action: str,
                   params: Dict) -> Tuple[PipelinePlan, str]:
    """Apply an agent action to the plan. Returns (updated_plan, result_message)."""
    p = PipelinePlan(**asdict(plan))

    if action == "adjust_peak_caller":
        caller = params.get("caller", "")
        reason = params.get("reason", "")
        if caller not in VALID_CALLERS:
            return plan, f"Invalid peak caller: {caller}"
        p.peak_caller = caller
        p.rationale.append(f"[Agent] Peak caller switched to **{caller}**: {reason}")
        for item in p.commands:
            s = SampleRecord(
                sample_id=item["sample_id"], condition=item["condition"],
                replicate=item["replicate"], target=item["target"],
                mark_type=item["mark_type"], genome=p.genome,
                bam_path=_bam_from_cmd(item["command"]), control_bam="",
            )
            item["peak_caller"] = caller
            item["command"] = build_command(s, item["mark_type"], caller)
        return p, f"Peak caller updated to {caller}."

    elif action == "change_genome":
        genome = params.get("genome", "")
        reason = params.get("reason", "")
        if genome not in VALID_GENOMES:
            return plan, f"Invalid genome: {genome}"
        p.genome = genome
        p.rationale.append(f"[Agent] Genome changed to **{genome}**: {reason}")
        for item in p.commands:
            s = SampleRecord(
                sample_id=item["sample_id"], condition=item["condition"],
                replicate=item["replicate"], target=item["target"],
                mark_type=item["mark_type"], genome=genome,
                bam_path=_bam_from_cmd(item["command"]), control_bam="",
            )
            item["command"] = build_command(s, item["mark_type"], item["peak_caller"])
        return p, f"Genome updated to {genome}."

    elif action == "change_mark_type":
        mark_type = params.get("mark_type", "")
        reason    = params.get("reason", "")
        if mark_type not in VALID_MARKS:
            return plan, f"Invalid mark type: {mark_type}"
        p.mark_type   = mark_type
        p.peak_caller = choose_caller(mark_type, False)
        p.rationale.append(
            f"[Agent] Mark type -> **{mark_type}**, "
            f"peak caller -> **{p.peak_caller}**: {reason}"
        )
        for item in p.commands:
            item["mark_type"]   = mark_type
            item["peak_caller"] = p.peak_caller
            s = SampleRecord(
                sample_id=item["sample_id"], condition=item["condition"],
                replicate=item["replicate"], target=item["target"],
                mark_type=mark_type, genome=p.genome,
                bam_path=_bam_from_cmd(item["command"]), control_bam="",
            )
            item["command"] = build_command(s, mark_type, p.peak_caller)
        return p, f"Mark type -> {mark_type}, peak caller -> {p.peak_caller}."

    elif action == "add_debug_note":
        note = params.get("note", "")
        p.rationale.append(f"[Debug] {note}")
        return p, "Debug note recorded."

    return plan, ""  # action == "none"


# ======================================================================
#  LOCAL LLM AGENT  (Ollama)
# ======================================================================

ACTION_SCHEMA = """\
You MUST output ONLY the following JSON format. No other text, no markdown fences:

{"action": "<action>", "params": {<params>}, "message": "<reply to user>"}

Action options and their params:
- "none"               -> params: {}
- "adjust_peak_caller" -> params: {"caller": "MACS2_narrow"|"MACS2_broad"|"SEACR_relaxed", "reason": "..."}
- "change_genome"      -> params: {"genome": "hg38"|"hg19"|"mm10"|"mm39", "reason": "..."}
- "change_mark_type"   -> params: {"mark_type": "broad"|"sharp", "reason": "..."}
- "add_debug_note"     -> params: {"note": "..."}

Rules:
- ONLY call an action if the user EXPLICITLY asks to change something (e.g. "switch to hg19", "use broad mode"). Do NOT change the pipeline when the user is just asking a question or reporting a problem.
- If user reports a QC problem (too few peaks, noisy signal) -> action="add_debug_note" only, explain causes in message. Do NOT switch callers or genome unless explicitly asked.
- If user asks a conceptual or interpretive question -> action="none", answer in message.
- Keep message under 120 words. Be concise and scientifically accurate."""


def build_system_prompt(plan: PipelinePlan, kb_context: str) -> str:
    summary = json.dumps({
        "step":        plan.step_name,
        "target":      plan.target,
        "mark_type":   plan.mark_type,
        "genome":      plan.genome,
        "peak_caller": plan.peak_caller,
        "n_samples":   len(plan.commands),
        "recent_log":  plan.rationale[-4:],
    }, indent=2)

    return (
        "You are CUT&Tag Pipeline Assistant, an expert bioinformatician specialising in "
        "chromatin profiling (CUT&Tag, CUT&RUN, ChIP-seq).\n"
        "Your users are wet-lab researchers who do not program. Help them understand "
        "results, debug QC issues, and adjust the analysis pipeline.\n\n"
        f"Current pipeline state:\n```json\n{summary}\n```\n\n"
        f"Relevant knowledge base (RAG retrieval):\n{kb_context}\n\n"
        f"Output format requirement:\n{ACTION_SCHEMA}"
    )


def extract_json(text: str) -> Dict:
    """Extract JSON from LLM output, handling markdown fences and extra text."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"action": "none", "params": {}, "message": text.strip()}


def agent_turn(
    user_message: str,
    plan: PipelinePlan,
    kb_items: List[Dict],
    model_name: str,
    ollama_url: str,
    conv_history: List[Dict],
) -> Tuple[PipelinePlan, str, List[Dict]]:
    """
    Run one agent turn.

    Flow:
      1. Retrieve relevant KB entries (RAG)
      2. Build system prompt (pipeline state + KB context)
      3. Call local LLM via Ollama OpenAI-compatible endpoint
      4. Parse JSON action -> execute -> update PipelinePlan
      5. Return natural-language reply + updated plan
    """
    retrieved  = kb_search(user_message, kb_items, k=3)
    kb_context = "\n".join(
        f"- **{item['title']}**: {item['text']}" for item in retrieved
    )
    system = build_system_prompt(plan, kb_context)

    messages = [{"role": "system", "content": system}]
    for m in conv_history[-10:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_message})

    client = OpenAI(
        base_url=ollama_url.rstrip("/") + "/v1",
        api_key="ollama",
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.3,
        max_tokens=800,
    )

    raw_output = response.choices[0].message.content or ""
    parsed  = extract_json(raw_output)
    action  = parsed.get("action", "none")
    params  = parsed.get("params", {})
    message = parsed.get("message", raw_output)

    action_log = ""
    if action and action != "none" and action in VALID_ACTIONS:
        plan, result_msg = execute_action(plan, action, params)
        if result_msg:
            action_log = f"**Pipeline updated:** {result_msg}\n\n"

    reply = action_log + message

    new_history = conv_history + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": reply},
    ]
    return plan, reply, new_history


# ======================================================================
#  RENDERING HELPERS
# ======================================================================

def render_overview(plan: PipelinePlan, load_msg: str) -> str:
    if not plan.commands:
        return f"*{load_msg}*"
    return (
        "### Pipeline Overview\n"
        "| Field | Value |\n|---|---|\n"
        f"| Status | {load_msg} |\n"
        f"| Focused step | {plan.step_name} |\n"
        f"| Epigenomic target | {plan.target} |\n"
        f"| Mark type | {plan.mark_type} |\n"
        f"| Reference genome | {plan.genome} |\n"
        f"| Peak caller | {plan.peak_caller} |\n"
        f"| Samples | {len(plan.commands)} |\n\n"
        "### Design Rationale\n"
        + "\n".join(f"- {r}" for r in plan.rationale)
        + "\n\n### QC Checklist\n"
        + "\n".join(f"- {q}" for q in plan.qc_checks)
    )


def render_commands(plan: PipelinePlan) -> str:
    if not plan.commands:
        return "*Upload metadata and click Generate Pipeline to see commands.*"
    chunks = ["### Sample-Level Shell Commands\n"]
    for item in plan.commands:
        wrapped = _wrap_command(item["command"])
        chunks.append(
            f"#### {item['sample_id']}\n"
            f"| Field | Value |\n|---|---|\n"
            f"| Condition | `{item['condition']}` |\n"
            f"| Replicate | `{item['replicate']}` |\n"
            f"| Target | `{item['target']}` |\n"
            f"| Mark type | `{item['mark_type']}` |\n"
            f"| Peak caller | `{item['peak_caller']}` |\n\n"
            f"```bash\n{wrapped}\n```"
        )
    return "\n\n---\n\n".join(chunks)


def render_state(plan: PipelinePlan) -> str:
    return (
        "### Current Pipeline State\n"
        f"- **Caller:** `{plan.peak_caller}`\n"
        f"- **Mark type:** `{plan.mark_type}`\n"
        f"- **Genome:** `{plan.genome}`\n"
        f"- **Samples:** {len(plan.commands)}\n\n"
        "### Agent Log (last 5 entries)\n"
        + "\n".join(f"- {r}" for r in plan.rationale[-5:])
    )


# ======================================================================
#  GRADIO CALLBACKS
# ======================================================================

def test_connection(model_name: str, ollama_url: str) -> str:
    """Test Ollama connectivity and model availability."""
    try:
        client = OpenAI(
            base_url=ollama_url.rstrip("/") + "/v1",
            api_key="ollama",
        )
        client.chat.completions.create(
            model=model_name or "gpt-oss:20b",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
        )
        return f"Connected successfully. Model `{model_name}` is available."
    except Exception as e:
        err = str(e)
        if "Connection refused" in err or "connect" in err.lower():
            return "Connection failed. Make sure `ollama serve` is running."
        elif "not found" in err.lower() or "404" in err:
            return f"Model `{model_name}` not found. Run `ollama list` to see available models."
        else:
            return f"Error: {err[:120]}"


def initialize(file_obj: Any):
    samples, load_msg = load_metadata(file_obj)
    plan      = create_plan(samples)
    plan_json = json.dumps(asdict(plan), indent=2)
    kb_json   = json.dumps(DEFAULT_KB, indent=2)
    return (
        render_overview(plan, load_msg),
        render_commands(plan),
        render_state(plan),
        plan_json,
        kb_json,
        [],   # conv_history
        [],   # chatbot display
    )


def chat(
    user_msg: str,
    plan_json: str,
    kb_json: str,
    model_name: str,
    ollama_url: str,
    conv_history: List[Dict],
    chatbot_display: list,
):
    if not user_msg.strip():
        plan = PipelinePlan(**json.loads(plan_json))
        return (chatbot_display, plan_json, render_state(plan),
                render_commands(plan), conv_history, "")

    plan     = PipelinePlan(**json.loads(plan_json))
    kb_items = json.loads(kb_json)
    model    = (model_name or "").strip() or "gpt-oss:20b"
    url      = (ollama_url or "").strip() or "http://localhost:11434"

    try:
        updated_plan, reply, new_history = agent_turn(
            user_msg, plan, kb_items, model, url, conv_history
        )
    except Exception as e:
        reply = (
            f"Ollama connection error: {e}\n\n"
            "Please check:\n"
            "1. `ollama serve` is running\n"
            "2. Model name is correct (run `ollama list` to verify)\n"
            "3. Ollama URL is correct (default: `http://localhost:11434`)"
        )
        updated_plan = plan
        new_history  = conv_history + [
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": reply},
        ]

    new_display = chatbot_display + [
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": reply},
    ]
    return (
        new_display,
        json.dumps(asdict(updated_plan), indent=2),
        render_state(updated_plan),
        render_commands(updated_plan),
        new_history,
        "",
    )


# ======================================================================
#  AGENT DESIGN DOCUMENTATION
# ======================================================================

DESIGN_MD = """
## Agent Engineering Design

This tool demonstrates an agentic workflow for the **peak calling** step of CUT&Tag
data analysis. Users provide sample metadata; the agent generates a customised pipeline
and supports natural-language interaction for interpretation, QC debugging, and live
pipeline adjustment.

---

### System Architecture

```
User input
    |
    v
+----------------------------------------------+
|              RAG Retrieval                   |
|  Keyword-overlap search over KB -> top-3     |
+--------------------+-------------------------+
                     | injected into system prompt
                     v
+----------------------------------------------+
|          Local LLM (Ollama)                  |
|  Input:  system prompt (pipeline state + KB) |
|          conversation history (memory)       |
|          user message                        |
|                                              |
|  Output: structured JSON                     |
|  { "action": "...", "params": {...},         |
|    "message": "..." }                        |
+--------+-------------------------------------+
         |
   +-----+-----------------------------+
   |                                   |
action="none"               action = pipeline change
   |                                   |
   |              +--------------------v------------------+
   |              |           Action Executor             |
   |              |  adjust_peak_caller                   |
   |              |  change_genome                        |
   |              |  change_mark_type                     |
   |              |  add_debug_note                       |
   |              +--------------------+------------------+
   |                                   | updates PipelinePlan
   +-------------+---------------------+
                 v
         Reply to user + update UI
```

---

### Component Summary

| Component | Technology | Role |
|---|---|---|
| **LLM backbone** | Local Ollama (OpenAI-compatible API) | Reasoning, interpretation, decision-making |
| **Agentic output** | Structured JSON (replaces Function Calling) | LLM autonomously decides whether and how to modify the pipeline |
| **RAG** | Keyword-overlap retrieval over 12 KB entries | Grounds answers in CUT&Tag domain knowledge |
| **Memory** | Sliding-window conversation history (last 10 turns) | Enables cross-turn debugging |
| **Pipeline state** | `PipelinePlan` dataclass serialised to JSON | Single source of truth; every action mutates it |
| **UI** | Gradio Blocks | No-code interface for wet-lab users |

---

### Why Structured JSON Output Instead of Function Calling?

Local open-source models have unstable support for native function calling format.
Requiring the model to always output a fixed JSON schema is more compatible:
- System prompt explicitly defines the output schema and valid action enum values
- Code-layer JSON parsing with fallback handles malformed output gracefully
- The LLM still autonomously decides the action type and parameters,
  preserving the **agentic decision-making capability**
- This "pseudo tool-use" pattern is widely used in local LLM deployments

---

### Focus Step: Peak Calling

Peak calling is the most consequential and error-prone step in CUT&Tag analysis:
- Wrong narrow/broad mode -> near-zero peaks
- Genome build mismatch -> silent downstream failure
- Threshold choice -> directly controls false-positive rate

The agent can diagnose and correct all three error classes from natural-language
input alone, without the user touching any code.

---

### Example Chat Interactions

| User Input | Expected Agent Behaviour |
|---|---|
| Why did you choose SEACR? | Explain the rationale (action=none) |
| Switch to hg19 | Call change_genome, rebuild commands |
| Treat H3K27me3 as broad mark | Call change_mark_type + adjust_peak_caller |
| My peak counts are very low | Diagnose causes, call add_debug_note |
| The signal looks very noisy | Suggest tightening thresholds, possibly adjust caller |
"""


# ======================================================================
#  GRADIO UI
# ======================================================================

with gr.Blocks(title="CUT&Tag Agent - Local LLM") as demo:

    plan_state = gr.State(value=json.dumps(asdict(create_plan([])), indent=2))
    kb_state   = gr.State(value=json.dumps(DEFAULT_KB, indent=2))
    conv_state = gr.State(value=[])

    gr.Markdown("""
# CUT&Tag Agent
**AI-powered epigenomic pipeline assistant** — upload sample metadata,
generate a customised peak-calling pipeline, then chat with the agent
to interpret results, debug QC issues, and adjust parameters.
""")

    with gr.Tabs():

        # -- Tab 1: Setup & Pipeline -----------------------------------
        with gr.TabItem("Setup & Pipeline"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Ollama Configuration")
                    model_box = gr.Textbox(
                        label="Model name",
                        value="gpt-oss:20b",
                        placeholder="e.g. gpt-oss:20b / qwen2.5:14b / llama3.1",
                        info="Must match the name shown by `ollama list`",
                    )
                    url_box = gr.Textbox(
                        label="Ollama URL",
                        value="http://localhost:11434",
                        info="Default for local machine. Change to IP:port for remote server.",
                    )
                    with gr.Row():
                        test_btn = gr.Button("Test Connection", size="sm")
                        test_status = gr.Textbox(
                            label="", interactive=False,
                            show_label=False, scale=3,
                        )
                    gr.Markdown("---")
                    gr.Markdown("### Sample Metadata")
                    metadata_file = gr.File(
                        label="Upload CSV file",
                        file_types=[".csv"],
                    )
                    gr.Markdown(
                        "**Required columns:** `sample_id`, `condition`, `replicate`, "
                        "`target`, `mark_type`, `genome`, `bam_path`  \n"
                        "**Optional:** `control_bam`"
                    )
                    init_btn = gr.Button(
                        "Generate Pipeline", variant="primary", size="lg"
                    )

                with gr.Column(scale=2):
                    overview_md = gr.Markdown(
                        "*Upload metadata and click Generate Pipeline.*"
                    )

            commands_md = gr.Markdown()

        # -- Tab 2: Chat with Agent ------------------------------------
        with gr.TabItem("Chat with Agent"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="CUT&Tag Agent",
                        height=500,
                    )
                    with gr.Row():
                        user_input = gr.Textbox(
                            label="Your message",
                            placeholder=(
                                "e.g. Why did you choose SEACR?  |  Switch to hg19  |  "
                                "My peak counts are very low  |  Use broad peak calling"
                            ),
                            scale=4,
                        )
                        send_btn = gr.Button(
                            "Send", variant="primary", scale=1
                        )
                    gr.Markdown(
                        "*The agent can explain choices, adjust the pipeline, "
                        "and help debug QC issues - no programming required.*"
                    )

                with gr.Column(scale=2):
                    state_md = gr.Markdown("*Generate a pipeline first.*")
                    gr.Markdown("---")
                    with gr.Accordion("Updated commands", open=False):
                        live_commands_md = gr.Markdown()
                    with gr.Accordion("Pipeline JSON", open=False):
                        plan_json_box = gr.Textbox(
                            lines=18, label="PipelinePlan JSON"
                        )

        # -- Tab 3: Agent Design ---------------------------------------
        with gr.TabItem("Agent Design"):
            gr.Markdown(DESIGN_MD)

    # -- Event wiring --------------------------------------------------

    test_btn.click(
        fn=test_connection,
        inputs=[model_box, url_box],
        outputs=[test_status],
    )

    init_btn.click(
        fn=initialize,
        inputs=[metadata_file],
        outputs=[overview_md, commands_md, state_md,
                 plan_state, kb_state, conv_state, chatbot],
    ).then(
        fn=lambda j: j,
        inputs=[plan_state], outputs=[plan_json_box],
    ).then(
        fn=lambda j: render_commands(PipelinePlan(**json.loads(j))),
        inputs=[plan_state], outputs=[live_commands_md],
    )

    def handle_send(msg, pj, kj, mn, url, ch, cd):
        return chat(msg, pj, kj, mn, url, ch, cd)

    send_btn.click(
        fn=handle_send,
        inputs=[user_input, plan_state, kb_state,
                model_box, url_box, conv_state, chatbot],
        outputs=[chatbot, plan_state, state_md,
                 live_commands_md, conv_state, user_input],
    ).then(
        fn=lambda j: j,
        inputs=[plan_state], outputs=[plan_json_box],
    )

    user_input.submit(
        fn=handle_send,
        inputs=[user_input, plan_state, kb_state,
                model_box, url_box, conv_state, chatbot],
        outputs=[chatbot, plan_state, state_md,
                 live_commands_md, conv_state, user_input],
    ).then(
        fn=lambda j: j,
        inputs=[plan_state], outputs=[plan_json_box],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
