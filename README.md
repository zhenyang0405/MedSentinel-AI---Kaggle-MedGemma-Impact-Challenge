# 🏥 MedSentinel AI

**Multimodal Missed-Diagnosis Prevention for Emergency Departments**

[![MedGemma](https://img.shields.io/badge/Model-MedGemma%201.5%204B-blue)](https://huggingface.co/google/medgemma-1.5-4b-it)
[![LangGraph](https://img.shields.io/badge/Framework-LangGraph-orange)](https://github.com/langchain-ai/langgraph)
[![Kaggle](https://img.shields.io/badge/Platform-Kaggle%20T4%20x2-lightgrey)](https://www.kaggle.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](#license)

> **Competition:** Kaggle MedGemma Impact Challenge  
> **Prize Target:** 🏆 Agentic Workflow Prize

---

## 🧠 Problem Statement

**12 million diagnostic errors occur annually in US emergency departments.** Current clinical decision support systems operate in silos — lab alerts only see labs, imaging AI only sees images, and EHR alerts only see history. Nobody connects them.

When a physician is managing 3 patients and receiving 47 alerts per hour, dangerous cross-modal contradictions slip through the cracks. A normal chest X-ray combined with elevated WBC, fever, and tachycardia in an immunocompromised patient may indicate occult pneumonia — but no single alert system catches this.

**MedSentinel AI automates the cross-modal reasoning that currently exists only in physicians' cognitive processes — and fails during cognitive overload, shift changes, and high-volume periods.**

<table>
<tr>
<th>❌ Current State (Siloed Alerts)</th>
<th>✅ MedSentinel AI (Cross-Modal)</th>
</tr>
<tr>
<td>
<b>Lab system:</b> "WBC 15.2 — HIGH" → Generic alert<br>
<b>Imaging AI:</b> "CXR — No acute findings" → Normal<br>
<b>EHR:</b> "Temp 38.7°C" → Vital sign flag<br><br>
<em>Three separate alerts. Nobody connects them.</em>
</td>
<td>
<b>One alert:</b> "⚠️ CRITICAL — Possible Occult Pneumonia"<br><br>
<em>Reasoning: Elevated WBC + fever + tachycardia CONTRADICTS normal CXR in immunocompromised patient. Early/atypical pneumonia may not be visible on plain film. Recommend CT chest.</em>
</td>
</tr>
</table>

---

## 🏗️ Architecture Overview

MedSentinel AI is an **8-node LangGraph StateGraph** that deploys **one MedGemma 1.5 4B model as 4 specialized tools** through prompt engineering alone.
<img width="1696" height="2496" alt="system architecture" src="https://github.com/user-attachments/assets/0a567b03-e3ca-492d-935a-058d5ccf6c47" />

### Single Model, Multiple Specialized Tools

| Agent Role | System Prompt Persona | Task Focus |
|---|---|---|
| Imaging Agent | Medical imaging specialist | Structured image findings (CXR, derm, fundus, pathology) |
| Lab Report Agent | Clinical pathologist | Lab value extraction + clinical flagging |
| Clinical History Agent | Attending physician | FHIR navigation + patient summary |
| Supervisor Agent | Senior attending, safety review | Cross-modal reasoning + contradiction detection |

---

## ⚡ Key Features

- **Cross-Modal Contradiction Detection** — Synthesizes imaging, labs, and clinical history to catch missed diagnoses that single-modality alerts cannot detect
- **Dynamic Triage Routing** — Conditionally activates only the agents needed based on available input data
- **Parallel Fan-Out / Fan-In** — 3 specialist agents execute simultaneously; Supervisor waits for all before reasoning
- **Evaluator-Optimizer Loop** — LLM-as-judge quality assurance with iterative refinement (max 2 iterations)
- **All-Normal Safety Check** — Rule-based post-processing filters hallucinated alerts when all data is concordant
- **Full Execution Trace** — Append-only audit trail via `operator.add` for complete explainability
- **Interactive Gradio Dashboard** — Pipeline flow visualization, severity-badged alerts with evidence chains, and pre-loaded example cases

---

## 🔗 LangGraph Patterns Demonstrated

| Pattern | Implementation |
|---|---|
| **Conditional Routing** | Triage Router dynamically selects agents based on available data |
| **Parallel Fan-Out** | 3 specialist agents execute simultaneously in one superstep |
| **Fan-In Convergence** | Supervisor waits for all agents before cross-modal reasoning |
| **Evaluator-Optimizer Loop** | Quality assurance with feedback-driven refinement |
| **Human-in-the-Loop Ready** | LangGraph interrupt support for clinical review gates |
| **Stateful Execution Trace** | Append-only log via `operator.add` for full auditability |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Model** | MedGemma 1.5 4B (instruction-tuned, bfloat16) |
| **Orchestration** | LangGraph StateGraph (8 nodes) |
| **Inference** | HuggingFace Transformers `pipeline("image-text-to-text")` |
| **Compute** | Kaggle T4 x2 GPU |
| **Interface** | Gradio Blocks |
| **Data Standard** | FHIR R4 (patient records) |
| **Language** | Python 3.10+ |

---

## 📊 Evaluation Results

### Synthetic Clinical Scenario Suite (10 scenarios)

| Metric | Score |
|---|---|
| **Accuracy** | 7/10 (70%) |
| **Recall** | 6/6 (100%) — all dangerous conditions caught |
| **Precision** | 6/9 (67%) |
| **F1 Score** | 0.80 |

### Confusion Matrix

| | Predicted Alert | Predicted No Alert |
|---|---|---|
| **Expected Alert** | TP: 6 | FN: 0 |
| **Expected No Alert** | FP: 3 | TN: 1 |

### Key Findings

- **100% recall on dangerous conditions** — sepsis, STEMI, PE, aortic dissection, tension pneumothorax, and DVT all correctly flagged
- **False positives on ambiguous cases** — panic attack (respiratory alkalosis), iron deficiency anemia, and viral URI generated unnecessary alerts
- **Root cause**: MedGemma 4B's limited reasoning capacity struggles with "expected abnormalities" — conditions where lab abnormalities are explained by the known diagnosis

---

## Hackathon Demo Video
https://github.com/user-attachments/assets/4aa068b7-ab68-4240-9159-e171f954cea9



## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (T4 or better)
- HuggingFace account with MedGemma access

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/medsentinel-ai.git
cd medsentinel-ai

# Install dependencies
pip install transformers>=4.50.0 accelerate bitsandbytes langgraph langchain-core gradio pillow huggingface_hub

# Authenticate with HuggingFace (requires MedGemma access)
huggingface-cli login
```

### Quick Start

```python
from medsentinel import run_medsentinel

result = run_medsentinel(
    medical_image=None,              # PIL Image or None
    lab_report_text="...",           # Lab report as text
    fhir_bundle={...},               # FHIR R4 Bundle dict
)

# Access results
print(result["safety_alerts"])       # Severity-sorted alerts
print(result["agent_trace"])         # Full execution trace
print(result["imaging_findings"])    # Structured imaging output
print(result["lab_findings"])        # Structured lab values
print(result["clinical_history"])    # Patient summary
```

### Run the Gradio Dashboard

```python
# Launch the interactive dashboard
demo.launch(share=True)
```

---

## 📁 Project Structure

```
medsentinel-ai/
├── medsentinel-ai-multimodal-missed-diagnosis-prevent.ipynb  # Full notebook (Phases 1-5)
├── README.md
├── LICENSE
└── assets/
    └── architecture.png          # Auto-generated LangGraph diagram
```

### Notebook Phases

| Phase | Title | What It Does |
|---|---|---|
| **Phase 1** | Foundation & Model Loading | Load MedGemma 1.5 4B, verify 4 modalities (CXR, labs, FHIR, reasoning) |
| **Phase 2** | LangGraph Agent Nodes | Define state schema, build 4 agent nodes, test independently |
| **Phase 3** | Agentic Workflow | Wire full StateGraph, add Supervisor + Evaluator-Optimizer, run scenarios |
| **Phase 4** | Interactive Dashboard | Gradio UI with pipeline flow visualization and pre-loaded examples |
| **Phase 5** | Evaluation & Metrics | 10 clinical scenarios, precision/recall/F1, error analysis |

---

## 🧪 Clinical Scenarios Tested

| ID | Scenario | Expected | Result |
|---|---|---|---|
| S01 | Healthy Routine Physical | No Alert | ✅ Correct |
| S02 | Panic Attack Mimicking ACS | No Alert | ❌ FP |
| S03 | Iron Deficiency Anemia | No Alert | ❌ FP |
| S04 | Viral Upper Respiratory Infection | No Alert | ❌ FP |
| S05 | Early Sepsis from UTI (Urosepsis) | Alert | ✅ Correct |
| S06 | Evolving Anterior STEMI | Alert | ✅ Correct |
| S07 | Atypical DVT with PE Risk | Alert | ✅ Correct |
| S08 | Tension Pneumothorax | Alert | ✅ Correct |
| S09 | Massive Pulmonary Embolism | Alert | ✅ Correct |
| S10 | Stanford Type A Aortic Dissection | Alert | ✅ Correct |

---

## ⚙️ Design Decisions & Trade-offs

### Why One Model as 4 Tools (vs. 4 Separate Models)?

MedGemma 1.5 4B is deployed as 4 specialized tools differentiated entirely through prompt engineering. This approach was chosen because Kaggle T4 x2 GPUs cannot fit multiple large medical models simultaneously. The trade-off is that the Supervisor and Evaluator share the same model weights, limiting the evaluator's ability to independently catch reasoning errors.

**Production alternative:** Use MedGemma 27B for the Supervisor/Evaluator (stronger reasoning) and MedGemma 4B for specialist agents, running on A100/H100 GPUs.

### Why Post-Processing Filters (vs. Pure LLM)?

MedGemma 4B sometimes hallucinates contradictions when all findings are normal. The all-normal safety check is a lightweight rule-based filter that prevents false alerts when labs, imaging, and vitals are all within normal ranges. This hybrid approach (LLM reasoning + rule-based validation) compensates for the smaller model's limitations.

### Why Evaluator-Optimizer Loop (vs. Single Pass)?

The evaluator-optimizer pattern adds a quality assurance layer. However, using the same 4B model for both roles limits its effectiveness. The evaluator catches ~30% of false positives in testing. With a larger evaluator model, this would improve significantly.

### Why FHIR (vs. Free-Text EHR)?

FHIR R4 provides structured patient data that MedGemma can navigate reliably. Free-text clinical notes would require an additional NLP extraction step and introduce more error. FHIR is the industry standard for health data interoperability.

---

## ⚠️ Known Limitations

1. **False positive rate on ambiguous cases** — The Supervisor struggles to distinguish "expected abnormalities" (e.g., anemia in a diagnosed iron deficiency patient) from genuine cross-modal contradictions
2. **Latency (~3-6 min per case on T4)** — Too slow for real-time clinical use; production deployment requires A100+ GPUs with vLLM/TensorRT-LLM for 5-10x throughput
3. **Same-model Evaluator** — Using MedGemma 4B for both Supervisor and Evaluator limits the eval loop's effectiveness
4. **Synthetic test data only** — Not validated on real clinical data; performance on real-world cases with ambiguous findings would likely be lower
5. **No real CXR integration in eval suite** — Evaluation scenarios use text-only (labs + FHIR) due to lack of paired CXR + lab + FHIR test sets

---

## 🔮 Future Improvements

- [ ] **Upgrade to MedGemma 27B** for Supervisor/Evaluator on A100 GPUs
- [ ] **vLLM inference server** for <10s end-to-end pipeline latency
- [ ] **Synthea-generated FHIR bundles** for larger-scale evaluation
- [ ] **Real CXR integration** with paired lab/FHIR data from MIMIC-CXR
- [ ] **Fine-tuned false-positive filter** trained on expected-abnormality patterns
- [ ] **Human-in-the-loop gates** for high-severity alerts before display
- [ ] **FHIR subscription hooks** for real-time EHR integration

---

## ⚕️ Medical Disclaimer

MedSentinel AI is a **research prototype** for the MedGemma Impact Challenge. It is **not a clinical diagnostic tool**, has **not been validated for clinical use**, and **must not be used** to make or replace clinical decisions. All clinical scenarios use synthetic data.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Google Health AI** — MedGemma 1.5 model family
- **LangChain / LangGraph** — Multi-agent orchestration framework
- **Kaggle** — Compute platform and competition hosting
- **Wikimedia Commons** — Sample chest X-ray images (CC0)
