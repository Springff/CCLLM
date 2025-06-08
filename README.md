# CCLLM 

<img src="./CCLLM.png">


## Overview


CCLLM is a graph-based LLM framework designed to identify conservative topological patterns of cell type combinations as CC motifs from cellular communities in SRT data. 


CCLLM constructs a cellular community by cell spatial coordinates, where nodes represent individual cells and edges represent hypothetical spatial relations. The graph structure of the cellular community is then encoded a descriptive prompt template consisting of system instruction, graph structure, task description, and output format. CCLLM leverages prompt engineering to capture contextual information from the cellular community, and is fine-tuned using Low-Rank Adaptation (LoRA) to enable CC motifs identification and provide end-to-end biological interpretation.



## Quick Start

These instructions guide you on how to run CCLLM locally.

### Step 1: Clone the Repository

```bash
git clone https://github.com/Springff/CCLLM.git
cd CCLLM
```
### Step 2: Configure the Environment

Create and activate a conda environment:
```bash
conda create -n myenv python=3.9
conda activate myenv
```

Install dependencies:
```bash
pip install -r environment.txt
```



### Step 3: Download the Llama3 Model

>Important: You need to obtain a license for the Llama3 model and follow Meta's official instructions for downloading and configuring it.

Hugging Face: [Llama3-8b-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)


Please consult the official Llama3 documentation for the correct download and usage instructions.

Place the downloaded Llama3 model in the llama3_models directory (recommended), or modify the model path in llama3_0.py.

### Step 4: Run
```bash
python ./CCLLM.py
```




## Project Structure 
```
CCLLM/
├── llama3_0.py             # Main execution script
├── environment.txt         # Python dependency file
├── README.md               # This document
├── llama3_models/           # (Optional) Directory for Llama3 models
│   └──  (Llama3 model files)
└── ...
```
