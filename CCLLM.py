from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

from utils import *

model_id = "./Llama3_models"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

output_dir = "./CCLLM-Llama3"
model = PeftModel.from_pretrained(model, output_dir)

task = ["task-1", "task-2"]
path = "./Data/test_data.pkl"
val_dataset = load_and_preprocess_data(task, path)

# output1 is the result of predicting whether CC motifs exists
# output2 is the result of predicting the number of CC motifs
output1, output2 = generate(val_dataset, tokenizer, model)

Column = ["id", "input", "output", "label", "subgraph"]
save_to_csv(output1, "./output1.csv", Column)
save_to_csv(output2, "./output2.csv", Column)
