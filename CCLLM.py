from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

from utils import *

model_id = "./llama3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Loading Llama3 model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

output_dir = "./CCLLM-Llama3"
# Loading the LoRA module
model = PeftModel.from_pretrained(model, output_dir)

task = {'1-3','2-5'}
path = "./Data/test_data.pkl"
val_dataset = load_and_preprocess_data(task,path)

# output1 is the result of predicting whether CC motifs exists
# output2 is the result of predicting the number of CC motifs
output1,output2 = generate(val_dataset,tokenizer,model)

Column1 = ['id', 'input','output', 'label', 'subgraph']
save_to_csv(output1, './output1.csv', Column1)
Column2 = ['id', 'input','output', 'label', 'subgraph']
save_to_csv(output2, './output2.csv', Column2)




