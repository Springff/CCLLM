import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from tqdm import tqdm
import pickle
import re
import csv

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def save_to_csv(data, filename, lists):
    directory = os.path.dirname(filename)

    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=lists)
        
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def get_preprocessed_data(task,path):
    dataset = load_pickle(path)
    dataset_final = []
    prompt="""
### System Instruction ###
You are an expert in subgraph matching, adept at precisely locating query subgraphs within target graphs. Your expertise lies in exact subgraph matching, requiring both topological consistency and identical node labels between the query and matched subgraph. I will provide you with detailed information about both the target graph and the query subgraph.
### Graph Structure ###
<query subgraph>
<index> {} </index>
<label> {} </label>
<edge> {} </edge>
</query subgraph >
<target graph>
<index> {} </index>
<label> {} </label>
<edge> {} </edge>
</target graph>
### Task Description ###
You task is to determine whether the provided query subgraph exists within the target graph. Please note that both the target graph and query subgraph are undirected, meaning edge directionality is irrelevant, that is, (a, b) is equivalent to (b, a). Please think step by step, carefully analyze the graph structures to identify all matching subgraphs. How many patterns exist on the graph? 
### Output Format ###
There are ... subgraphs that match the given subgraph. They are:... 
"""
    for i in tqdm(range(len(dataset))):
    # for i in tqdm(range(1)):
        # print(len(dataset))
        counts = dataset[i]['counts']
        #print(dataset[i]['subisomorphisms'])
        #print(counts)
        label = dataset[i]['counts'] > 0
        subpat = dataset[i]['subisomorphisms']
        # 节点数量
        
        #print(num_nodes)
        pat_id_list = dataset[i]['pattern'].vs['id']
        pat_id_list = [int(x) for x in pat_id_list]
        
        
        # 边的信息
        pat_edges = dataset[i]['pattern'].get_edgelist()  # 边的信息类似于 (节点1, 节点2)
        #print(edges)


        # 节点标签数据
        pat_node_labels = dataset[i]['pattern'].vs['label']
        #print(node_labels_data)

        # 节点数量
        
        #print(num_nodes)
        gra_id_list = dataset[i]['graph'].vs['id']
        gra_id_list = [int(x) for x in gra_id_list]
        
        
        # 边的信息
        gra_edges = dataset[i]['graph'].get_edgelist()  # 边的信息类似于 (节点1, 节点2)
        #print(edges)
        

        # 节点标签数据
        gra_node_labels = dataset[i]['graph'].vs['label']
        #print(node_labels_data)

        id = dataset[i]['id']

        # 打印节点标签字典
        #print(node_labels)
        source = prompt.format(pat_id_list,pat_node_labels,pat_edges,gra_id_list,gra_node_labels,gra_edges)
        
        
        if '1-3' in task: 
            question = "True represents the presence of subgraphs in the large graph that match the given subgraph; False indicates that there is no such subgraph in a large graph. Please tell me True or False?"
            source_text = source + question
            dataset_final.append({
                "task": '1-3',
                "id": id,
                "input": source_text,
                "labels":label,
                "subisomorphisms":subpat,
            })
            #print(source_text)
        if '2-5' in task: 
            question = "Please think step by step. How many patterns exist on the graph?"
            source_text = source + question
            dataset_final.append({
                "task": '2-5',
                "input": source,
                "id": id,
                "counts":counts,
                "subisomorphisms":subpat,
            })
            #print(source_text)

    #random.shuffle(dataset_final)    

    return dataset_final

def evaluate_epoch(val_dataset,tokenizer,model):   
    ACC={}
    ACC['1-3']=0
    ACC['2-5']=0
    output1= []
    output2= []
    model.eval()

    with torch.no_grad():
        
        for i in tqdm(range(len(val_dataset))):  
            torch.cuda.empty_cache()
            messages = [
                {"role": "user", "content": val_dataset[i]['input']},
            ]

            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            

            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=5000,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
            )   
            response = outputs[0][input_ids.shape[-1]:]
            response_text = tokenizer.decode(response, skip_special_tokens=True)

            if val_dataset[i]['task']=='1-3':
                modified_text = response_text[-5:].strip()
                label = val_dataset[i]['labels'] 
                label = str(label)
                subgraph = val_dataset[i]['subisomorphisms']
                id = val_dataset[i]['id'] 
                if modified_text==label:                
                    ACC['1-3']+=1
                    print(True)
                else:
                    # print('Failed to predict True/False:')
                    # print(val_dataset[i]['input'])
                    print(response_text)
                    # print(label)
                    
                output1.append({
                "id":id,
                "predict":response_text,
                "label": label,
                "subgraph":subgraph
                })



            elif val_dataset[i]['task']=='2-5':
                
                modified_text = re.search(r"There are (\d+)", response_text)
                modified_text1 = re.search(r"There is (\d+)", response_text)
                counts = val_dataset[i]['counts'] 
                subpat = val_dataset[i]['subisomorphisms']
                id = val_dataset[i]['id'] 
                counts = str(counts)
                if modified_text:
                    if modified_text.group(1)==counts:                
                        print(True)
                        ACC['2-5']+=1
                    else:
                        print('Failed to predict Num:')
                        print(val_dataset[i]['input'])
                        print(response_text)
                        print(modified_text)
                        print(modified_text.group(1))
                        print(subpat)
                        print(counts)
                    output2.append({
                    "id":id,
                    "predict":modified_text.group(1),
                    "output":response_text,
                    "label": counts,
                    "subgraph":subpat,
                    })
                elif modified_text1:
                    if modified_text1.group(1)==counts:                
                        print(True)
                        ACC['2-5']+=1
                    else:
                        print('Failed to predict Num:')
                        print(val_dataset[i]['input'])
                        print(response_text)
                        print(modified_text1)
                        print(modified_text1.group(1))
                        print(subpat)
                        print(counts)
                    output2.append({
                    "id":id,
                    "predict":modified_text1.group(1),
                    "output":response_text,
                    "label": counts,
                    "subgraph":subpat,
                    })
                else:                   
                    print(response_text)
                    output2.append({
                    "id":id,
                    "predict":'',
                    "output":response_text,
                    "label": counts,
                    "subgraph":subpat,
                    })

            else:
                raise ImportError('Running a task that does not exist.') 
    num = len(val_dataset)


    ACC['1-3'] = ACC['1-3']/num
    ACC['2-5'] = ACC['2-5']/num

    return ACC,output1,output2   


model_id = "/home/chaichunyang/InstructGLM/llama3"

tokenizer = AutoTokenizer.from_pretrained(model_id)


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
# model =AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map=0, torch_dtype=torch.float16)

output_dir = "/home/chaichunyang/InstructGLM/llama3-output-mid"

model = PeftModel.from_pretrained(model, output_dir)
# '1-3',
paths = ['/home/chaichunyang/InstructGLM/data/test_data/200/G2_test_NL8_N16.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/G2_test_NL8_N32.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/G2_test_NL16_N16.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/G2_test_NL16_N32.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/G2_test_NL32_N16.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/G2_test_NL32_N32.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/G7_test_NL8_N16.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/G7_test_NL8_N32.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/G7_test_NL16_N16.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/G7_test_NL16_N32.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/G7_test_NL32_N16.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/G7_test_NL32_N32.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/G18_test_NL8_N16.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/G18_test_NL8_N32.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/G18_test_NL16_N16.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/G18_test_NL16_N32.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/G18_test_NL32_N16.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/G18_test_NL32_N32.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/G24_test_NL8_N16.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/G24_test_NL8_N32.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/G24_test_NL16_N16.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/G24_test_NL16_N32.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/G24_test_NL32_N16.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/G24_test_NL32_N32.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/6order_test_NL8_N16.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/6order_test_NL8_N32.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/6order_test_NL16_N16.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/6order_test_NL16_N32.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/6order_test_NL32_N16.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/6order_test_NL32_N32.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/7order_test_NL8_N32.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/7order_test_NL8_N64.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/7order_test_NL16_N32.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/7order_test_NL16_N64.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/7order_test_NL32_N32.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/7order_test_NL32_N64.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/8order_test_NL8_N32.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/8order_test_NL8_N64.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/8order_test_NL16_N32.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/8order_test_NL16_N64.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/8order_test_NL32_N32.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/8order_test_NL32_N64.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/9order_test_NL8_N32.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/9order_test_NL8_N64.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/9order_test_NL16_N32.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/9order_test_NL16_N64.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/9order_test_NL32_N32.pkl', '/home/chaichunyang/InstructGLM/data/test_data/200/9order_test_NL32_N64.pkl']
for i in range(0,12):
    task = {'2-5'}
    path = paths[i]
    val_dataset = get_preprocessed_data(task,path)


    accurary,output1,output2 = evaluate_epoch(val_dataset,tokenizer,model)

    print(path)    
    print(accurary)         

    # list1 = ['id', 'predict','output', 'label', 'subgraph']
    # save_to_csv(output1, '/home/chaichunyang/InstructGLM/Result/Result_xr/'+str(i)+'_test1.csv', list1)
    list2 = ['id', 'predict','output', 'label', 'subgraph']
    save_to_csv(output2, '/home/chaichunyang/InstructGLM/Result/all/mid/CCLLM_mid_'+str(i)+'.csv', list2)




