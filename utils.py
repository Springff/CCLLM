
import os
import torch
from tqdm import tqdm
import pickle
import csv

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

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

def load_and_preprocess_data(task,path):
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
    You task is to determine whether the provided query subgraph exists within the target graph. Please note that both the target graph and query subgraph are undirected, meaning edge directionality is irrelevant, that is, (a, b) is equivalent to (b, a). Please think step by step, carefully analyze the graph structures to identify all matching subgraphs. {}? 
    """

    for i in tqdm(range(len(dataset))):

        counts = dataset[i]['counts']
        label = dataset[i]['counts'] > 0
        subpat = dataset[i]['subisomorphisms']
        id = dataset[i]['id']


        pat_id_list = dataset[i]['pattern'].vs['id']
        pat_id_list = [int(x) for x in pat_id_list]
        pat_node_labels = dataset[i]['pattern'].vs['label']
        pat_edges = dataset[i]['pattern'].get_edgelist()  


        gra_id_list = dataset[i]['graph'].vs['id']
        gra_id_list = [int(x) for x in gra_id_list]
        gra_node_labels = dataset[i]['graph'].vs['label']
        gra_edges = dataset[i]['graph'].get_edgelist()  

       
        if '1-3' in task: 
            question = "True represents the presence of subgraphs in the large graph that match the given subgraph; False indicates that there is no such subgraph in a large graph. Please tell me True or False?"
            source = prompt.format(pat_id_list,pat_node_labels,pat_edges,gra_id_list,gra_node_labels,gra_edges,question)
            dataset_final.append({
                "task": '1-3',
                "id": id,
                "input": source,
                "labels":label,
                "subisomorphisms":subpat,
            })
        if '2-5' in task: 
            question = """
            Please think step by step. How many patterns exist on the graph?
            ### Output Format ###
            There are ... subgraphs that match the given subgraph. They are:... 
            """
            source = prompt.format(pat_id_list,pat_node_labels,pat_edges,gra_id_list,gra_node_labels,gra_edges,question)
            dataset_final.append({
                "task": '2-5',
                "id": id,
                "input": source,
                "counts":counts,
                "subisomorphisms":subpat,
            })   

    return dataset_final

def generate(val_dataset,tokenizer,model):   
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
                output1.append({
                "id":val_dataset[i]['id'],
                "input": val_dataset[i]['input'],
                "output":response_text,
                "label": val_dataset[i]['labels'],
                "subgraph":val_dataset[i]['subisomorphisms']
                })

            elif val_dataset[i]['task']=='2-5':                                
                output2.append({                
                "id":val_dataset[i]['id'],
                "input": val_dataset[i]['input'],
                "output":response_text,
                "label": val_dataset[i]['counts'],
                "subgraph":val_dataset[i]['subisomorphisms']
                })

            else:
                raise ImportError('Running a task that does not exist.') 


    return output1,output2   

