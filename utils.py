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

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=lists)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def load_and_preprocess_data(task, path):
    dataset = load_pickle(path)
    dataset_final = []
    prompt = """
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
    """.strip()

    for i in tqdm(range(len(dataset))):

        counts = dataset[i]["counts"]
        label = dataset[i]["counts"] > 0
        subpat = dataset[i]["subisomorphisms"]
        id = dataset[i]["id"]

        pat_id_list = dataset[i]["pattern"].vs["id"]
        pat_id_list = [int(x) for x in pat_id_list]
        pat_node_labels = dataset[i]["pattern"].vs["label"]
        pat_edges = dataset[i]["pattern"].get_edgelist()

        gra_id_list = dataset[i]["graph"].vs["id"]
        gra_id_list = [int(x) for x in gra_id_list]
        gra_node_labels = dataset[i]["graph"].vs["label"]
        gra_edges = dataset[i]["graph"].get_edgelist()

        if "task-1" in task:
            question = """True represents the presence of subgraphs in the large graph that match the given subgraph; False indicates that there is no such subgraph in a large graph. Please tell me True or False?""".strip()
            source = prompt.format(
                pat_id_list,
                pat_node_labels,
                pat_edges,
                gra_id_list,
                gra_node_labels,
                gra_edges,
                question,
            )
            dataset_final.append(
                {
                    "task": "task-1",
                    "id": id,
                    "input": source,
                    "labels": label,
                    "subisomorphisms": subpat,
                }
            )
        if "task-2" in task:
            question = """
            Please think step by step. How many patterns exist on the graph?
            ### Output Format ###
            There are ... subgraphs that match the given subgraph. They are:... 
            """.strip()
            source = prompt.format(
                pat_id_list,
                pat_node_labels,
                pat_edges,
                gra_id_list,
                gra_node_labels,
                gra_edges,
                question,
            )
            dataset_final.append(
                {
                    "task": "task-2",
                    "id": id,
                    "input": source,
                    "counts": counts,
                    "subisomorphisms": subpat,
                }
            )

    return dataset_final


def validate_prompt_information(prompt_text):
    """
    Validate whether prompt contains necessary graph information for both query subgraph and target graph.

    Args:
        prompt_text: prompt text content

    Returns:
        tuple: (is_valid, missing_info)
            - is_valid: bool, whether validation passed
            - missing_info: list, list of missing information
    """
    missing_info = []

    def extract_all_content(text, start_tag, end_tag):
        """Extract all occurrences of content between start_tag and end_tag"""
        contents = []
        search_start = 0
        while True:
            start_idx = text.find(start_tag, search_start)
            if start_idx == -1:
                break
            end_idx = text.find(end_tag, start_idx)
            if end_idx == -1:
                break
            content = text[start_idx + len(start_tag):end_idx].strip()
            if content:
                contents.append(content)
            search_start = end_idx + len(end_tag)
        return contents

    # Extract all label, edge, and index contents
    labels = extract_all_content(prompt_text, '<label>', '</label>')
    edges = extract_all_content(prompt_text, '<edge>', '</edge>')
    indices = extract_all_content(prompt_text, '<index>', '</index>')

    # Validate query subgraph (first occurrence)
    if len(labels) < 1:
        missing_info.append('Query subgraph node label information (<label> tag)')
    elif not labels[0]:
        missing_info.append('Query subgraph node label is empty')

    if len(edges) < 1:
        missing_info.append('Query subgraph edge list information (<edge> tag)')
    elif not edges[0]:
        missing_info.append('Query subgraph edge list is empty')

    if len(indices) < 1:
        missing_info.append('Query subgraph node index information (<index> tag)')
    elif not indices[0]:
        missing_info.append('Query subgraph node index is empty')

    # Validate target graph (second occurrence)
    if len(labels) < 2:
        missing_info.append('Target graph node label information (<label> tag)')
    elif not labels[1]:
        missing_info.append('Target graph node label is empty')

    if len(edges) < 2:
        missing_info.append('Target graph edge list information (<edge> tag)')
    elif not edges[1]:
        missing_info.append('Target graph edge list is empty')

    if len(indices) < 2:
        missing_info.append('Target graph node index information (<index> tag)')
    elif not indices[1]:
        missing_info.append('Target graph node index is empty')

    is_valid = len(missing_info) == 0
    return is_valid, missing_info


def prompt_confirmation(prompt_text, task_id, item_id, verbose=True):
    """
    Prompt user to confirm prompt information completeness.

    Args:
        prompt_text: prompt text
        task_id: task ID
        item_id: data item ID
        verbose: bool, whether to print detailed information

    Returns:
        bool, whether user confirms to continue
    """
    is_valid, missing_info = validate_prompt_information(prompt_text)

    if is_valid:
        if verbose:
            print(f"✓ [Task {task_id} - ID {item_id}] Information validation passed!")
        return True
    else:
        print(f"\n⚠️  [Task {task_id} - ID {item_id}] Information validation failed!")
        print("Missing information:")
        for i, info in enumerate(missing_info, 1):
            print(f"  {i}. {info}")
        print()

        if verbose:
            response = input("Continue sending prompt to the model? (yes/no): ").strip().lower()
            if response in ['yes', 'y']:
                print("User chose to continue.\n")
                return True
            else:
                print("Skipped this item. Please complete the information and try again.\n")
                return False
        return False


def generate(val_dataset, tokenizer, model, validate=False):
    output1 = []
    output2 = []
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(len(val_dataset))):
            prompt_text = val_dataset[i]["input"]
            task_id = val_dataset[i]["task"]
            item_id = val_dataset[i]["id"]

            # Validate prompt information if requested
            if validate and not prompt_confirmation(prompt_text, task_id, item_id, verbose=True):
                continue

            messages = [
                {"role": "user", "content": prompt_text},
            ]

            input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(model.device)

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]

            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=5000,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
            )
            response = outputs[0][input_ids.shape[-1] :]
            response_text = tokenizer.decode(response, skip_special_tokens=True)

            if val_dataset[i]["task"] == "task-1":
                output1.append(
                    {
                        "id": val_dataset[i]["id"],
                        "input": val_dataset[i]["input"],
                        "output": response_text,
                        "label": val_dataset[i]["labels"],
                        "subgraph": val_dataset[i]["subisomorphisms"],
                    }
                )

            elif val_dataset[i]["task"] == "task-2":
                output2.append(
                    {
                        "id": val_dataset[i]["id"],
                        "input": val_dataset[i]["input"],
                        "output": response_text,
                        "label": val_dataset[i]["counts"],
                        "subgraph": val_dataset[i]["subisomorphisms"],
                    }
                )

            else:
                raise ImportError("Running a task that does not exist.")

    return output1, output2
