import json
import random
import json
import os
import argparse
import torch
import datasets
from tqdm import tqdm
import time 

from math import comb
from transformers import AutoTokenizer
import threading
import together
import voyageai

from mistralai.client import MistralClient


from mamba_ssm.models.config_mamba import MambaConfig
from model import MambaEmbedModel

SAVE_EMBEDDING = True

together_client = together.Together()
together.api_key = "YOUR API KEY"
mistral_client = MistralClient(api_key="YOUR API KEY")
voyage_client = voyageai.Client(api_key="YOUR API KEY")



def _cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    source: https://github.com/embeddings-benchmark/mteb
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def _process_pos_instance(inst, max_docs=3):
    pos_docs_idxs = []
    for ans in inst['answers']:
        for sent_ist in ans['sentences']:
            if not sent_ist['index'] in pos_docs_idxs:
                pos_docs_idxs.append(sent_ist['index'])
    if len(pos_docs_idxs) > len(inst['quotes']): # this dataset is a bit flawed...
        # print("passing an illegal instance...")
        return None, None

    if len(pos_docs_idxs) > max_docs: 
        pos_docs_idxs = pos_docs_idxs[:max_docs]
    
    pos_docs = []
    pos_docs_docids = []
    for i in pos_docs_idxs:
        if i >= len(inst['quotes']):
            # print('passing an illegal instance...')
            return None, None
        assert inst['quotes'][i]['docid'] not in pos_docs_docids
        pos_docs_docids.append(inst['quotes'][i]['docid'])
        pos_docs.append(inst['quotes'][i]['text'])

    return pos_docs, pos_docs_docids

def _process_neg_instance(pos_idx, pos_docs_docids, hagrid, N = 15):
    neg_docs = []
    neg_docs_docids = []
    while len(neg_docs) < N - len(pos_docs_docids): 
        neg_idx = random.randint(0, len(hagrid)-1)
        if neg_idx == pos_idx:
            continue 
        j = random.randint(0, len(hagrid[neg_idx]['quotes'])-1)
        try:
            neg_docid = hagrid[neg_idx]['quotes'][j]['docid']
        except:
            continue
        if neg_docid not in neg_docs_docids:
            neg_docs_docids.append(neg_docid)
            neg_docs.append(hagrid[neg_idx]['quotes'][j]['text'])
    return neg_docs, neg_docs_docids

def _create_hagrid_instance(query,pos_docs, pos_docs_docids, neg_docs, neg_docs_docids):
    docs = []
    for doc, docid in zip(pos_docs, pos_docs_docids):
        docs.append({"text": doc, "docid": docid})
    for doc, docid in zip(neg_docs, neg_docs_docids):
        docs.append({"text": doc, "docid": docid})
    return {"query": query, "docs": docs, "golds": pos_docs_docids}

def _compute_random_baseline(N):
    # compute expected precison score  
    print("N = ", N)
    with open(args.output_path, "r") as f:
        dataset = json.load(f)
    
    avg_precision = 0
    avg_pass_5 = 0
    avg_pass_10 = 0
    avg_n = 0
    for inst in dataset: 
        n = len(inst['golds'])
        avg_n += n
        avg_precision += n / N
        avg_pass_5 += comb(N-n,5-n)/comb(N,5)
        avg_pass_10 += comb(N-n,10-n)/comb(N,10)
    avg_precision /= len(dataset)
    avg_pass_5 /= len(dataset)
    avg_pass_10 /= len(dataset)

    print("Random Baseline:")
    print("avg_n: ", avg_n / len(dataset))
    print("Precision: ", avg_precision)
    print("Pass@5: ", avg_pass_5)
    print("Pass@10: ", avg_pass_10)
        

    

def _create_hagrid_dataset(N = 15,
                          dataset_path = "hagrid_dataset.json",
                          ):
    if N <= 10:
        raise ValueError("N should be greater than 10, since we use pass@10 score.")
    hagrid = hagrid = datasets.load_dataset("miracl/hagrid", 
                                            trust_remote_code=True,
                                            split="train")
    dataset = []
    for inst_idx, inst in enumerate(hagrid):
        query = inst['query']
        pos_docs, pos_docs_docids = _process_pos_instance(inst)
        if pos_docs is None:
            continue

        neg_docs, neg_docs_docids = _process_neg_instance(inst_idx, pos_docs_docids, hagrid, N=N)

        data_inst = _create_hagrid_instance(query,
                                            pos_docs, 
                                            pos_docs_docids, 
                                            neg_docs, 
                                            neg_docs_docids)
        dataset.append(data_inst)
    with open(dataset_path,"w") as f: 
        json.dump(dataset, f, indent=4)
    return 

def _generate_together_embedding(client, sents, model_api = "togethercomputer/m2-bert-80M-8k-retrieval"): 
    if isinstance(sents, str):
        sents_list = [sents]
    else:
        sents_list = sents
    out = client.embeddings.create(
        input = sents_list,
        model = model_api
    )
    time.sleep(1)
    if isinstance(sents, str):
        return torch.tensor(out.data[0].embedding)
    
    emb_list = []
    for i in range(len(sents_list)):
        emb_list.append(
            torch.tensor(out.data[i].embedding)
        )
    return emb_list

def _generate_mistral_embedding(client, sents,):
    if isinstance(sents, str):
        sents_list = [sents]
    else:
        sents_list = sents
    out = client.embeddings(
        input = sents_list,
        model = "mistral-embed"
    )
    time.sleep(1)
    if isinstance(sents, str):
        return torch.tensor(out.data[0].embedding)
    emb_list = []
    for i in range(len(sents_list)):
        emb_list.append(
            torch.tensor(out.data[i].embedding)
        )
    return emb_list

def _generate_voyage_embedding(client, sents):
    if isinstance(sents, str):
        sents_list = [sents]
    else:
        sents_list = sents
    out = client.embed(
        sents_list,
        model = "voyage-2"
    )
    time.sleep(1)
    if isinstance(sents, str):
        return torch.tensor(out.embeddings[0])
    emb_list = []
    for i in range(len(sents_list)):
        emb_list.append(
            torch.tensor(out.embeddings[i])
        )
    return emb_list

def _evaluate_instance(args, model, inst, layer_idx=-5):
    if args.model == "MambaEmbed":
        query_embedding = model.encode(inst['query'], layer_idx=layer_idx)[0]
        results = []
        if SAVE_EMBEDDING:
            doc_embeddings = []
        for doc in inst['docs']:
            if args.model == "MambaEmbed":
                emb = model.encode(doc['text'], layer_idx = layer_idx)[0]

            score = _cos_sim(query_embedding, emb)
            docid = doc['docid']
            results.append((score, docid))
            if SAVE_EMBEDDING:
                doc_embeddings.append({
                    "doc": doc['text'],
                    "embedding": emb.tolist(),
                    "docid": docid,
                })
        if SAVE_EMBEDDING:
            emb_inst = {
                "query": inst['query'],
                "query_embedding": query_embedding.tolist(),
                "docs": doc_embeddings,
            }
            if os.path.exists("hagrid_embedding.json"):
                with open("hagrid_embedding.json","r") as f:
                    data = json.load(f)
            else:
                data = []
            data.append(emb_inst)
            with open("hagrid_embedding.json", "w") as f:
                json.dump(data, f, indent=4)

    
    elif args.model == "together":
        query_embedding = _generate_together_embedding(together_client, inst['query'])
        doc_embeddings = _generate_together_embedding(together_client, [doc['text'] for doc in inst['docs']])
        results = []
        for emb, doc in zip(doc_embeddings, inst['docs']):
            score = _cos_sim(query_embedding, emb)
            docid = doc['docid']
            results.append((score, docid))
        
    elif args.model == "mistral":
        query_embedding = _generate_mistral_embedding(mistral_client, inst['query'])
        doc_embeddings = _generate_mistral_embedding(mistral_client, [doc['text'] for doc in inst['docs']])
        results = []
        for emb, doc in zip(doc_embeddings, inst['docs']):
            score = _cos_sim(query_embedding, emb)
            docid = doc['docid']
            results.append((score, docid))

    elif args.model == "voyage":
        query_embedding = _generate_voyage_embedding(voyage_client, inst['query'])
        doc_embeddings = _generate_voyage_embedding(voyage_client, [doc['text'] for doc in inst['docs']])
        results = []
        for emb, doc in zip(doc_embeddings, inst['docs']):
            score = _cos_sim(query_embedding, emb)
            docid = doc['docid']
            results.append((score, docid))

    results = sorted(results, key=lambda x: x[0], reverse=True)
    """
    Precision score. 
    Let n be the number of gold documents in the given instance. Precision compute the percentage of true positives in the top n documents. 
    """
    n = len(inst['golds'])
    top_n = [docid for _, docid in results[:n]]
    true_pos = 0
    for docid in top_n:
        if docid in inst['golds']:
            true_pos += 1
    precision = true_pos / n

    """
    Pass@5 score.
    Return 1 if all gold documents are in the top 5 documents, 0 otherwise.
    """
    top_5 = [docid for _, docid in results[:5]]
    pass_5 = 1 if all([goldid in top_5 for goldid in inst['golds']]) else 0

    """
    Pass@10 score."""
    top_10 = [docid for _, docid in results[:10]]
    pass_10 = 1 if all([goldid in top_10 for goldid in inst['golds']]) else 0

    return precision, pass_5, pass_10
    

def evaluate(args):
    if SAVE_EMBEDDING:
        print("warning: saving embeddings. this could be significant in memory usage and slow down the evaluation process. please make sure the dataset size is small.")
    with open(args.dataset_path, "r") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        raise ValueError("Dataset should be a list of dictionaries, with fields 'query', 'docs', and 'golds'")
    
    dataset = dataset[:10]
    if args.model == "MambaEmbed":
        model = MambaEmbedModel.from_pretrained("state-spaces/mamba-2.8b", 
                                                device="cuda",
                                                dtype=torch.float16)
        model.eval()
    else:
        model = None 

    precision_scores = []
    pass_5_scores = []
    pass_10_scores = []
    for inst in tqdm(dataset):
        precision, pass_5, pass_10 = _evaluate_instance(args, model, inst, args.layer_idx)
        precision_scores.append(precision)
        pass_5_scores.append(pass_5)
        pass_10_scores.append(pass_10)
    print("Precision: ", sum(precision_scores) / len(precision_scores))
    print("Pass@5: ", sum(pass_5_scores) / len(pass_5_scores))
    print("Pass@10: ", sum(pass_10_scores) / len(pass_10_scores))
    res = {
        "layer_idx": args.layer_idx,
        "Precision": sum(precision_scores) / len(precision_scores),
        "Pass@5": sum(pass_5_scores) / len(pass_5_scores),
        "Pass@10": sum(pass_10_scores) / len(pass_10_scores),
    }
    # lock = threading.Lock()
    # with lock:
    output_path = args.output_path+"_"+args.model+".json"
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            data = json.load(f)
    else: 
        data = []
    data.append(res)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=15)
    parser.add_argument("--dataset_path", type=str, default="hagrid_dataset.json")
    parser.add_argument("--output_path", type=str, default="hagrid_eval")
    parser.add_argument("--random_baseline", default=False, action="store_true")
    parser.add_argument("--model", type=str, default="MambaEmbed", choices=["MambaEmbed", "together", "mistral", "voyage"])
    parser.add_argument("--layer_idx", type=int, default=-8)
    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        _create_hagrid_dataset(N=args.N, dataset_path=args.dataset_path)
        print("created hagrid dataset at ", args.dataset_path,". Please run the script again to evaluate.")
    
    
    elif args.random_baseline:
        _compute_random_baseline(args.N)

    else: 
        evaluate(args)