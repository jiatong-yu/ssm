import json
import random
import json
import os
import argparse
import torch
import datasets
from tqdm import tqdm

from math import comb
from transformers import AutoTokenizer

from mamba_ssm.models.config_mamba import MambaConfig
from model import MambaEmbedModel

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
                          output_path = "hagrid_dataset.json",
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
    with open(output_path,"w") as f: 
        json.dump(dataset, f, indent=4)
    return 

def _evaluate_instance(model, inst):
    query_embedding = model.encode(inst['query'])[0]
    results = []
    for doc in inst['docs']:
        emb = model.encode(doc['text'])[0]

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
    with open(args.output_path, "r") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        raise ValueError("Dataset should be a list of dictionaries, with fields 'query', 'docs', and 'golds'")
    
    dataset = dataset[:500]
    model = MambaEmbedModel.from_pretrained("state-spaces/mamba-2.8b", 
                                            device="cuda",
                                            dtype=torch.float16)
    model.eval()
    precision_scores = []
    pass_5_scores = []
    pass_10_scores = []
    for inst in tqdm(dataset):
        precision, pass_5, pass_10 = _evaluate_instance(model, inst)
        precision_scores.append(precision)
        pass_5_scores.append(pass_5)
        pass_10_scores.append(pass_10)
    print("Precision: ", sum(precision_scores) / len(precision_scores))
    print("Pass@5: ", sum(pass_5_scores) / len(pass_5_scores))
    print("Pass@10: ", sum(pass_10_scores) / len(pass_10_scores))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=15)
    parser.add_argument("--output_path", type=str, default="hagrid_dataset.json")
    parser.add_argument("--random_baseline", default=False, action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        _create_hagrid_dataset(N=args.N, output_path=args.output_path)
        print("created hagrid dataset at ", args.output_path,". Please run the script again to evaluate.")
    
    elif args.random_baseline:
        _compute_random_baseline(args.N)
    else: 
        evaluate(args)