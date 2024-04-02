### Using State Vector as Embedding
SSM Models like Mamba theoretically needs to rely solely on its state vector to store past information, which makes them suitable for embeddings. The state vectors carry special update rules that allow them to be merged to recover collective information, making them suited for retrieval-augmented generation.  
### Baseline 
- **Model**  
We use the realized state vectors of SSM-based pre-trained models such as ``StripedHyena-Hessian-7B`` and ``mamba-2.8b-hf``. These models contain stacked SSM layers. We experiment with each layer's state vector and use the best-performing layer's results in below. 
- **Evaluation**  
We modify the [hagrid](https://huggingface.co/datasets/miracl/hagrid) dataset. It contains around 1500 instances of human annotated (query, supporting documents) pairs. We modify it to contain (query, noisy documents, gold documents) so that embedding quality can be evaluated. We calculate `precision`,`pass@5`,and `pass@10`. Each hagrid dataset instance contains a short query, a set of around 15 documents, at most 3 of which are gold documents relevant to the query. After ranking the documents by their embedding similarity to the query, ``precision`` is the ratio of gold labels in the top n documents, where n is the number of true gold docs in an instance. ``pass@k`` is a boolean flag of whether all gold documents are present in the top k documents. This benchmark is relatively small and superficially easy, used for proof of concept. 
- **Results**  
  
|Model Name | Precision | pass@5 | pass@10|
|---|---|---|---|
|random| 0.10 | 0.22 | 0.53 |  
|*stripedhyena-7b| 0.52 | 0.8 | 0.93 |  
|*mamba-2.8b | 0.96 | 0.99|0.99|  
|m2-bert-80M-8k-retrieval | 0.74 | 0.91 | 0.99|  
|mistral-embed | 1.0 | 1.0 | 1.0|  
|voyage-2| 1.0 | 1.0 | 1.0|  

`*` indicate state vector embed models.

### Merging State Vector for Faster Inference  


