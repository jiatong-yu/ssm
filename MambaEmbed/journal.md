# Research Journal
### Using Mamba State Vector as Embedding 
SSM Models like Mamba theoretically needs to rely solely on its state vector to store past information, which makes them suitable for embeddings. The state vectors are generally smaller, and could be used to further train an reranker that takes SSM state vectors as inputs.   
### Baseline 
- **Model**  
We use pre-trained `mamba-2.8b` model and use its last layer's state vector. It has `tokenizer.vocab_size` number of ssm channels, each of size 16. The channels are concatenated as final output.   
- **Evaluation**  
We modify the [hagrid](https://huggingface.co/datasets/miracl/hagrid) dataset. It contains around 1500 instances of human annotated (query, supporting documents) pairs. We modify it to contain (query, noisy documents, gold documents) so that embedding quality can be evaluated. We calculate `precision`,`pass@5`,and `pass@10`. 
- **Results**  
The dataset contains 1324 instances. Each instance has at most 3 gold documents and an average of 1.61 gold documents. Each instance has a total of 15 docs. The rest (negative documents) are obtained from the original supporting docs of other queries.  
Random has precision = 0.1, pass@5 = 0.22, pass@10 = 0.53.   
Baseline has precision = 0.51, pass@5 = 0.72, pass@10 = 0.91  
  

