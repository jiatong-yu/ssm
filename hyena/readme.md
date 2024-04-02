### Using StrpedHyena-7b  
To run the artificially easy embedding evaluation dataset, run ``python hagrid_eval.py``. Modify it to use a custom number of data points. 
|Model Name | Precision | pass@5 | pass@10|
|---|---|---|---|
|random| 0.1 | 0.22 | 0.53 |  
|**SH-7b**| **0.52** | **0.8** | **0.93** |  
|mamba-2.8b | 0.96 | 0.994|0.996|  
|m2-bert-80M-8k-retrieval | 0.74 | 0.913 | 0.996|  
|mistral-embed | 1.0 | 1.0 | 1.0|  
|voyage-2| 1.0 | 1.0 | 1.0|
