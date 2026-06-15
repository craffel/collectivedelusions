# Real-World BERT-Tiny GLUE Sequence Classification Evaluation

We evaluated the ensembling models on a real-world multi-task sequence classification stream using `prajjwal1/bert-tiny` with PEFT LoRA adapters fine-tuned on SST-2, MRPC, and CoLA.

| Ensembling Method | Downstream Sequence Accuracy (%) |
| :--- | :---: |
| Uniform | 61.08% |
| SABLE | 60.25% |
| PAC-Kinetics | 60.25% |
| Softmax (Static) | 60.08% |
| MLP (Static) | 61.00% |
| GRU Router | 61.42% |
| LVCS | 61.25% |

