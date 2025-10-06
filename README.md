
This repository contains the necessary code to reproduce the results of our submitted paper to EACL 2026 ***"Evidential Semantic Entropy for LLM Uncertainty Quantification"***.

This code is build on [kernel language entropy codebase](https://github.com/AlexanderVNikitin/kernel-language-entropy), which itself is build on [Semantic Uncertainty codebase](https://github.com/jlko/semantic_uncertainty/tree/master). 


**Installation**

   ```bash
   python3 -m venv path/to/new/EvSemE
   source path/to/new/EvSemE/bin/activate  
   pip install -r requirements.txt
   ```

**Experiment command**

```
python generate_answers.py --num_samples=500 --model_name=$MODEL --dataset=$DATASET --num_generations=5 --random_seed=42 --compute_kle $EXTRA_CFG
```

* `$MODEL` is one of `[Llama-2-7b, falcon-40b-instruct, Mistral-7B-v0.1, Mistral-7B-Instruct-v0.1]`
* `$DATASET` is one of `[trivia_qa, squad, nq, svamp]`
* EXTRA_CFGS is either `""` for short sequence setting and `"--num_few_shot=0 --model_max_new_tokens=100 --brief_prompt=chat --metric=llm --metric_model=Meta-Llama-3-8B-Instruct --no-compute_accuracy_at_all_temps"` for long sequence setting. 


