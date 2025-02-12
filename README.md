# Tiny Specialized Models
Can tiny specialized models outperform frontier LLMs on very specific tasks? Yes, of course!

While the world pursues general intelligence with large models, we take a step in the other direction -- towards tiny specialized models, that serve a particular outcome, and run at a fraction of the cost.

We begin by exploring a proof-of-concept on the differentiation task, i.e. find the derivative $\frac{d}{dx} f(x)$ of a given expression $f(x)$. For example,

<div style="border: 1px solid; border-radius: 8px; padding: 12px; margin: 0px 0;">

**User:** _Find the derivative of_ $3 e^{2x} \sin(14 x)$

**Assistant:** _Think think think..._ <br>
_The final answer is_ $\boxed{6 e^{2x} (\sin(14 x) + 7 \cos(14 x))}$

</div> <br>


Our goal is to find the **_smallest_** model that can solve this simple, yet interesting, reasoning task.

> 📄 **Paper**: Coming soon!

### Models
We release 4 differentiation models, _Diff-TSMs_, spanning 135M to 1.5B parameters, trained using a combination of supervised finetuning (SFT) and reinforcement learning (RL). More details to follow in the paper and training code.

| Link| Model Name |  Base Model |
| :-: |----------------------|------------|
| [🤗](https://huggingface.co/AvataarAI/Diff-TSM-SmolLM-135m)   |Diff-TSM-SmolLM-135M  | [SmolLM-135M-Instruct](https://huggingface.co/HuggingFaceTB/smollm-135m-instruct) |
| [🤗](https://huggingface.co/AvataarAI/Diff-TSM-SmolLM-360m)   |Diff-TSM-SmolLM-360M  | [SmolLM-360M-Instruct](https://huggingface.co/HuggingFaceTB/smollm-360m-instruct) |
| [🤗](https://huggingface.co/AvataarAI/Diff-TSM-Qwen-0.5B)   |Diff-TSM-Qwen-0.5B  | [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |
| [🤗](https://huggingface.co/AvataarAI/Diff-TSM-Qwen-1.5B)   |Diff-TSM-Qwen-1.5B  | [Qwen2.5-Math-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct) |

### Give it a try!
```python
# pip install transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
checkpoint = "AvataarAI/Diff-TSM-Qwen-0.5B"
device = "cuda" # for GPU usage or "cpu" for CPU usage
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
messages = [
    {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{{}}."},
    {"role": "user", "content": r"What is the derivative of the following function w.r.t. x: e^(2x) \sin(14x)"},
]
input_text=tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = model.generate(inputs, temperature=0.1, top_p=0.7, top_k=50, do_sample=True, max_new_tokens=2048)
print(tokenizer.decode(outputs[0]))
```

### Installation
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Download the veRL submodule
git submodule update --init --recursive   

# Install dependencies
pip install hydra-core==1.4.0dev1 \
            antlr4-python3-runtime==4.11 \
            sympy flash-attn typer
pip install -e external/verl
```

### Generate synthetic data
Create pairs of `(function, derivative)` without repetition, format them into LLM prompts, and split them into train and test sets. `.parquet` files are saved in the `--local-dir` directory.
```bash
python -m tslm.data main-generate-and-format \
    --local-dir "./data/calc-diff" \
    --simplicity 0.5 \
    --n-train 512 \
    --n-test 512 \
    --workers 40
```

### Run models to generate responses
```bash
# Outputs are saved under the key 'responses' in the output parquet file.
MODEL_PATH="AvataarAI/Diff-TSM-Qwen-0.5B" # HuggingFace model or local path
OUTPUT_PATH=./outputs/diff-tsm-qwen-0.5b.out.parquet

python -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    data.path="./data/calc-diff/test.parquet" \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.batch_size=1000 \
    data.output_path=$OUTPUT_PATH \
    model.path=$MODEL_PATH \
    rollout.temperature=0.1 \
    rollout.top_k=50 \
    rollout.top_p=0.7 \
    rollout.prompt_length=256 \
    rollout.response_length=2048 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.8 \
    +actor.ulysses_sequence_parallel_size=1 \
    +rollout.n=1
```

### Evaluate model performance
The `tslm/eval.py` script symbolically evaluates the predicted derivatives using the reward function in `tslm/reward.py` and reports average accuracy.
```bash
python -m tslm.eval ./outputs/diff-tsm-qwen-0.5b.out.parquet --workers 40
```

### Benchmarking Results
We evaluate the performance of base-line models upto 8B parameters, including those trained for the math domain without task specificity, and OpenAI’s o-series of models as representatives of large “intelligent” LLM
Results are in decreasing order of model size, with ours at the bottom.


Results:
| Model                             | Accuracy@1   |
|:----------------------------------|:-------------|
| o1                                | 95.1%        |
| gpt-4o                            | 88.3%        |
| o1-mini                           | 94.5%        |
| DeepSeek-R1-Distill-Llama-8B-10k  | 88.3%        |
| DeepSeek-R1-Distill-Qwen-7B-10k   | 93.8%        |
| Qwen2.5-Math-7B-Instruct          | 93.5%        |
| Mathstral-7B-v0.1                 | 68.8%        |
| Gemma-2-2B-it                     | 56.7%        |
| DeepSeek-R1-Distill-Qwen-1.5B-10k | 81.5%        |
| Qwen2.5-Math-1.5B-Instruct        | 94.2%        |
| Qwen2.5-0.5B-Instruct             | 69.8%        |
| Diff-TSM-Qwen-1.5B                | **99.1%**    |
| Diff-TSM-Qwen-0.5B                | **97.6%**    |
| Diff-TSM-SmolLM-360M              | **97.3%**    |
| Diff-TSM-SmolLM-135M-sft-s2404    | **93.1%**    |
| Diff-TSM-SmolLM-135M              | **95.4%**    |

<small>
<sup>*</sup> Deepseek-Distill models are evaluated with a longer 10k response length to accommodate their extremely long chain-of-thoughts. <br>
<sup>†</sup> SmolLM models are evaluated with a smaller 1780 response length, to prevent exceeding the maximum sequence length. Also, the starting SmolLM instruction-tuned models, though not in the table, have near 0% accuracy on the differentiation task.
</small>


### Training Code
Coming soon!

### TODO

- [ ] Harder hand-crafted examples to test OOD generalization.
- [ ] Update reward to incentivize simpler expressions.
- [ ] More tasks!

### Acknowledgements
- We use the [veRL](github.com/volcengine/verl) library for both training and running inference.
- We use [SymPy](https://github.com/sympy/sympy) extensively for all symbolic operations.
- We thank [HuggingFace](https://huggingface.co/) and the [Qwen team](https://qwenlm.github.io/) for open-sourcing the [SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966) and [Qwen2.5](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e) series of models, which we finetune for our experiments.
- Last but not least, [Typer](https://github.com/fastapi/typer) is awesome for building CLIs!

### Citing
If you find this work helpful, please cite us as follows:
```bibtex
@misc{tslm,
  author       = {Shubham Goel and Shubham Jain and Gaurav Baid and Sravanth Aluru},
  title        = {Task Specific Langugage Models},
  howpublished = {https://github.com/AvataarAILabs/tslm},
  year         = {2025}
}
```
