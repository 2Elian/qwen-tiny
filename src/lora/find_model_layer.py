from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path="/data1/nuist_llm/TrainLLM/ModelCkpt/qwen3-06b")

print(model)