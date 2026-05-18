from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/data1/nuist_llm/TrainLLM/ModelCkpt/qwen3-06b-base")
print(tokenizer)

# token_ids = [151645, 198, 151644, 77091, 198, 151667]
# token = tokenizer.decode(token_ids)
# print(token)
tokens = ["\\(", "\\boxed"]
token_ids = tokenizer.encode(tokens)
print(token_ids)

token_ids = [59]
token = tokenizer.decode(token_ids)
print(token)