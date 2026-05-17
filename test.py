from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/data1/nuist_llm/TrainLLM/ModelCkpt/qwen3-06b-base")
print(tokenizer)

# token_ids = [151645, 198, 151644, 77091, 198, 151667]
# token = tokenizer.decode(token_ids)
# print(token)
tokens = ["นำข้อมูลมาใช้"]
token_ids = tokenizer.encode(tokens)
print(token_ids)

token_ids = [124822, 47839, 80614, 91200, 124029, 124207, 19841]
token = tokenizer.decode(token_ids)
print(token)