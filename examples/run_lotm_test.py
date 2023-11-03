from transformers import AutoTokenizer
import transformers
import torch

import random
import string

model = "lmsys/vicuna-13b-v1.5"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

def make_guess(prompts):
    for i in range(len(prompts)):
        prompts[i] = "USER: " + prompts[i] + "\n\nASSISTANT: "
    
    input_len = len(tokenizer.encode(prompts[i]))
    output_len = 70
    sequences = pipeline(
        prompts,
        do_sample=True,
        top_k=1,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=input_len + output_len,
    )

    # only return outputs
    for i in range(len(sequences)):
        sequences[i] = sequences[i][0]['generated_text'].split("\n\nASSISTANT: ")[1]

    return sequences


def create_prompts(n, k, p):
    # create n prompts with k key-value pairs, with target key at percentile p
    def generate_uuid():
        # Generate random alphanumeric value of length 8
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

    prompts = []
    vals = []

    for _ in range(n):
        # Create k random key-value pairs
        data = {generate_uuid(): generate_uuid() for _ in range(k)}
        
        # Get the key at the desired percentile
        keys = list(data.keys())
        target_key = keys[int(len(keys) * p)]
        
        # Convert data to a formatted string representation
        data_str = ',\n    '.join([f'"{key}": "{value}"' for key, value in data.items()])
        
        prompt = (
            f"Please extract the value corresponding to the specified key in the JSON object below.\n\n"
            f"JSON data:\n"
            f"{{\n    {data_str}\n}}\n"
            f"Key: {target_key}\n"
            f"Output only the corresponding value:\n"
        )
        
        prompts.append(prompt)
        vals.append(data[target_key])
        
    return prompts, vals

def run_benchmark(n, k):
    # Run a benchmark with k key-value pairs in the prompt
    # For percentiles 0, 0.1, 0.2, ..., 0.9, 1.0, run n trials and record the average accuracy

    # Save results to a spreadsheet. First sheet contains results, second sheet contains prompts
    # First sheet: first column is percentile, second column is accuracy
    # Second sheet: first column is percentile, second column is prompt, third column is response, fourth column is correct value, fifth column is True/False

    # Create a graph of percentile vs accuracy, and save it to a file

    import time
    start = time.time()
    print(f"Running benchmark for n={n}, k={k}...")

    percentiles = [i / 10 for i in range(11)]
    percentiles[-1] -= 0.0001 # avoid 1.0

    accuracies = []
    data = []
    for p in percentiles:
        print(f"Running benchmark for percentile {p}...")

        prompts, vals = create_prompts(n, k, p)
        guesses = make_guess(prompts)
        # correct if val is in guess
        correct = [ (val in guess) for val, guess in zip(vals, guesses) ]
        accuracies.append(sum(correct) / len(correct))
        
        for i in range(len(prompts)):
            data.append([p, prompts[i], guesses[i], vals[i], correct[i]])

    # Save results to a spreadsheet
    import pandas as pd
    df1 = pd.DataFrame({'percentile': percentiles, 'accuracy': accuracies})
    df2 = pd.DataFrame(data, columns=['percentile', 'prompt', 'response', 'gold', 'correct?'])
    with pd.ExcelWriter(f'data/data_n{n}_k{k}.xlsx') as writer:
        df1.to_excel(writer, sheet_name='results')
        df2.to_excel(writer, sheet_name='prompts')

    # Create a graph of percentile vs accuracy
    import matplotlib.pyplot as plt
    print(percentiles, accuracies)
    plt.plot(percentiles, accuracies)
    plt.xlabel('Percentile')
    plt.ylabel('Accuracy')
    plt.title(f'{k} key-value pairs, {n} trials per data point')
    plt.savefig(f'data/graph_n{n}_k{k}.png')

    end = time.time()
    print(f"Benchmark finished in {end - start} seconds.")

if __name__ == "__main__":
    import sys
    n, k = int(sys.argv[1]), int(sys.argv[2])
    run_benchmark(n, k)
    # run_benchmark(100, 75)
    # run_benchmark(100, 130)
    # run_benchmark(100, 200)