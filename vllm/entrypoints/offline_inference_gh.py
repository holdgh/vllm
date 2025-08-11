from vllm import LLM, SamplingParams
# from vllm.sampling_params import SamplingParams
# from vllm.entrypoints.llm import LLM

if __name__ == '__main__':
    # Define a list of input prompts
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The largest ocean is",
    ]

    # Define sampling parameters
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)  # 定义模型推理时的随机采样参数

    # Initialize the LLM engine with the OPT-125M model
    llm = LLM(model=r"E:\aiModel\llmModel\Qwen2.5-0.5B-Instruct")

    # Generate outputs for the input prompts
    outputs = llm.generate(prompts, sampling_params)

    # Print the generated outputs
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")