from huggingface_hub import InferenceClient

client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")
# client=InferenceClient("TheBloke/Mistral-7B-v0.1-GGUF")

def format_prompt(message):

    system_prompt = "As a seasoned legal expert specialized in the Indian Penal Code (IPC), your task is to provide a meticulously response. For the given scenario, furnish the relevant IPC sections along with a brief, line-by-line description of each section and the corresponding punishments. Ensure clarity and coherence in your response, presenting the information in a well-organized manner."

    prompt = f"<s>[SYS] {system_prompt} [/SYS]"

    prompt += f"[INST] {message} [/INST]"
    return prompt

def generate(
    prompt,temperature=0.2, max_new_tokens=None, top_p=0.95, repetition_penalty=1.0,
):
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=42,
    )

    formatted_prompt = format_prompt(prompt)

    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""

    for response in stream:
        output += response.token.text
        # yield output

    # print(output)
    return output