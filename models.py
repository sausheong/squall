import os
from dotenv import load_dotenv, find_dotenv
from llama_index import ServiceContext
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
load_dotenv(find_dotenv())

llm = LlamaCPP(
    model_path=os.getenv('LOCAL_MODEL'),
    temperature=0.0,
    max_new_tokens=512,
    context_window=3072,
    generate_kwargs={},
    model_kwargs={
        "n_gpu_layers": 38, 
        "f16_kv": True,
        "n_batch": 1024,
        "rms_norm_eps": 1e-5,
    },
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model="local:" + os.getenv('EMBED_MODEL'),
)