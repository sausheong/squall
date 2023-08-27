import os
from dotenv import load_dotenv, find_dotenv

from llama_index import ServiceContext, StorageContext, load_index_from_storage, set_global_service_context
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt


load_dotenv(find_dotenv())
llm = LlamaCPP(
    model_path=os.getenv('LOCAL_MODEL'),
    temperature=0.1,
    max_new_tokens=256,
    context_window=3900,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": 38, "f16_kv": True},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model="local:BAAI/bge-large-en",
    )

storage_context = StorageContext.from_defaults(persist_dir="./data")
index = load_index_from_storage(
    service_context=service_context, 
    storage_context=storage_context, 
    index_id="squalldb",
    )
engine = index.as_query_engine(
    service_context=service_context,
    streaming=True,
    )

response = engine.query("Who is Sau Sheong?")
response.print_response_stream() 