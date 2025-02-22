import uvicorn

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from semantic_kernel import Kernel
from semantic_kernel.contents.chat_history import ChatHistory
from plugins.ragPlugin import RAGPlugin
from utils.settings import setup, load_settings_from_yaml

from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone as PineconeStore
from langchain.embeddings import OpenAIEmbeddings

import pinecone 
from langchain_pinecone import Pinecone as PineconeStore

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = load_settings_from_yaml("settings.yaml")
llm, text_embedder, chat_completion, execution_settings = setup(settings)
kernel = Kernel()

# Prepare OpenAI service using credentials stored in the `.env` file
kernel.add_service(chat_completion)

# Register the Math and String plugins
routing_plugin = RoutingPlugin(settings.ORS_API_KEY)
rag_plugin = RAGPlugin(settings, text_embedder, llm)

kernel.add_plugin(
    routing_plugin,
    plugin_name="Routing",
)

kernel.add_plugin(
    rag_plugin,
    plugin_name="RAG",
)

# Create a history of the conversation
history = ChatHistory()

embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)

@app.get("/ask")
async def local_search(query: str = Query(..., description="Ask anything")):
    global history

    # query = "The user lives in Cobham and uses M25J10 for commute to Imperial College London. Ask the following question:\n" + query
    try:
        history.add_message({"role": "user", "content": query})

        # Get the response from the AI
        response = await chat_completion.get_chat_message_content(
            chat_history=history,
            settings=execution_settings,
            kernel=kernel,
        )

        # Add the message from the agent to the chat history
        history.add_message(response)

        raw_result = None
        if routing_plugin.raw_route_json:
            raw_result = routing_plugin.raw_route_json
        elif rag_plugin.raw_RAG_result:
            raw_result = rag_plugin.raw_RAG_result

        response_dict = {
            "llm_reply": response.content,
            "plugin_called": "route" if routing_plugin.raw_route_json else "RAG",
            "raw_json": raw_result
        }

        routing_plugin.raw_route_json = None
        rag_plugin.raw_RAG_result = None
        return JSONResponse(content=response_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/refresh_history")
async def local_search():
    global history
    try:
        history = ChatHistory()
        routing_plugin.raw_route_json = None
        rag_plugin.raw_RAG_result = None
        return JSONResponse(content={"status": "Chat history has been refreshed"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def status():
    return JSONResponse(content={"status": "Server is up and running"})

@app.get("/ask_vector")
async def vector_search(query: str = Query(..., description="Ask anything")):

    # query = "The user lives in Cobham and uses M25J10 for commute to Imperial College London. Ask the following question:\n" + query

    pc = pinecone.Pinecone(
        api_key=settings.PINECONE_API_KEY,
        environment=settings.PINECONE_ENV
    )
    
    if settings.PINECONE_INDEX_NAME not in pc.list_indexes().names():
        raise Exception(f"Index {settings.PINECONE_INDEX_NAME} not found in Pinecone")

    index = pc.Index(settings.PINECONE_INDEX_NAME)

    embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
    docsearch = PineconeStore(
        index=index,
        embedding=embeddings,
        text_key="text"
    )

    embedded_query_vector = embeddings.embed_query(query)
    pinecone_result = index.query(
        vector=embedded_query_vector,
        top_k=3,  # Adjust as needed
        include_values=True,
        include_metadata=True
    )

    retrieved_vectors = []
    for match in pinecone_result.matches:
        retrieved_vectors.append({
            "id": match.id,
            "score": match.score,
            "values": match.values,        # The actual vector
            "metadata": match.metadata     # Any associated metadata
        })

    chain = RetrievalQA.from_chain_type(
        llm=OpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0
        ),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
    )
    answer = chain.run(query)

    vector_dict = {
        "embedded_query_vector": embedded_query_vector,
        "retrieved_vectors": retrieved_vectors
    }

    response_dict = {
        "llm_reply": answer,
        "plugin_called": "vector",
        "raw_json": vector_dict
    }
    return JSONResponse(content=response_dict)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)