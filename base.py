from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import os
from dotenv import load_dotenv, find_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain.schema.document import Document

from langchain.embeddings import CohereEmbeddings
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever

from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA

import pinecone
from langchain_community.llms import Cohere	

# DOTENV
load_dotenv(find_dotenv())

# COHERE_API_KEY = os.environ['COHERE_API_KEY']
COHERE_API_KEY = os.environ.get('COHERE_API_KEY')


# initialize pinecone
index_name = "langchain"
# namespace = "eecs2001"

# pinecone.init(
#     api_key=os.environ['PINECONE_API_KEY'],
#     environment=os.environ['PINECONE_ENVIRONMENT']
# )
pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY'),
    environment=os.environ.get('PINECONE_ENVIRONMENT')
)

embeddings = CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key=COHERE_API_KEY)
llm = Cohere(model="command-nightly", temperature=0.2, cohere_api_key=COHERE_API_KEY, max_tokens=1024)


template = """You are an AI assistant for answering questions about the Document you have uploaded.
            You are given the following extracted parts of a long document and a question that requires those contexts. 
            Provide a conversational answer.
            If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
            Answer in markdown format.
            =========
            {context}
            =========
            User: {question}
            AI Assistant:
            """

prompt_template = PromptTemplate(input_variables=["question", "context"], template=template)
chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt_template)

# 2 stage RAG with CohereRerank
compressor = CohereRerank(model="rerank-multilingual-v2.0", cohere_api_key=COHERE_API_KEY, user_agent="langchain", top_n=6)

def pretty_print_docs(docs):
    return(
        f"\n\n---\n\n".join(
            [f"Document {i+1} :\n\n" + str(d.metadata) + "\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

# FLASK
app = Flask(__name__)
CORS(app, resources={r"/query": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}})


@app.route("/query", methods=["POST"], strict_slashes=False)
@cross_origin() 
def llm():
    
    # return jsonify({'message': 'Test response'}) # For testing
    # try:
    #     # Your route logic
    #     query = request.form.get("query", "")
    #     # retriever = docsearch.as_retriever(search_type="mmr")
    #     # matched_docs = compression_retriever.get_relevant_documents(query)
    #     compressed_docs = compression_retriever.get_relevant_documents(query)
    #     # answer = chain({"input_documents": compressed_docs, "question": query}, return_only_outputs=True)
    #     qa = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff", retriever=compression_retriever)
    #     return jsonify({
    #         # "message": answer["output_text"]
    #         "message": qa({"query": query})["result"],
    #         "docs": pretty_print_docs(compressed_docs)
    #     })
    # except Exception as e:
    #     print(e)  # Log the error
    #     return jsonify({'error': 'An error occurred'}), 500

    try:
        data = request.json
        query = data.get('query')
        namespace = data.get('course').lower()
        docsearch = Pinecone.from_existing_index(index_name, embeddings, namespace=namespace)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=docsearch.as_retriever(search_kwargs={"k": 20}))
        # Your logic to handle 'query'
        # query = request.form.get("query", "")
        # retriever = docsearch.as_retriever(search_type="mmr")
        # matched_docs = compression_retriever.get_relevant_documents(query)
        compressed_docs = compression_retriever.get_relevant_documents(query)
        answer = chain({"input_documents": compressed_docs, "question": query}, return_only_outputs=True)
        return jsonify({
            "message": answer["output_text"],
            # "message": qa({"query": query})["result"],
            "docs": pretty_print_docs(compressed_docs)
        })
    except Exception as e:
        print(e)  # This will print the exception to the Flask console
        return jsonify({'error': str(e)}), 500
   

# Running app
if __name__ == '__main__':
    app.run(port=5000)
