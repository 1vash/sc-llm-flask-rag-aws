"""
Queries:
1. Explain Transformers as in the article
2. My name is Ivan
3. What did I say?
4. What is the capital of France?
"""

import os
import openai

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from custom_chat_history import CustomChatBufferHistory
from flask import Flask, render_template, request, jsonify, make_response
from dotenv import load_dotenv, find_dotenv

# .env file is read locally with OPENAI_API_KEY inside
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ.get('OPENAI_API_KEY')

app = Flask(__name__)


class RagGeneratorMixin:
    def __init__(self, llm, custom_chat_history, text_splitter):
        self.llm = llm
        self.custom_chat_history = custom_chat_history
        self.vectordb = None
        self.embeddings_model = None
        self.text_splitter = text_splitter

    def initialize(self):
        # More complicated initialization is happening here, divided to make code more modular
        self.initialize_embedding_model()
        self.initialize_vector_db_and_fill_it()

    def initialize_embedding_model(self):
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L12-v2',
            model_kwargs={"device": "cpu"},  # cuda
            # Set True to compute cosine similarity
            encode_kwargs={"normalize_embeddings": True})

    def initialize_vector_db_and_fill_it(self):
        print('Getting document chunks...')
        docs = self.get_chunks(embeddings_model=self.embeddings_model, splitter=self.text_splitter)
        self.vectordb = FAISS.from_documents(docs, self.embeddings_model)

    @staticmethod
    def get_chunks(embeddings_model, splitter):
        pdf_file_url = 'https://arxiv.org/pdf/1706.03762.pdf'
        loader = PyPDFLoader(pdf_file_url)

        print('Text Splitter Name:', splitter)
        if splitter == 'character_splitter':
            text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=50,
                                                  length_function=len)
        elif splitter == 'semantic_splitter':
            text_splitter = SemanticChunker(embeddings_model, breakpoint_threshold_type="percentile")
        else:
            raise AssertionError(f'There is not such `splitter` {splitter}')
        return loader.load_and_split(text_splitter)

    def search_results(self, _input, k=3):
        # Similarity search against embedding
        search_results = self.vectordb.similarity_search(_input, k=k)
        # Combine content
        content_combined = ""
        for result in search_results:
            content_combined += (result.page_content + " \n")
        return {"content_combined": content_combined}


class OpenAIRagLLM(RagGeneratorMixin):
    def __init__(self, text_splitter):
        self.llm = self.connect_to_model('gpt-3.5-turbo')
        self.custom_chat_history = CustomChatBufferHistory(human_prefix_start='Human:', ai_prefix_start='AI:')
        # If you don't know the answer, say 'I don't know' is added to compare model generation capabilities only.
        self.prompt_template = """
        You are a helpful, respectful and honest assistant. You help to answer questions regarding `Attention is all you need` article.
        Answer the following Question based on the Context only. Only answer from the Context. If you don't know the answer, say 'I don't know'.
 
        Chat history: {chat_history}

        Context: {content}

        Question: {question}
        """

        super().__init__(
            llm=self.llm,
            custom_chat_history=self.custom_chat_history,
            text_splitter=text_splitter
        )

    @staticmethod
    def connect_to_model(model_name):
        print("Load OpenAI LangChain LLM Object...")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(temperature=0, model=model_name, callbacks=[StreamingStdOutCallbackHandler()])

    def generate(self, _input):
        content = self.search_results(_input)
        prompt = self.prompt_template.format(content=content["content_combined"],
                                             chat_history=list(self.custom_chat_history.memory),
                                             question=_input)
        print('Final Prompt:', prompt, end='\n\n\n')
        response = self.llm.invoke(prompt)
        self.custom_chat_history.update_history(human_query=_input, ai_response=response)
        return response.content


class HfRagLlm(RagGeneratorMixin):
    def __init__(self, text_splitter):
        self.llm = self.connect_to_model()
        # Refer to Llama-2 prompt guide: https://replicate.com/blog/how-to-prompt-llama
        self.custom_chat_history = CustomChatBufferHistory(human_prefix_start='[INST]', human_prefix_end='[/INST]',
                                                           ai_prefix_start='', ai_prefix_end='')
        self.prompt_template = """<s>[INST]<<SYS>>
        You are a helpful, respectful and honest assistant. You help to answer questions regarding `Attention is all you need` article.
        Answer the following Question based on the CONTEXT only provided in the <context> tag. Only answer from the Context. If you don't know the answer, say 'I don't know'.
        The format of the answer is: `Final Answer: <Put your answer here>`.
        <</SYS>>

        chat history: <chat_history> {chat_history} </chat_history>

        CONTEXT: <context> {content} </context>

        [INST]QUESTION: {question}[/INST]
        [/INST]"""

        super().__init__(
            llm=self.llm,
            custom_chat_history=self.custom_chat_history,
            text_splitter=text_splitter
        )

    @staticmethod
    def connect_to_model():
        print('Connect to LLM TGI Inference ...')
        from langchain_community.llms import HuggingFaceTextGenInference
        url = 'http://llm_inference_service:8080/'
        print('Inference Url:', url)
        return HuggingFaceTextGenInference(
            inference_server_url=url,
            max_new_tokens=256,
            top_k=50,
            top_p=0.95,
            typical_p=0.95,
            temperature=0.01,
            repetition_penalty=1.03,
            callbacks=[StreamingStdOutCallbackHandler()],
            stop_sequences=["</s>"]
        )

    @staticmethod
    def post_processing_final_answer(text):
        final_answer_prefix = "Final Answer:"
        final_answer_index = text.find(final_answer_prefix)
        if final_answer_index != -1:
            return text[final_answer_index + len(final_answer_prefix):].strip()
        return text

    def generate(self, _input):
        content = self.search_results(_input)
        prompt = self.prompt_template.format(content=content["content_combined"],
                                             chat_history=list(self.custom_chat_history.memory),
                                             question=_input)
        print('Final Prompt:', prompt, end='\n\n\n')
        response = self.llm.invoke(prompt)
        response = self.post_processing_final_answer(response)
        self.custom_chat_history.update_history(human_query=_input, ai_response=response)
        return response


@app.route("/openai")
def home_openai():
    openai_llm_rag.custom_chat_history.clean_memory()
    return render_template("index_openai.html")


@app.route("/hf")
def home_hf():
    hf_llm_rag.custom_chat_history.clean_memory()
    return render_template("index_hf.html")


@app.route("/generate", methods=['POST'])
def get_bot_response():
    data = request.json
    user_text = data.get('msg', '')
    index = data.get('index', '')

    if index == 'hf':
        response = hf_llm_rag.generate(user_text)
    elif index == 'openai':
        response = openai_llm_rag.generate(user_text)
    else:
        error_response = jsonify({"error": "Invalid html index"})
        return make_response(error_response, 400)

    return jsonify({"response": response})


if __name__ == '__main__':
    """
    1. Connecting to APIs & 2. Initialize LLM RAG Generator object
    """

    # passed in dockerfile
    text_splitter_name = os.environ.get('TEXT_SPLITTER', 'character_splitter')

    print('Initialize OpenAI GenAI Project')
    openai_llm_rag = OpenAIRagLLM(text_splitter=text_splitter_name)
    openai_llm_rag.initialize()

    print('Initialize HuggingFace GenAI Project')
    # Required connection to already set up TGI inference
    hf_llm_rag = HfRagLlm(text_splitter=text_splitter_name)
    hf_llm_rag.initialize()

    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
