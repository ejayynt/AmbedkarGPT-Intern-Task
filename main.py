import os

# Disable ChromaDB telemetry to avoid error messages
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class AmbedkarGPT:
    def __init__(self, corpus_path="corpus", chunk_size=500, chunk_overlap=50):
        """
        Set up the document Q&A system
        corpus_path - where the documents are stored
        chunk_size - how big each text chunk should be
        chunk_overlap - how much chunks should overlap
        """
        self.docs_folder = corpus_path
        self.text_chunk_size = chunk_size
        self.overlap_size = chunk_overlap
        self.vector_db = None
        self.question_chain = None

        # Load the sentence transformer - this runs on your machine, no internet needed
        print("Starting up the embedding model...")
        self.text_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Connect to the local Mistral model via Ollama
        print("Connecting to local Mistral model...")
        self.language_model = Ollama(model="mistral", temperature=0.1)

    def load_documents(self):
        """Read all the text files from the documents folder"""
        print(f"Reading documents from {self.docs_folder}...")

        # Figure out if we're dealing with a folder or just one file
        if os.path.isdir(self.docs_folder):
            doc_loader = DirectoryLoader(
                self.docs_folder, glob="*.txt", loader_cls=TextLoader
            )
        else:
            doc_loader = TextLoader(self.docs_folder)

        docs = doc_loader.load()
        print(f"Found and loaded {len(docs)} documents")
        return docs

    def split_documents(self, documents):
        """Break up the documents into smaller, manageable pieces"""
        print(
            f"Breaking documents into chunks (size: {self.text_chunk_size}, overlap: {self.overlap_size})..."
        )

        # Set up the text splitter with reasonable breakpoints
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.text_chunk_size,
            chunk_overlap=self.overlap_size,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],  # Try to break at natural points
        )

        text_chunks = splitter.split_documents(documents)
        print(f"Got {len(text_chunks)} text chunks to work with")
        return text_chunks

    def create_vectorstore(self, chunks, db_folder="./chroma_db"):
        """Build the searchable database from document chunks"""
        print("Building the search database...")

        # Clean up any old database first
        if os.path.exists(db_folder):
            import shutil

            shutil.rmtree(db_folder)

        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.text_embeddings,
            persist_directory=db_folder,
        )

        print(f"Database ready with {len(chunks)} document chunks indexed")
        return self.vector_db

    def setup_qa_chain(self):
        """Wire up the question answering system"""
        print("Setting up the Q&A system...")

        # Instructions for the AI on how to respond to questions
        system_instructions = """You are a Document Q&A Assistant, designed to help people find information from a collection of documents. The document contains speeches of Dr. Bhimrao Ramji Ambedkar (1891-1956) - the visionary architect of India's Constitution, renowned social reformer, brilliant jurist, economist, and tireless champion of Dalit rights.

YOUR ROLE:
- Provide accurate, helpful answers based on the provided documents
- Focus on the content within the document collection
- Maintain a clear and informative tone
- Stay grounded in the source material

INSTRUCTIONS:
Use the following pieces of context to answer the question at the end. Base your answers strictly on the provided documents. If you don't know the answer based on the context provided, say "I cannot answer this question based on the provided documents", don't try to make up an answer.

Context:
{context}

Question: {question}

Answer: """

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        self.question_chain = RetrievalQA.from_chain_type(
            llm=self.language_model,
            chain_type="stuff",
            retriever=self.vector_db.as_retriever(
                search_kwargs={"k": 3}  # Get the 3 most relevant document chunks
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True,
        )

        print("Q&A system is ready to go!")

    def initialize(self):
        """Get everything set up and ready to answer questions"""
        docs = self.load_documents()
        text_chunks = self.split_documents(docs)
        self.create_vectorstore(text_chunks)
        self.setup_qa_chain()
        print("\n✓ Everything's loaded and ready to answer questions!\n")

    def ask(self, question):
        """
        Ask a question and get an answer

        Args:
            question: User's question

        Returns:
            dict with 'answer' and 'source_documents'
        """
        if self.question_chain is None:
            raise ValueError("Need to call initialize() first before asking questions")

        response = self.question_chain.invoke({"query": question})
        return response

    def interactive_mode(self):
        """Start the interactive question-and-answer session"""
        print("=" * 60)
        print("AmbedkarGPT - Interactive Q&A Mode")
        print("=" * 60)
        print("Ask questions about Dr. Ambedkar's works")
        print("Type 'quit' or 'exit' to stop\n")

        while True:
            question = input("Your Question: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("\nThank you for using AmbedkarGPT!")
                break

            if not question:
                continue

            print("\nThinking...\n")
            result = self.ask(question)

            print("Answer:", result["result"])
            print("\nSources:")
            for i, doc in enumerate(result["source_documents"], 1):
                source = doc.metadata.get("source", "Unknown")
                print(f"  [{i}] {source}")
            print("\n" + "-" * 60 + "\n")


def main():
    """Start up the document Q&A assistant"""

    # Set up the basic configuration
    DOCUMENTS_PATH = (
        "corpus"  # Change this to a specific file if you have just one document
    )
    TEXT_CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # Make sure the documents folder actually exists
    if not os.path.exists(DOCUMENTS_PATH):
        print(f"Can't find the documents at: {DOCUMENTS_PATH}")
        print("Make sure you have a 'corpus' folder with your text documents in it")
        return

    try:
        # Initialize system
        system = AmbedkarGPT(
            corpus_path=CORPUS_PATH, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

        system.initialize()

        # Start interactive mode
        system.interactive_mode()

    except Exception as error:
        print(f"\nSomething went wrong: {error}")
        print("\nHere's what to check:")
        print("1. Is Ollama running? Try: ollama serve")
        print("2. Do you have Mistral installed? Try: ollama pull mistral")
        print("3. Are your document files in the right place?")
        print("4. Maybe restart your terminal and try again")


if __name__ == "__main__":
    main()
