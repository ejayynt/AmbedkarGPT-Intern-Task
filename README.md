# AmbedkarGPT - RAG-based Q&A System

A command-line question-answering system that helps users learn about Dr. B.R. Ambedkar's works and philosophy using Retrieval-Augmented Generation (RAG) with local LLM setup.

## Features

- **Local LLM Integration**: Uses Ollama with Mistral model for completely offline operation
- **Document Retrieval**: ChromaDB vector database for efficient document search
- **Interactive Q&A**: Command-line interface for asking questions about Dr. Ambedkar's works
- **Source Attribution**: Shows which documents were used to generate each answer
- **Comprehensive Evaluation**: Includes evaluation metrics for system performance analysis

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- Mistral model pulled in Ollama

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd RAG-assignment
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Set up Ollama:
```bash
# Install Ollama (follow instructions at https://ollama.ai/)
ollama serve
ollama pull mistral
```

4. Prepare your document corpus:
   - Create a `corpus` folder in the project directory
   - Add Dr. Ambedkar's text files (speeches, writings) as `.txt` files
   - Or use a single file and update the `CORPUS_PATH` in `main.py`

## Usage

### Interactive Q&A Mode

Run the main application:
```bash
python main.py
```

This will start an interactive session where you can ask questions about Dr. Ambedkar's works:

```
==============================================================
AmbedkarGPT - Interactive Q&A Mode
==============================================================
Ask questions about Dr. Ambedkar's works
Type 'quit' or 'exit' to stop

Your Question: What is the real remedy for caste system according to Ambedkar?

Thinking...

Answer: According to Dr. Ambedkar, the real remedy for the caste system lies in destroying the belief in the sanctity of the shastras...

Sources:
  [1] corpus/speech1.txt
  [2] corpus/speech3.txt
```

### Evaluation Mode

Run comprehensive evaluation:
```bash
python evaluation.py
```

This will:
- Test different chunking strategies (Small: 250, Medium: 550, Large: 900 characters)
- Generate detailed performance metrics
- Create comparison reports in JSON format

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Text Documents│    │   ChromaDB      │    │   Ollama/Mistral│
│   (Dr.Ambedkar's│───→│   Vector Store  │───→│   Local LLM     │
│   speeches)     │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Components

- **Document Loading**: Processes text files from corpus directory
- **Text Chunking**: Splits documents into overlapping chunks for better retrieval
- **Vector Embeddings**: Uses HuggingFace sentence-transformers for semantic search
- **LLM Integration**: Connects to local Mistral model via Ollama
- **RAG Pipeline**: Combines retrieval and generation for contextual answers

## Configuration

Key parameters in `main.py`:

```python
CORPUS_PATH = "corpus"      # Path to document folder
CHUNK_SIZE = 500           # Size of text chunks
CHUNK_OVERLAP = 50         # Overlap between chunks
```

## Evaluation Metrics

The system includes comprehensive evaluation using:

- **Retrieval Metrics**: Hit Rate, Mean Reciprocal Rank (MRR), Precision@3
- **Answer Quality**: ROUGE-L, BLEU, Cosine Similarity
- **Advanced Metrics**: Faithfulness, Answer Relevance
- **Question Types**: Factual, Comparative, Conceptual queries

## File Structure

```
RAG-assignment/
├── main.py                 # Main application
├── evaluation.py           # Evaluation system
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup
├── test_dataset.json      # Test questions
├── README.md              # This file
├── corpus/                # Document collection
│   ├── speech1.txt
│   ├── speech2.txt
│   └── ...
├── chroma_db/             # Vector database (generated)
└── test_results_*.json    # Evaluation results (generated)
```

## Results

Based on evaluation with 25 test questions:

| Chunking Strategy | Hit Rate | ROUGE-L | Faithfulness | Best For |
|------------------|----------|---------|--------------|----------|
| Small (250)      | 84%      | 0.266   | 30%          | Precise facts |
| Medium (550)     | 88%      | 0.266   | 45%          | Balanced performance |
| Large (900)      | 88%      | 0.273   | 58%          | Complete answers |

**Recommendation**: Large chunks (900 characters) for production use due to highest answer quality and faithfulness.

## Troubleshooting

**Common Issues:**

1. **"Connection refused" error**: Make sure Ollama is running (`ollama serve`)
2. **"Model not found"**: Pull the Mistral model (`ollama pull mistral`)
3. **"No documents found"**: Check that corpus folder exists with .txt files
4. **Import errors**: Install dependencies (`pip install -r requirements.txt`)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built using [LangChain](https://github.com/langchain-ai/langchain) for RAG pipeline
- [ChromaDB](https://github.com/chroma-core/chroma) for vector storage
- [Ollama](https://ollama.ai/) for local LLM inference
- [HuggingFace Transformers](https://huggingface.co/transformers/) for embeddings

## Author

Built as part of a RAG system development assignment focusing on Dr. B.R. Ambedkar's works and philosophy.