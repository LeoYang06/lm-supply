# LMSupply.Reranker

Local semantic reranking for .NET with cross-encoder models.

## Features

- **Zero-config**: Models download automatically from HuggingFace
- **GPU Acceleration**: CUDA, DirectML (Windows), CoreML (macOS)
- **Cross-platform**: Windows, Linux, macOS
- **RAG Integration**: Perfect for improving retrieval quality
- **Multi-Tokenizer Support**: Automatic detection of WordPiece, Unigram, BPE tokenizers

## Quick Start

```csharp
using LMSupply.Reranker;

// Load the default model
await using var reranker = await LocalReranker.LoadAsync("default");

// Rerank documents
var results = await reranker.RerankAsync(
    query: "What is machine learning?",
    documents: ["ML is a branch of AI...", "The weather is nice..."],
    topK: 5);

foreach (var result in results)
    Console.WriteLine($"{result.Index}: {result.Score:F3}");
```

## Available Models

| Alias | Model | Size | Tokenizer | Description |
|-------|-------|------|-----------|-------------|
| `default` | MS MARCO MiniLM L6 | ~90MB | WordPiece | Best speed/quality balance |
| `fast` | MS MARCO TinyBERT | ~18MB | WordPiece | Ultra-fast, latency-critical |
| `quality` | BGE Reranker Base | ~440MB | Unigram | Higher accuracy, multilingual |
| `large` | BGE Reranker Large | ~1.1GB | Unigram | Highest accuracy |
| `multilingual` | BGE Reranker v2-m3 | ~1.1GB | Unigram | 8K context, 100+ languages |

## Tokenizer Auto-Detection

The reranker automatically detects the correct tokenizer type:

| Type | Detection | Example Models |
|------|-----------|----------------|
| WordPiece | `vocab.txt` | MS MARCO MiniLM, TinyBERT |
| Unigram | `tokenizer.json` (type: Unigram) | bge-reranker-base, XLM-RoBERTa |
| BPE | `tokenizer.json` (type: BPE) | Some multilingual models |

This ensures compatibility with virtually any cross-encoder model from HuggingFace.

## GPU Acceleration

```bash
# NVIDIA GPU
dotnet add package Microsoft.ML.OnnxRuntime.Gpu

# Windows (AMD/Intel/NVIDIA)
dotnet add package Microsoft.ML.OnnxRuntime.DirectML
```

## Configuration

```csharp
var options = new RerankerOptions
{
    Provider = ExecutionProvider.Auto,  // GPU auto-detection
    MaxSequenceLength = 512,
    BatchSize = 32
};

var reranker = await LocalReranker.LoadAsync("default", options);
```

## Version History

### v0.8.7
- Added automatic tokenizer type detection (WordPiece, Unigram, BPE)
- Fixed compatibility with bge-reranker-base and other Unigram-based models

### v0.8.6
- Fixed vocab parsing for Array vs Object format
