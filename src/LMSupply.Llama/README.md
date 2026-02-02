# LMSupply.Llama

Shared llama-server management for LMSupply libraries.

## Features

- **On-demand binary downloading**: Native binaries are downloaded on first use
- **GPU auto-detection**: Automatic selection of CUDA, Vulkan, Metal, or CPU backend
- **Fallback chain**: CUDA 13 → CUDA 12 → Vulkan → CPU (platform-dependent)
- **Caching**: Downloaded binaries are cached locally for reuse
- **Server pooling**: Efficient server instance reuse
- **Mode-aware pooling**: Separate servers for generation, embedding, and reranking

## Supported Backends

| Backend | Platform | GPU |
|---------|----------|-----|
| CUDA 12/13 | Windows, Linux | NVIDIA |
| Vulkan | Windows, Linux | AMD, Intel, NVIDIA |
| Metal | macOS | Apple Silicon |
| Hip/ROCm | Windows, Linux | AMD |
| CPU | All | AVX2/AVX512 optimized |

## Architecture

LMSupply.Llama uses [llama-server](https://github.com/ggml-org/llama.cpp) as a unified backend:

```
LMSupply.Llama/Server/
├── LlamaServerDownloader.cs   - Downloads from GitHub releases
├── LlamaServerPool.cs         - Server instance pooling
├── LlamaServerProcess.cs      - Process lifecycle management
├── LlamaServerClient.cs       - HTTP API client (embeddings, completions, rerank)
└── LlamaServerStateManager.cs - Health and state tracking
```

### Server Modes

| Mode | Flag | Use Case |
|------|------|----------|
| Generation | (default) | Text generation |
| Embedding | `--embedding` | Vector embeddings |
| Reranking | `--embedding --pooling rank` | Document reranking |

## Usage

This package is used internally by:

| Package | Server Mode | Use Case |
|---------|-------------|----------|
| `LMSupply.Generator` | Generation | GGUF text generation |
| `LMSupply.Embedder` | Embedding | GGUF embeddings |
| `LMSupply.Reranker` | Reranking | GGUF reranking |
