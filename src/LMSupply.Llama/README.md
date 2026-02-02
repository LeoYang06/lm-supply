# LMSupply.Llama

Shared llama.cpp runtime management for LMSupply libraries.

## Features

- **On-demand binary downloading**: Native binaries are downloaded on first use
- **GPU auto-detection**: Automatic selection of CUDA, Vulkan, Metal, or CPU backend
- **Fallback chain**: CUDA 13 → CUDA 12 → Vulkan → CPU (platform-dependent)
- **Caching**: Downloaded binaries are cached locally for reuse
- **Server pooling**: Efficient server instance reuse (Generator)

## Supported Backends

| Backend | Platform | GPU |
|---------|----------|-----|
| CUDA 12/13 | Windows, Linux | NVIDIA |
| Vulkan | Windows, Linux | AMD, Intel, NVIDIA |
| Metal | macOS | Apple Silicon |
| Hip/ROCm | Windows, Linux | AMD |
| CPU | All | AVX2/AVX512 optimized |

## Architecture

LMSupply.Llama provides two runtime backends:

### llama-server (for Generator)

Downloads and manages [llama-server](https://github.com/ggml-org/llama.cpp) binaries from GitHub releases for text generation.

```
LMSupply.Llama/Server/
├── LlamaServerDownloader.cs   - Downloads from GitHub releases
├── LlamaServerPool.cs         - Server instance pooling
├── LlamaServerProcess.cs      - Process lifecycle management
├── LlamaServerClient.cs       - HTTP API client
└── LlamaServerStateManager.cs - Health and state tracking
```

### LLamaSharp (for Embedder)

Downloads LLamaSharp.Backend.* NuGet packages for embedding generation.

```
LMSupply.Llama/
├── LlamaBackend.cs           - Backend enum (Cpu, Cuda12, Vulkan, Metal, etc.)
├── LlamaRuntimeManager.cs    - Singleton manager with fallback chain
├── LlamaNuGetDownloader.cs   - Downloads LLamaSharp backend packages
└── DownloadProgress.cs       - Progress reporting
```

## Usage

This package is used internally by:

| Package | Runtime | Use Case |
|---------|---------|----------|
| `LMSupply.Generator` | llama-server | GGUF text generation |
| `LMSupply.Embedder` | LLamaSharp | GGUF embedding |
