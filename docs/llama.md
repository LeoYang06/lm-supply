# LMSupply.Llama

Shared llama.cpp runtime management for GGUF model support in LMSupply.

## Overview

`LMSupply.Llama` provides centralized management of llama.cpp runtimes, enabling GGUF model support across LMSupply packages.

This package follows LMSupply's on-demand philosophy: native binaries are downloaded only when first needed, with automatic GPU backend detection and fallback.

## Architecture

LMSupply.Llama provides two runtime backends for different use cases:

### llama-server (Generator)

For text generation, LMSupply uses [llama-server](https://github.com/ggml-org/llama.cpp) - the official HTTP server from llama.cpp. This provides:

- **Server pooling** - Efficient reuse of server instances across multiple requests
- **HTTP API** - OpenAI-compatible chat completions endpoint
- **Auto-download** - Binaries downloaded from GitHub releases on first use
- **GPU acceleration** - CUDA, Vulkan, Metal, ROCm support

```
LMSupply.Llama/Server/
├── LlamaServerDownloader.cs   - Downloads binaries from GitHub releases
├── LlamaServerPool.cs         - Manages server instance pooling
├── LlamaServerProcess.cs      - Server process lifecycle management
├── LlamaServerClient.cs       - HTTP client for API calls
└── LlamaServerStateManager.cs - Server state and health tracking
```

### LLamaSharp (Embedder)

For embeddings, LMSupply uses [LLamaSharp](https://github.com/SciSharp/LLamaSharp) - C# bindings for llama.cpp. This provides:

- **Native bindings** - Direct llama.cpp integration without HTTP overhead
- **Embedding extraction** - Optimized for embedding generation
- **NuGet packages** - LLamaSharp.Backend.* packages for different platforms

```
LMSupply.Llama/
├── LlamaRuntimeManager.cs    - Singleton runtime initialization
├── LlamaNuGetDownloader.cs   - Downloads LLamaSharp backend NuGet packages
└── LlamaBackend.cs           - Backend enum definition
```

## Features

- **On-demand binary download**: Native binaries downloaded on first use
- **Automatic backend selection**: Detects hardware and selects optimal backend
- **Fallback chain**: Automatically falls back to CPU if GPU backends fail
- **Cross-platform**: Windows, Linux, macOS (including Apple Silicon)
- **Server pooling**: Efficient server instance reuse (Generator)

## Supported Backends

### llama-server (Generator)

| Backend | Platform | Hardware |
|---------|----------|----------|
| `Cuda12/13` | Windows/Linux | NVIDIA GPU with CUDA 12.x/13.x |
| `Vulkan` | Windows/Linux | AMD/Intel/NVIDIA GPU |
| `Metal` | macOS | Apple Silicon (M1/M2/M3/M4) |
| `Hip` | Windows/Linux | AMD ROCm |
| `Cpu` | All | CPU with AVX2/AVX512 optimization |

### LLamaSharp (Embedder)

| Backend | Platform | Hardware |
|---------|----------|----------|
| `Cuda13` | Windows/Linux | NVIDIA GPU with CUDA 13.x |
| `Cuda12` | Windows/Linux | NVIDIA GPU with CUDA 12.x |
| `Vulkan` | Windows/Linux | AMD/Intel discrete GPU |
| `Metal` | macOS | Apple Silicon (M1/M2/M3) |
| `Cpu` | All | CPU with AVX2/AVX512 optimization |

## Backend Selection

The runtime manager automatically detects your hardware and selects the best backend:

```
macOS ARM64:  Metal → CPU
NVIDIA GPU:   CUDA 13 → CUDA 12 → Vulkan → CPU
AMD GPU:      Vulkan → CPU
Intel GPU:    Vulkan → CPU
No GPU:       CPU
```

## Usage

### Generator (llama-server)

Generator GGUF models use llama-server internally. The server management is automatic:

```csharp
using LMSupply.Generator;

// llama-server is automatically downloaded and started
await using var model = await LocalGenerator.LoadAsync("gguf:default");

// Server pooling: same model reuses existing server instance
await using var model2 = await LocalGenerator.LoadAsync("gguf:default");

// Generate text
await foreach (var token in model.GenerateAsync("Hello!"))
{
    Console.Write(token);
}
```

### Embedder (LLamaSharp)

Embedder GGUF models use LLamaSharp for native binding:

```csharp
using LMSupply.Embedder;

// LLamaSharp backend is automatically downloaded
await using var model = await LocalEmbedder.LoadAsync("nomic-ai/nomic-embed-text-v1.5-GGUF");

// Generate embeddings
float[] embedding = await model.EmbedAsync("Hello!");
```

### Direct Runtime Access

For advanced scenarios, you can access the runtime managers directly:

```csharp
using LMSupply.Llama;
using LMSupply.Runtime;

// LLamaSharp runtime (for Embedder)
var manager = LlamaRuntimeManager.Instance;
await manager.EnsureInitializedAsync(ExecutionProvider.Auto);

Console.WriteLine($"Backend: {manager.ActiveBackend}");
Console.WriteLine($"Binary Path: {manager.BinaryPath}");
Console.WriteLine(manager.GetEnvironmentSummary());
```

## Server Pooling (Generator)

The llama-server backend uses intelligent server pooling:

- **Shared instances**: Same model reuses existing server
- **Automatic cleanup**: Idle servers are stopped after timeout
- **Health monitoring**: Unhealthy servers are restarted
- **Resource management**: Memory-aware server allocation

## GPU Layer Calculation

The runtime manager can recommend GPU layer counts based on available VRAM:

```csharp
var manager = LlamaRuntimeManager.Instance;
await manager.EnsureInitializedAsync();

// Estimate layers for a 7B model (~14GB)
long modelSize = 14L * 1024 * 1024 * 1024;
int gpuLayers = manager.GetRecommendedGpuLayers(modelSize);

Console.WriteLine($"Recommended GPU layers: {gpuLayers}");
```

## Manual Backend Override

Force a specific backend:

```csharp
// For Embedder (LLamaSharp)
var options = new EmbedderOptions { Provider = ExecutionProvider.Cuda };
await using var model = await LocalEmbedder.LoadAsync("nomic-ai/nomic-embed-text-v1.5-GGUF", options);

// For Generator (llama-server)
var genOptions = new GeneratorOptions { Provider = ExecutionProvider.Cuda };
await using var model = await LocalGenerator.LoadAsync("gguf:default", genOptions);
```

## Consumer Packages

The following packages use LMSupply.Llama for GGUF support:

| Package | Runtime | Use Case |
|---------|---------|----------|
| **LMSupply.Generator** | llama-server | GGUF language models (e.g., Llama 3.2, Qwen 2.5) |
| **LMSupply.Embedder** | LLamaSharp | GGUF embedding models (e.g., nomic-embed-text) |

## Troubleshooting

### llama-server Issues (Generator)

1. **Server download failed**
   - Check network access to GitHub releases
   - Verify cache directory permissions

2. **Server won't start**
   - Check if port is already in use
   - Verify model file exists and is valid GGUF

3. **GPU not detected**
   - Install appropriate GPU drivers
   - For CUDA: Ensure NVIDIA drivers are installed
   - For Vulkan: Install Vulkan runtime

### LLamaSharp Issues (Embedder)

1. **Backend initialization failed**
   - Ensure network access for first-run download
   - Check cache directory permissions
   - For CUDA, verify NVIDIA drivers are installed

2. **Force CPU backend**
   ```csharp
   var options = new EmbedderOptions { Provider = ExecutionProvider.Cpu };
   await using var model = await LocalEmbedder.LoadAsync("nomic-ai/nomic-embed-text-v1.5-GGUF", options);
   ```

### Clear Cache

Delete cached binaries to force re-download:

```bash
# Windows - llama-server (Generator)
del /s /q %LOCALAPPDATA%\LMSupply\cache\llama-server

# Windows - LLamaSharp (Embedder)
del /s /q %LOCALAPPDATA%\LMSupply\cache\runtimes\llamasharp

# Linux/macOS - llama-server
rm -rf ~/.local/share/LMSupply/cache/llama-server

# Linux/macOS - LLamaSharp
rm -rf ~/.local/share/LMSupply/cache/runtimes/llamasharp
```

## Version Information

- **llama-server**: Downloaded from [llama.cpp GitHub releases](https://github.com/ggml-org/llama.cpp/releases)
- **LLamaSharp**: Version specified in project dependencies (NuGet)
