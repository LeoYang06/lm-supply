using System.Diagnostics;
using LMSupply;
using LMSupply.Generator;
using LMSupply.Generator.Models;

// Quick test mode for comparison with LM Studio
if (args.Contains("--quick"))
{
    await QuickTest.RunAsync();
    return;
}

Console.WriteLine("=== LMSupply GGUF Performance Test (llama-server backend) ===\n");
Console.WriteLine("이제 LLamaSharp 대신 llama-server (llama.cpp HTTP server)를 사용합니다.");
Console.WriteLine("최신 llama.cpp 바이너리가 자동으로 다운로드됩니다.\n");

// Model path
var modelPath = @"C:\Users\achunja\.lmstudio\models\lmstudio-community\gemma-3-4b-it-GGUF\gemma-3-4b-it-Q4_K_M.gguf";

if (!File.Exists(modelPath))
{
    Console.WriteLine($"Model not found: {modelPath}");
    return;
}

Console.WriteLine($"Model: {Path.GetFileName(modelPath)}");
Console.WriteLine($"Size: {new FileInfo(modelPath).Length / (1024.0 * 1024.0 * 1024.0):F2} GB\n");

// Load model with optimized settings
Console.WriteLine("Loading llama-server and model...");
var loadSw = Stopwatch.StartNew();

var options = new GeneratorOptions
{
    MaxContextLength = 8192,
    Provider = ExecutionProvider.Auto,  // Auto-select best backend (Vulkan for broad compatibility)
    LlamaOptions = new LlamaOptions
    {
        GpuLayerCount = -1,           // All layers on GPU if available
        BatchSize = 512,
        FlashAttention = false,       // Safety first
        UseMemoryMap = true
    }
};

// Progress callback
var progress = new Progress<DownloadProgress>(p =>
{
    if (p.Phase == DownloadPhase.Downloading && p.TotalBytes > 0)
    {
        var pct = p.BytesDownloaded * 100 / p.TotalBytes;
        Console.Write($"\rDownloading {p.FileName}: {pct}%   ");
    }
});

await using var model = await LocalGenerator.LoadAsync(modelPath, options, progress);
loadSw.Stop();

Console.WriteLine($"\rModel loaded in {loadSw.Elapsed.TotalSeconds:F2}s          \n");

// Get model info
var info = model.GetModelInfo();
Console.WriteLine("=== Model Information ===");
Console.WriteLine($"Model ID: {info.ModelId}");
Console.WriteLine($"Context: {info.MaxContextLength}");
Console.WriteLine($"Chat Format: {info.ChatFormat}");
Console.WriteLine($"Backend: {info.ExecutionProvider}");
Console.WriteLine($"GPU Active: {model.IsGpuActive}");
Console.WriteLine($"Active Providers: {string.Join(", ", model.ActiveProviders)}");

// Display GGUF metadata if available
if (info.GgufMetadata != null)
{
    Console.WriteLine();
    Console.WriteLine("=== GGUF Metadata ===");
    Console.WriteLine($"Architecture: {info.Architecture}");
    Console.WriteLine($"Quantization: {info.QuantizationType}");
    Console.WriteLine($"Layers: {info.GgufMetadata.LayerCount}");
    Console.WriteLine($"Embedding Size: {info.GgufMetadata.EmbeddingLength}");
    if (info.GgufMetadata.HeadCount.HasValue)
        Console.WriteLine($"Attention Heads: {info.GgufMetadata.HeadCount}" +
            (info.GgufMetadata.HeadCountKv.HasValue ? $" (KV: {info.GgufMetadata.HeadCountKv})" : ""));
    Console.WriteLine($"Summary: {info.GetArchitectureSummary()}");
}

// Resource estimation
Console.WriteLine();
Console.WriteLine("=== Resource Estimation ===");
var resourceEstimate = await MemoryEstimator.EstimateFromGgufFileAsync(
    modelPath,
    contextLength: 8192);
if (resourceEstimate != null)
{
    Console.WriteLine(resourceEstimate.GetSummary());
}
Console.WriteLine();

// Warmup
Console.WriteLine("Warming up...");
await model.WarmupAsync();
Console.WriteLine("Warmup complete.\n");

// ============================================================================
// Test 1: Same as LM Studio - "hello" prompt
// ============================================================================
Console.WriteLine("=== Test 1: LM Studio Comparison (hello) ===");
Console.WriteLine("LM Studio Reference: ~86 tok/sec (Vulkan)\n");

var messages = new[]
{
    ChatMessage.User("hello")
};

Console.Write("User: hello\nAssistant: ");

var sw = Stopwatch.StartNew();
var firstTokenTime = TimeSpan.Zero;
var tokenCount = 0;
var response = "";

await foreach (var token in model.GenerateChatAsync(messages, new GenerationOptions
{
    MaxTokens = 128,
    Temperature = 0.7f,
    TopP = 0.9f
}))
{
    if (tokenCount == 0)
        firstTokenTime = sw.Elapsed;
    Console.Write(token);
    response += token;
    tokenCount++;
}

sw.Stop();

Console.WriteLine("\n");
Console.WriteLine($"--- Performance ---");
Console.WriteLine($"Tokens: {tokenCount}");
Console.WriteLine($"Time: {sw.Elapsed.TotalSeconds:F2}s");
Console.WriteLine($"First Token: {firstTokenTime.TotalMilliseconds:F0}ms");
Console.WriteLine($"Speed: {tokenCount / sw.Elapsed.TotalSeconds:F2} tok/sec");
Console.WriteLine();

// ============================================================================
// Test 2: Korean language test
// ============================================================================
Console.WriteLine("=== Test 2: Korean Language ===\n");

var koreanMessages = new[]
{
    ChatMessage.User("안녕하세요! 자기소개 해주세요.")
};

Console.Write("User: 안녕하세요! 자기소개 해주세요.\nAssistant: ");

sw.Restart();
firstTokenTime = TimeSpan.Zero;
tokenCount = 0;

await foreach (var token in model.GenerateChatAsync(koreanMessages, new GenerationOptions
{
    MaxTokens = 150,
    Temperature = 0.7f
}))
{
    if (tokenCount == 0)
        firstTokenTime = sw.Elapsed;
    Console.Write(token);
    tokenCount++;
}

sw.Stop();

Console.WriteLine("\n");
Console.WriteLine($"--- Performance ---");
Console.WriteLine($"Tokens: {tokenCount}");
Console.WriteLine($"Time: {sw.Elapsed.TotalSeconds:F2}s");
Console.WriteLine($"First Token: {firstTokenTime.TotalMilliseconds:F0}ms");
Console.WriteLine($"Speed: {tokenCount / sw.Elapsed.TotalSeconds:F2} tok/sec");
Console.WriteLine();

// ============================================================================
// Test 3: Longer generation
// ============================================================================
Console.WriteLine("=== Test 3: Long Generation ===\n");

var longMessages = new[]
{
    ChatMessage.System("You are a helpful assistant."),
    ChatMessage.User("Explain quantum computing in simple terms.")
};

Console.Write("User: Explain quantum computing in simple terms.\nAssistant: ");

sw.Restart();
firstTokenTime = TimeSpan.Zero;
tokenCount = 0;

await foreach (var token in model.GenerateChatAsync(longMessages, new GenerationOptions
{
    MaxTokens = 256,
    Temperature = 0.5f
}))
{
    if (tokenCount == 0)
        firstTokenTime = sw.Elapsed;
    Console.Write(token);
    tokenCount++;
}

sw.Stop();

Console.WriteLine("\n");
Console.WriteLine($"--- Performance ---");
Console.WriteLine($"Tokens: {tokenCount}");
Console.WriteLine($"Time: {sw.Elapsed.TotalSeconds:F2}s");
Console.WriteLine($"First Token: {firstTokenTime.TotalMilliseconds:F0}ms");
Console.WriteLine($"Speed: {tokenCount / sw.Elapsed.TotalSeconds:F2} tok/sec");
Console.WriteLine();

Console.WriteLine("=== Test Complete ===");
