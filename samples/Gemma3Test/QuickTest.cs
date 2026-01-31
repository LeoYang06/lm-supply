using System.Diagnostics;
using LMSupply;
using LMSupply.Generator;
using LMSupply.Generator.Models;
using LMSupply.Hardware;
using LMSupply.Llama.Server;

public static class QuickTest
{
    public static async Task RunAsync()
    {
        Console.WriteLine("=== LMSupply llama-server Backend Performance Test ===\n");
        Console.WriteLine("Using llama-server (llama.cpp HTTP server).");
        Console.WriteLine("Latest llama.cpp binaries are auto-downloaded.\n");

        // Display hardware profile for GPU selection verification
        var profile = HardwareProfile.Current;
        Console.WriteLine("=== Hardware Profile ===");
        Console.WriteLine($"GPU Vendor: {profile.GpuInfo.Vendor}");
        Console.WriteLine($"GPU Name: {profile.GpuInfo.DeviceName}");
        Console.WriteLine($"GPU Memory: {profile.GpuMemoryGB:F1} GB");
        Console.WriteLine($"Recommended Provider: {profile.RecommendedProvider}");
        Console.WriteLine($"Performance Tier: {profile.Tier}");
        Console.WriteLine();

        var modelPath = @"C:\Users\achunja\.lmstudio\models\lmstudio-community\gemma-3-4b-it-GGUF\gemma-3-4b-it-Q4_K_M.gguf";

        if (!File.Exists(modelPath))
        {
            Console.WriteLine($"Model not found: {modelPath}");
            return;
        }

        Console.WriteLine($"Model: {Path.GetFileName(modelPath)}");
        Console.WriteLine($"Size: {new FileInfo(modelPath).Length / (1024.0 * 1024.0 * 1024.0):F2} GB\n");

        Console.WriteLine("Downloading llama-server and loading model...");
        var loadSw = Stopwatch.StartNew();

        // Check for CPU-only mode via environment variable
        var forceCpu = Environment.GetEnvironmentVariable("LMSUPPLY_FORCE_CPU") == "1";

        // Use hardware-optimized settings for best first token latency
        var llamaOptions = LlamaOptions.GetOptimalForHardware();
        if (forceCpu)
        {
            llamaOptions = new LlamaOptions
            {
                GpuLayerCount = 0,
                BatchSize = 512,
                UBatchSize = 256,
                FlashAttention = false,
                UseMemoryMap = true
            };
        }

        var options = new GeneratorOptions
        {
            MaxContextLength = 8192,
            Provider = forceCpu ? ExecutionProvider.Cpu : ExecutionProvider.Auto,
            LlamaOptions = llamaOptions
        };

        Console.WriteLine($"Batch Size: {llamaOptions.BatchSize}, UBatch: {llamaOptions.UBatchSize}");

        if (forceCpu)
        {
            Console.WriteLine("*** CPU-only mode (LMSUPPLY_FORCE_CPU=1) ***\n");
        }

        // Progress callback for download status
        var progress = new Progress<DownloadProgress>(p =>
        {
            if (p.Phase == DownloadPhase.Downloading && p.TotalBytes > 0)
            {
                Console.Write($"\rDownloading: {p.FileName} - {p.BytesDownloaded * 100 / p.TotalBytes}%   ");
            }
            else if (p.Phase == DownloadPhase.Extracting)
            {
                Console.Write($"\rExtracting...                          ");
            }
        });

        var model = await LocalGenerator.LoadAsync(modelPath, options, progress);
        loadSw.Stop();
        Console.WriteLine($"\rLoad complete: {loadSw.Elapsed.TotalSeconds:F2}s                    \n");

        // Display model info
        var info = model.GetModelInfo();
        Console.WriteLine("=== Model Info ===");
        Console.WriteLine($"Backend: {info.ExecutionProvider}");
        Console.WriteLine($"Runtime Version: {info.RuntimeVersion ?? "unknown"}");
        Console.WriteLine($"GPU Active: {model.IsGpuActive}");
        Console.WriteLine($"Active Providers: {string.Join(", ", model.ActiveProviders)}");

        if (info.GgufMetadata != null)
        {
            Console.WriteLine($"Architecture: {info.Architecture}");
            Console.WriteLine($"Quantization: {info.QuantizationType}");
            Console.WriteLine($"Layers: {info.GgufMetadata.LayerCount}");
        }
        Console.WriteLine();

        // Display backend startup log for diagnostics
        if (!string.IsNullOrEmpty(info.BackendLog))
        {
            Console.WriteLine("=== llama-server Startup Log ===");
            // Show first 20 lines for key information
            var lines = info.BackendLog.Split('\n').Take(20);
            foreach (var line in lines)
            {
                Console.WriteLine(line);
            }
            Console.WriteLine("...\n");
        }

        // Warmup
        Console.WriteLine("Warming up...");
        await model.WarmupAsync();
        Console.WriteLine("Warmup complete.\n");

        // Test 1: LLM explanation test
        Console.WriteLine("=== Test 1: LLM Explanation ===");
        Console.WriteLine("LM Studio Reference: ~6 tok/sec (Intel Iris Xe Vulkan)\n");

        var messages = new[]
        {
            ChatMessage.User("What is an LLM? Explain briefly.")
        };

        Console.Write("User: What is an LLM? Explain briefly.\n\nAssistant: ");

        var sw = Stopwatch.StartNew();
        var firstTokenTime = TimeSpan.Zero;
        var tokenCount = 0;
        var response = new System.Text.StringBuilder();

        await foreach (var token in model.GenerateChatAsync(messages, new GenerationOptions
        {
            MaxTokens = 300,
            Temperature = 0.7f,
            TopP = 0.9f
        }))
        {
            if (tokenCount == 0)
            {
                firstTokenTime = sw.Elapsed;
            }
            Console.Write(token);
            response.Append(token);
            tokenCount++;
        }

        sw.Stop();

        Console.WriteLine("\n");
        Console.WriteLine("=== Performance Results ===");
        Console.WriteLine($"Total Tokens: {tokenCount}");
        Console.WriteLine($"Total Time: {sw.Elapsed.TotalSeconds:F2}s");
        Console.WriteLine($"First Token Latency: {firstTokenTime.TotalMilliseconds:F0}ms");
        Console.WriteLine($"Generation Speed: {tokenCount / sw.Elapsed.TotalSeconds:F2} tok/sec");
        Console.WriteLine();

        // Test 2: Simple hello test
        Console.WriteLine("=== Test 2: Hello (Simple Response) ===");

        messages = new[]
        {
            ChatMessage.User("hello")
        };

        Console.Write("User: hello\nAssistant: ");

        sw.Restart();
        firstTokenTime = TimeSpan.Zero;
        tokenCount = 0;

        await foreach (var token in model.GenerateChatAsync(messages, new GenerationOptions
        {
            MaxTokens = 128,
            Temperature = 0.7f
        }))
        {
            if (tokenCount == 0)
            {
                firstTokenTime = sw.Elapsed;
            }
            Console.Write(token);
            tokenCount++;
        }

        sw.Stop();

        Console.WriteLine("\n");
        Console.WriteLine("=== Performance Results ===");
        Console.WriteLine($"Total Tokens: {tokenCount}");
        Console.WriteLine($"Total Time: {sw.Elapsed.TotalSeconds:F2}s");
        Console.WriteLine($"First Token Latency: {firstTokenTime.TotalMilliseconds:F0}ms");
        Console.WriteLine($"Generation Speed: {tokenCount / sw.Elapsed.TotalSeconds:F2} tok/sec");
        Console.WriteLine();

        Console.WriteLine("=== Tests Complete ===");

        // Display pool status before dispose
        DisplayPoolStatus("Before model dispose");

        // Dispose model and show pool status
        await model.DisposeAsync();
        DisplayPoolStatus("After model dispose (server returned to pool)");

        // Test pool reuse - reload same model
        Console.WriteLine("\n=== Test 3: Server Pool Reuse ===");
        Console.WriteLine("Reloading same model to verify pool reuse.\n");

        var reloadSw = Stopwatch.StartNew();
        await using var model2 = await LocalGenerator.LoadAsync(modelPath, options);
        reloadSw.Stop();

        Console.WriteLine($"Reload Time: {reloadSw.Elapsed.TotalMilliseconds:F0}ms");
        Console.WriteLine($"First Load vs Reload: {loadSw.Elapsed.TotalSeconds:F2}s -> {reloadSw.Elapsed.TotalMilliseconds:F0}ms");
        DisplayPoolStatus("After reload");

        // Quick generation test with reused server
        Console.Write("\nQuick generation test: ");
        sw.Restart();
        tokenCount = 0;
        await foreach (var token in model2.GenerateChatAsync(
            new[] { ChatMessage.User("hi") },
            new GenerationOptions { MaxTokens = 10 }))
        {
            if (tokenCount == 0) firstTokenTime = sw.Elapsed;
            Console.Write(token);
            tokenCount++;
        }
        sw.Stop();
        Console.WriteLine($"\nFirst Token: {firstTokenTime.TotalMilliseconds:F0}ms, Speed: {tokenCount / sw.Elapsed.TotalSeconds:F1} tok/s");

        Console.WriteLine("\n=== All Tests Complete ===");
    }

    private static void DisplayPoolStatus(string label)
    {
        var status = LlamaServerPool.Instance.GetStatus();
        Console.WriteLine($"\n=== Server Pool Status: {label} ===");
        Console.WriteLine($"Total: {status.TotalServers}, Active: {status.ActiveServers}, Idle: {status.IdleServers}");
        foreach (var entry in status.Entries)
        {
            Console.WriteLine($"  - {Path.GetFileName(entry.ModelPath)} | {entry.Backend} | PID:{entry.ProcessId} | InUse:{entry.IsInUse}");
        }
    }
}
