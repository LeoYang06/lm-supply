using System.Runtime.InteropServices;
using LLama.Native;
using LMSupply.Runtime;

namespace LMSupply.Generator.Internal.Llama;

/// <summary>
/// Manages llama.cpp native binary download and configuration.
/// Follows LMSupply's on-demand philosophy: binaries are downloaded only when first needed.
/// </summary>
public sealed class LlamaRuntimeManager
{
    private static readonly Lazy<LlamaRuntimeManager> _instance = new(() => new());

    /// <summary>
    /// Gets the singleton instance of the Llama runtime manager.
    /// </summary>
    public static LlamaRuntimeManager Instance => _instance.Value;

    private readonly SemaphoreSlim _initLock = new(1, 1);
    private bool _initialized;
    private LlamaBackend _activeBackend;
    private string? _binaryPath;

    /// <summary>
    /// Gets whether the runtime has been initialized.
    /// </summary>
    public bool IsInitialized => _initialized;

    /// <summary>
    /// Gets the currently active backend.
    /// </summary>
    public LlamaBackend ActiveBackend => _activeBackend;

    /// <summary>
    /// Gets the path to the loaded native binaries.
    /// </summary>
    public string? BinaryPath => _binaryPath;

    /// <summary>
    /// Ensures the llama.cpp runtime is initialized with the specified backend.
    /// Downloads native binaries on first use.
    /// </summary>
    /// <param name="provider">The execution provider to use. Auto will select the best available.</param>
    /// <param name="progress">Optional progress reporter for download operations.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task EnsureInitializedAsync(
        ExecutionProvider provider = ExecutionProvider.Auto,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        if (_initialized)
            return;

        await _initLock.WaitAsync(cancellationToken);
        try
        {
            if (_initialized)
                return;

            // 1. Detect platform and GPU
            var platform = EnvironmentDetector.DetectPlatform();
            var gpu = EnvironmentDetector.DetectGpu();

            // 2. Determine the best backend
            var backend = DetermineBackend(provider, platform, gpu);

            // 3. Try to download native binaries (if not cached)
            string? binaryPath = null;
            try
            {
                binaryPath = await DownloadNativeBinaryAsync(
                    backend, platform, progress, cancellationToken);
            }
            catch (Exception ex)
            {
                // Download failed - will rely on LLamaSharp.Backend.* packages or system libraries
                System.Diagnostics.Debug.WriteLine($"[LlamaRuntimeManager] Binary download failed: {ex.Message}");
                System.Diagnostics.Debug.WriteLine("[LlamaRuntimeManager] Falling back to NuGet package or system libraries");
            }

            // 4. Configure LLamaSharp to use downloaded binaries or fallback
            ConfigureNativeLibrary(binaryPath, backend);

            _activeBackend = backend;
            _binaryPath = binaryPath;
            _initialized = true;
        }
        finally
        {
            _initLock.Release();
        }
    }

    /// <summary>
    /// Determines the best backend based on provider and detected hardware.
    /// </summary>
    private static LlamaBackend DetermineBackend(
        ExecutionProvider provider,
        PlatformInfo platform,
        GpuInfo gpu)
    {
        // Explicit provider selection
        if (provider != ExecutionProvider.Auto)
        {
            return provider switch
            {
                ExecutionProvider.Cuda when gpu.CudaDriverVersionMajor >= 13 => LlamaBackend.Cuda13,
                ExecutionProvider.Cuda when gpu.CudaDriverVersionMajor >= 12 => LlamaBackend.Cuda12,
                ExecutionProvider.Cuda => LlamaBackend.Cuda12, // fallback to CUDA 12
                ExecutionProvider.DirectML => LlamaBackend.Vulkan, // Use Vulkan as DirectML alternative
                ExecutionProvider.CoreML when platform.IsMacOS => LlamaBackend.Metal,
                _ => LlamaBackend.Cpu
            };
        }

        // Auto selection based on hardware
        var isArm64 = platform.Architecture == Architecture.Arm64;

        if (platform.IsMacOS && isArm64)
        {
            // Apple Silicon - always use Metal
            return LlamaBackend.Metal;
        }

        if (gpu.Vendor == GpuVendor.Nvidia)
        {
            // NVIDIA GPU - use CUDA (most reliable for llama.cpp)
            if (gpu.CudaDriverVersionMajor >= 13)
                return LlamaBackend.Cuda13;
            if (gpu.CudaDriverVersionMajor >= 12)
                return LlamaBackend.Cuda12;
        }

        if (gpu.Vendor == GpuVendor.Amd && !platform.IsMacOS)
        {
            // AMD discrete GPU - try Vulkan (but may be unstable)
            // Only use if explicitly requested via ExecutionProvider
            // For auto mode, prefer CPU for stability
        }

        // Intel integrated GPUs and other cases - use CPU for stability
        // Intel Iris Xe, Intel UHD, etc. often have issues with Vulkan backend
        // CPU backend is well-optimized with AVX2/AVX512 support
        return LlamaBackend.Cpu;
    }

    /// <summary>
    /// Downloads the native binary for the specified backend.
    /// </summary>
    private async Task<string> DownloadNativeBinaryAsync(
        LlamaBackend backend,
        PlatformInfo platform,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        var downloader = new LlamaBinaryDownloader();
        return await downloader.DownloadAsync(
            backend,
            platform,
            progress: progress,
            cancellationToken: cancellationToken);
    }

    /// <summary>
    /// Configures LLamaSharp to use the downloaded native binaries.
    /// </summary>
    private static void ConfigureNativeLibrary(string? binaryPath, LlamaBackend backend)
    {
        // Configure LLamaSharp to search in our download directory (if available)
        if (!string.IsNullOrEmpty(binaryPath) && Directory.Exists(binaryPath))
        {
            NativeLibraryConfig.All.WithSearchDirectory(binaryPath);
        }

        // Enable auto-fallback to search system paths and NuGet package paths
        // This allows using LLamaSharp.Backend.* packages as fallback
        NativeLibraryConfig.All.WithAutoFallback(true);

        // Configure backend-specific settings
        switch (backend)
        {
            case LlamaBackend.Cuda12:
            case LlamaBackend.Cuda13:
                NativeLibraryConfig.All.WithCuda();
                break;

            case LlamaBackend.Vulkan:
                NativeLibraryConfig.All.WithVulkan();
                break;

            case LlamaBackend.Metal:
                // Metal is automatically used on macOS arm64
                break;

            case LlamaBackend.Cpu:
            default:
                // CPU is the default
                break;
        }

        // Set logging level (use WithLogCallback for custom logging)
        NativeLibraryConfig.All.WithLogCallback((level, message) =>
        {
            if (level >= LLamaLogLevel.Warning)
            {
                System.Diagnostics.Debug.WriteLine($"[LLamaSharp:{level}] {message}");
            }
        });

        // Register with NativeLoader for dependency resolution (if we have a binary path)
        if (!string.IsNullOrEmpty(binaryPath) && Directory.Exists(binaryPath))
        {
            NativeLoader.Instance.RegisterDirectory(binaryPath, preload: true, primaryLibrary: "llama");
        }
    }

    /// <summary>
    /// Gets the recommended GPU layer count based on available VRAM.
    /// </summary>
    /// <param name="modelSizeBytes">The model size in bytes.</param>
    /// <returns>Recommended number of layers to offload to GPU.</returns>
    public int GetRecommendedGpuLayers(long modelSizeBytes)
    {
        if (_activeBackend == LlamaBackend.Cpu)
            return 0;

        var gpu = EnvironmentDetector.DetectGpu();
        var vramMB = gpu.TotalMemoryMB ?? 0;
        if (vramMB <= 0)
            return 0;

        var vramBytes = vramMB * 1024L * 1024L;

        // Reserve ~2GB for system overhead
        var availableVram = vramBytes - (2L * 1024 * 1024 * 1024);
        if (availableVram <= 0)
            return 0;

        // Estimate layers that can fit in VRAM
        // Typical 7B model: ~32 layers, ~14GB total
        // Each layer is roughly modelSize / numLayers
        const int typicalLayerCount = 32;
        var bytesPerLayer = modelSizeBytes / typicalLayerCount;

        if (bytesPerLayer <= 0)
            return typicalLayerCount; // Assume full offload for small models

        var layersInVram = (int)(availableVram / bytesPerLayer);
        return Math.Clamp(layersInVram, 0, 999); // LLamaSharp uses 999 as "all layers"
    }

    /// <summary>
    /// Gets environment summary for diagnostics.
    /// </summary>
    public string GetEnvironmentSummary()
    {
        if (!_initialized)
            return "LlamaRuntimeManager not initialized";

        return $"""
            Llama Backend: {_activeBackend}
            Binary Path: {_binaryPath}
            Initialized: {_initialized}
            """;
    }
}
