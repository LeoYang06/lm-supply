using System.Diagnostics;
using System.Runtime.InteropServices;
using LLama.Native;
using LMSupply.Download;
using LMSupply.Runtime;

namespace LMSupply.Llama;

/// <summary>
/// Manages llama.cpp native binary download and configuration.
/// Follows LMSupply's on-demand philosophy: binaries are downloaded only when first needed.
/// Supports automatic runtime updates with background downloading.
/// </summary>
public sealed class LlamaRuntimeManager : IAsyncDisposable
{
    private const string PackageType = "llamasharp";

    private static readonly Lazy<LlamaRuntimeManager> _instance = new(() => new());

    /// <summary>
    /// Gets the singleton instance of the Llama runtime manager.
    /// </summary>
    public static LlamaRuntimeManager Instance => _instance.Value;

    private readonly SemaphoreSlim _initLock = new(1, 1);
    private readonly RuntimeUpdateOptions _updateOptions;
    private bool _initialized;
    private bool _disposed;
    private LlamaBackend _activeBackend;
    private string? _binaryPath;
    private string? _currentVersion;
    private PlatformInfo? _platform;

    /// <summary>
    /// Creates a new instance with default options.
    /// </summary>
    public LlamaRuntimeManager() : this(RuntimeUpdateOptions.Default)
    {
    }

    /// <summary>
    /// Creates a new instance with custom update options.
    /// </summary>
    public LlamaRuntimeManager(RuntimeUpdateOptions options)
    {
        _updateOptions = options;
    }

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
    /// Gets the current runtime version.
    /// </summary>
    public string? CurrentVersion => _currentVersion;

    /// <summary>
    /// Ensures the llama.cpp runtime is initialized with the specified backend.
    /// Downloads native binaries on first use with automatic fallback.
    /// </summary>
    /// <param name="provider">The execution provider to use. Auto will use fallback chain.</param>
    /// <param name="progress">Optional progress reporter for download operations.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task EnsureInitializedAsync(
        ExecutionProvider provider = ExecutionProvider.Auto,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        if (_initialized)
            return;

        await _initLock.WaitAsync(cancellationToken);
        try
        {
            if (_initialized)
                return;

            // 1. Detect platform and GPU
            _platform = EnvironmentDetector.DetectPlatform();
            var gpu = EnvironmentDetector.DetectGpu();

            // 2. Get backend fallback chain
            var chain = provider == ExecutionProvider.Auto
                ? GetBackendFallbackChain(_platform, gpu)
                : [DetermineBackend(provider, _platform, gpu)];

            Exception? lastException = null;

            // 3. Try each backend in the fallback chain
            foreach (var backend in chain)
            {
                try
                {
                    Debug.WriteLine($"[LlamaRuntimeManager] Trying backend: {backend}");

                    // Get current version from assembly
                    var currentVersion = GetLLamaSharpVersion();

                    // Get update service
                    var updateService = RuntimeUpdateService.GetInstance(PackageType, _updateOptions);
                    var packageId = GetBackendPackageId(backend);

                    // Download native binaries with update service
                    var binaryPath = await updateService.GetRuntimePathAsync(
                        packageId,
                        backend.ToString().ToLowerInvariant(),
                        _platform,
                        currentVersion,
                        (version, prog, ct) => DownloadNativeBinaryFromNuGetAsync(backend, _platform, version, prog, ct),
                        progress,
                        cancellationToken);

                    // Configure LLamaSharp to use downloaded binaries
                    ConfigureNativeLibrary(binaryPath, backend);

                    _activeBackend = backend;
                    _binaryPath = binaryPath;
                    _currentVersion = currentVersion;
                    _initialized = true;

                    Debug.WriteLine($"[LlamaRuntimeManager] Successfully initialized with backend: {backend}");
                    return;
                }
                catch (OperationCanceledException)
                {
                    throw; // Don't catch cancellation
                }
                catch (Exception ex) when (backend != LlamaBackend.Cpu)
                {
                    // Log and continue to next backend in chain
                    Debug.WriteLine($"[LlamaRuntimeManager] Backend '{backend}' failed: {ex.Message}. Trying next...");
                    lastException = ex;
                }
            }

            // Should not reach here since CPU is always in chain
            throw lastException ?? new InvalidOperationException("No backend available for LLamaSharp");
        }
        finally
        {
            _initLock.Release();
        }
    }

    /// <summary>
    /// Gets a prioritized list of backends to try based on detected hardware.
    /// Fallback chain: CUDA → Vulkan → Metal → CPU
    /// </summary>
    public static IReadOnlyList<LlamaBackend> GetBackendFallbackChain(PlatformInfo platform, GpuInfo gpu)
    {
        var chain = new List<LlamaBackend>();
        var isArm64 = platform.Architecture == Architecture.Arm64;

        // macOS ARM64: Metal first
        if (platform.IsMacOS && isArm64)
        {
            chain.Add(LlamaBackend.Metal);
        }

        // NVIDIA GPU: CUDA (try newest first, fallback to older)
        if (gpu.Vendor == GpuVendor.Nvidia)
        {
            if (gpu.CudaDriverVersionMajor >= 13)
                chain.Add(LlamaBackend.Cuda13);
            if (gpu.CudaDriverVersionMajor >= 12)
                chain.Add(LlamaBackend.Cuda12);
        }

        // AMD/Intel discrete GPU on Windows/Linux: Vulkan
        // Also try Vulkan for Unknown vendor with DirectML support (likely AMD/Intel)
        if (!platform.IsMacOS &&
            (gpu.Vendor == GpuVendor.Amd ||
             gpu.Vendor == GpuVendor.Intel ||
             (gpu.Vendor == GpuVendor.Unknown && gpu.DirectMLSupported)))
        {
            // Vulkan provides GPU acceleration for non-NVIDIA GPUs
            chain.Add(LlamaBackend.Vulkan);
        }

        // CPU always as final fallback (well-optimized with AVX2/AVX512)
        chain.Add(LlamaBackend.Cpu);

        return chain;
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
    /// Downloads the native binary from NuGet LLamaSharp.Backend.* packages.
    /// </summary>
    private async Task<string> DownloadNativeBinaryFromNuGetAsync(
        LlamaBackend backend,
        PlatformInfo platform,
        string? version,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        using var downloader = new LlamaNuGetDownloader();
        return await downloader.DownloadAsync(
            backend,
            platform,
            version: version,
            progress: progress,
            cancellationToken: cancellationToken);
    }

    /// <summary>
    /// Gets the NuGet package ID for a backend.
    /// </summary>
    private static string GetBackendPackageId(LlamaBackend backend) => backend switch
    {
        LlamaBackend.Cpu => "llamasharp.backend.cpu",
        LlamaBackend.Cuda12 => "llamasharp.backend.cuda12",
        LlamaBackend.Cuda13 => "llamasharp.backend.cuda12", // Use CUDA 12 for now
        LlamaBackend.Vulkan => "llamasharp.backend.vulkan",
        LlamaBackend.Metal => "llamasharp.backend.cpu", // Metal included in CPU package for macOS
        _ => "llamasharp.backend.cpu"
    };

    /// <summary>
    /// Gets the LLamaSharp version from the loaded assembly.
    /// </summary>
    private static string GetLLamaSharpVersion()
    {
        try
        {
            var assembly = typeof(LLama.LLamaWeights).Assembly;

            var infoVersionAttr = assembly.GetCustomAttributes(
                typeof(System.Reflection.AssemblyInformationalVersionAttribute), false)
                .OfType<System.Reflection.AssemblyInformationalVersionAttribute>()
                .FirstOrDefault();

            if (infoVersionAttr != null)
            {
                var ver = infoVersionAttr.InformationalVersion;
                var plusIdx = ver.IndexOf('+');
                if (plusIdx > 0)
                    ver = ver[..plusIdx];
                if (!string.IsNullOrEmpty(ver) && ver != "0.0.0")
                    return ver;
            }

            var version = assembly.GetName().Version;
            if (version != null && version.Major > 0)
            {
                return $"{version.Major}.{version.Minor}.{version.Build}";
            }
        }
        catch
        {
            // Ignore
        }

        return "0.25.0"; // Fallback version
    }

    /// <summary>
    /// Checks for runtime updates and applies them synchronously.
    /// Called during WarmupAsync to ensure latest runtime before inference.
    /// </summary>
    public async Task<RuntimeUpdateResult> CheckAndApplyUpdateAsync(
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        if (!_initialized || _platform is null)
        {
            return RuntimeUpdateResult.Failed("Runtime not initialized. Call EnsureInitializedAsync first.");
        }

        var updateService = RuntimeUpdateService.GetInstance(PackageType, _updateOptions);
        var packageId = GetBackendPackageId(_activeBackend);
        var currentVersion = _currentVersion ?? GetLLamaSharpVersion();

        var result = await updateService.CheckAndApplyUpdateAsync(
            packageId,
            _activeBackend.ToString().ToLowerInvariant(),
            _platform,
            currentVersion,
            (version, prog, ct) => DownloadNativeBinaryFromNuGetAsync(_activeBackend, _platform, version, prog, ct),
            progress,
            cancellationToken);

        if (result.Updated && !string.IsNullOrEmpty(result.RuntimePath))
        {
            // Re-configure with new runtime
            ConfigureNativeLibrary(result.RuntimePath, _activeBackend);
            _binaryPath = result.RuntimePath;
            _currentVersion = result.NewVersion;
        }

        return result;
    }

    /// <summary>
    /// Gets runtime update information for diagnostics.
    /// </summary>
    public async Task<RuntimeUpdateInfo> GetRuntimeUpdateInfoAsync(CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();

        if (!_initialized || _platform is null)
        {
            return new RuntimeUpdateInfo
            {
                InstalledVersion = "unknown",
                Provider = "not initialized"
            };
        }

        var updateService = RuntimeUpdateService.GetInstance(PackageType, _updateOptions);
        return await updateService.GetUpdateInfoAsync(
            _activeBackend.ToString().ToLowerInvariant(),
            _platform,
            cancellationToken);
    }

    /// <summary>
    /// Configures LLamaSharp to use the downloaded native binaries.
    /// </summary>
    private static void ConfigureNativeLibrary(string? binaryPath, LlamaBackend backend)
    {
        // Configure LLamaSharp to search in our download directory (if available)
        if (!string.IsNullOrEmpty(binaryPath) && Directory.Exists(binaryPath))
        {
            // Check if files are in variant subdirectories (avx, avx2, avx512, noavx)
            // Note: Using avx2 first because avx512 may have compatibility issues on some CPUs
            var variantDirs = new[] { "avx2", "avx", "noavx", "avx512" };
            string? selectedPath = null;

            foreach (var variant in variantDirs)
            {
                var variantPath = Path.Combine(binaryPath, variant);
                if (Directory.Exists(variantPath) && Directory.GetFiles(variantPath, "llama.*").Length > 0)
                {
                    selectedPath = variantPath;
                    break;
                }
            }

            // If no variant subdirectories, use the base path
            selectedPath ??= binaryPath;

            // Try to find llama library file and specify it directly
            var llamaLib = Directory.GetFiles(selectedPath, "llama.*").FirstOrDefault();
            if (llamaLib != null)
            {
                NativeLibraryConfig.All.WithLibrary(llamaLib, null);
            }
            else
            {
                NativeLibraryConfig.All.WithSearchDirectory(selectedPath);
            }
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
                Debug.WriteLine($"[LLamaSharp:{level}] {message}");
            }
        });
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
        var platform = EnvironmentDetector.DetectPlatform();
        var gpu = EnvironmentDetector.DetectGpu();

        var sb = new System.Text.StringBuilder();

        // Platform info
        sb.AppendLine($"Platform: {platform.OS} {platform.Architecture}");
        sb.AppendLine($"Runtime ID: {platform.RuntimeIdentifier}");

        // GPU info - check for discrete GPU or DirectML/CoreML support
        var hasGpu = gpu.Vendor != GpuVendor.Unknown || gpu.DirectMLSupported || gpu.CoreMLSupported;
        if (hasGpu)
        {
            sb.AppendLine($"GPU: {gpu.DeviceName ?? gpu.Vendor.ToString()}");
            if (gpu.TotalMemoryMB.HasValue)
                sb.AppendLine($"VRAM: {gpu.TotalMemoryMB.Value / 1024.0:F1} GB");
            if (gpu.Vendor != GpuVendor.Unknown)
                sb.AppendLine($"GPU Vendor: {gpu.Vendor}");
            if (gpu.DirectMLSupported)
                sb.AppendLine("DirectML: Supported");
            if (gpu.CoreMLSupported)
                sb.AppendLine("CoreML: Supported");
            if (gpu.CudaDriverVersionMajor.HasValue)
                sb.AppendLine($"CUDA Driver: {gpu.CudaDriverVersionMajor}.{gpu.CudaDriverVersionMinor}");
        }
        else
        {
            sb.AppendLine("GPU: Not detected (CPU only)");
        }

        // Current backend
        if (_initialized)
        {
            sb.AppendLine($"Active Backend: {_activeBackend}");
        }
        else
        {
            sb.AppendLine("Active Backend: Not initialized");
        }

        // Recommended fallback chain
        var fallbackChain = GetBackendFallbackChain(platform, gpu);
        sb.AppendLine($"Recommended Backends: {string.Join(" → ", fallbackChain)}");

        // Note about llama.cpp limitations
        if (gpu.DirectMLSupported && gpu.Vendor == GpuVendor.Unknown)
        {
            sb.AppendLine("Note: llama.cpp does not support DirectML. Using CPU for GGUF models.");
        }

        return sb.ToString().TrimEnd();
    }

    /// <summary>
    /// Disposes resources held by the runtime manager.
    /// Note: Singleton pattern means this is typically called only at application shutdown.
    /// </summary>
    public ValueTask DisposeAsync()
    {
        if (_disposed)
            return ValueTask.CompletedTask;

        _disposed = true;
        _initialized = false;
        _initLock.Dispose();

        return ValueTask.CompletedTask;
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }
}
