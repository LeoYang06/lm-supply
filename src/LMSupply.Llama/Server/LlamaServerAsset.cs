namespace LMSupply.Llama.Server;

/// <summary>
/// Represents a llama.cpp release asset for a specific platform and backend.
/// </summary>
public sealed record LlamaServerAsset
{
    /// <summary>
    /// Asset name from GitHub release.
    /// </summary>
    public required string Name { get; init; }

    /// <summary>
    /// Download URL.
    /// </summary>
    public required string DownloadUrl { get; init; }

    /// <summary>
    /// Build version (e.g., "b7898").
    /// </summary>
    public required string Version { get; init; }

    /// <summary>
    /// Target operating system.
    /// </summary>
    public required LlamaServerPlatform Platform { get; init; }

    /// <summary>
    /// GPU backend type.
    /// </summary>
    public required LlamaServerBackend Backend { get; init; }

    /// <summary>
    /// CPU architecture.
    /// </summary>
    public required LlamaServerArchitecture Architecture { get; init; }

    /// <summary>
    /// File size in bytes.
    /// </summary>
    public long? SizeBytes { get; init; }
}

/// <summary>
/// Supported platforms for llama-server.
/// </summary>
public enum LlamaServerPlatform
{
    Windows,
    Linux,
    MacOS
}

/// <summary>
/// Supported GPU backends for llama-server.
/// </summary>
public enum LlamaServerBackend
{
    /// <summary>CPU only (AVX2/AVX512 optimized).</summary>
    Cpu,

    /// <summary>Vulkan (cross-platform GPU).</summary>
    Vulkan,

    /// <summary>NVIDIA CUDA 12.4.</summary>
    Cuda12,

    /// <summary>NVIDIA CUDA 13.1.</summary>
    Cuda13,

    /// <summary>AMD HIP/ROCm.</summary>
    Hip,

    /// <summary>Intel SYCL.</summary>
    Sycl,

    /// <summary>Apple Metal (macOS).</summary>
    Metal
}

/// <summary>
/// CPU architecture.
/// </summary>
public enum LlamaServerArchitecture
{
    X64,
    Arm64
}
