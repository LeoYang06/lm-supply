using LMSupply.Hardware;

namespace LMSupply.Generator;

/// <summary>
/// Quantization types for KV cache memory optimization.
/// </summary>
public enum KvCacheQuantizationType
{
    /// <summary>
    /// 16-bit floating point (default, highest quality).
    /// </summary>
    F16 = 0,

    /// <summary>
    /// 8-bit quantization. Good balance of memory savings and quality.
    /// Reduces KV cache memory by ~50% with minimal quality loss.
    /// </summary>
    Q8_0 = 1,

    /// <summary>
    /// 4-bit quantization. Maximum memory savings.
    /// Reduces KV cache memory by ~75% but may affect output quality.
    /// </summary>
    Q4_0 = 2,

    /// <summary>
    /// 32-bit floating point. Maximum quality, highest memory usage.
    /// </summary>
    F32 = 3
}

/// <summary>
/// Advanced configuration options for GGUF model loading and inference via LLamaSharp.
/// These options provide fine-grained control over llama.cpp backend behavior.
/// </summary>
public sealed class LlamaOptions
{
    /// <summary>
    /// Gets or sets the number of layers to offload to GPU.
    /// -1 = All layers on GPU (default for GPU providers)
    /// 0 = CPU only
    /// N = Offload N layers to GPU
    /// </summary>
    public int? GpuLayerCount { get; set; }

    /// <summary>
    /// Gets or sets the batch size for prompt processing.
    /// Higher values improve throughput but use more memory.
    /// Defaults to 512.
    /// </summary>
    public uint? BatchSize { get; set; }

    /// <summary>
    /// Gets or sets the RoPE frequency base for context extension.
    /// Use with RoPE-scaling-aware models for extended context lengths.
    /// </summary>
    public float? RopeFrequencyBase { get; set; }

    /// <summary>
    /// Gets or sets the RoPE frequency scale factor.
    /// Use with RoPE-scaling-aware models for extended context lengths.
    /// </summary>
    public float? RopeFrequencyScale { get; set; }

    /// <summary>
    /// Gets or sets whether to use Flash Attention for improved performance.
    /// Requires compatible GPU (CUDA compute capability 7.0+).
    /// Defaults to false for compatibility.
    /// </summary>
    public bool? FlashAttention { get; set; }

    /// <summary>
    /// Gets or sets whether to use memory mapping for model loading.
    /// Enables faster model loading and reduced memory usage when the same model
    /// is loaded multiple times. Defaults to true.
    /// </summary>
    public bool? UseMemoryMap { get; set; }

    /// <summary>
    /// Gets or sets whether to lock model memory to prevent swapping.
    /// Improves inference latency but may require elevated privileges.
    /// Defaults to false.
    /// </summary>
    public bool? UseMemoryLock { get; set; }

    /// <summary>
    /// Gets or sets the main GPU index for multi-GPU systems.
    /// 0-based index of the GPU to use as the primary device.
    /// </summary>
    public int? MainGpu { get; set; }

    /// <summary>
    /// Gets or sets the number of threads for CPU computation.
    /// Defaults to system-detected optimal thread count.
    /// </summary>
    public int? Threads { get; set; }

    /// <summary>
    /// Gets or sets the quantization type for KV cache keys.
    /// Q8_0 offers good memory/quality balance, Q4_0 for maximum memory savings.
    /// Defaults to F16 (no quantization).
    /// </summary>
    public KvCacheQuantizationType? TypeK { get; set; }

    /// <summary>
    /// Gets or sets the quantization type for KV cache values.
    /// Q8_0 offers good memory/quality balance, Q4_0 for maximum memory savings.
    /// Defaults to F16 (no quantization).
    /// </summary>
    public KvCacheQuantizationType? TypeV { get; set; }

    /// <summary>
    /// Gets or sets the physical batch size for prompt processing.
    /// Smaller than BatchSize, controls memory usage during processing.
    /// Defaults to 512.
    /// </summary>
    public uint? UBatchSize { get; set; }

    /// <summary>
    /// Gets optimal LlamaOptions based on current hardware profile.
    /// </summary>
    /// <returns>LlamaOptions configured for optimal performance on detected hardware.</returns>
    public static LlamaOptions GetOptimalForHardware()
    {
        var profile = HardwareProfile.Current;

        return profile.Tier switch
        {
            PerformanceTier.Ultra => new LlamaOptions
            {
                GpuLayerCount = -1,          // All layers on GPU
                BatchSize = 2048,            // Large batch for throughput
                UBatchSize = 512,            // Physical batch size
                FlashAttention = true,       // Enable if supported
                UseMemoryMap = true,
                UseMemoryLock = false,       // Don't require elevated privileges
                TypeK = KvCacheQuantizationType.Q8_0,  // Quantized KV cache for memory savings
                TypeV = KvCacheQuantizationType.Q8_0
            },
            PerformanceTier.High => new LlamaOptions
            {
                GpuLayerCount = -1,
                BatchSize = 1024,
                UBatchSize = 512,
                FlashAttention = true,
                UseMemoryMap = true,
                UseMemoryLock = false,
                TypeK = KvCacheQuantizationType.Q8_0,
                TypeV = KvCacheQuantizationType.Q8_0
            },
            PerformanceTier.Medium => new LlamaOptions
            {
                GpuLayerCount = -1,
                BatchSize = 512,
                UBatchSize = 256,
                FlashAttention = false,      // Conservative for mid-range
                UseMemoryMap = true,
                UseMemoryLock = false,
                TypeK = KvCacheQuantizationType.Q4_0,  // More aggressive quantization for memory
                TypeV = KvCacheQuantizationType.Q4_0
            },
            _ => new LlamaOptions              // Low tier
            {
                GpuLayerCount = 0,            // CPU only for safety
                BatchSize = 256,             // Smaller batch for memory
                UBatchSize = 128,
                FlashAttention = false,
                UseMemoryMap = true,
                UseMemoryLock = false
                // No KV cache quantization for CPU (default F16)
            }
        };
    }
}
