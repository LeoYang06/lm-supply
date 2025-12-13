namespace LocalAI.Embedder.Utils;

/// <summary>
/// Registry of pre-configured embedding models.
/// </summary>
internal static class ModelRegistry
{
    private static readonly Dictionary<string, ModelInfo> _models = new(StringComparer.OrdinalIgnoreCase)
    {
        ["all-MiniLM-L6-v2"] = new ModelInfo
        {
            RepoId = "sentence-transformers/all-MiniLM-L6-v2",
            Dimensions = 384,
            MaxSequenceLength = 256,
            PoolingMode = PoolingMode.Mean,
            DoLowerCase = true,
            Description = "Fast, good quality, English",
            Subfolder = "onnx"
        },
        ["all-mpnet-base-v2"] = new ModelInfo
        {
            RepoId = "sentence-transformers/all-mpnet-base-v2",
            Dimensions = 768,
            MaxSequenceLength = 384,
            PoolingMode = PoolingMode.Mean,
            DoLowerCase = true,
            Description = "Higher quality, English",
            Subfolder = "onnx"
        },
        ["bge-small-en-v1.5"] = new ModelInfo
        {
            RepoId = "BAAI/bge-small-en-v1.5",
            Dimensions = 384,
            MaxSequenceLength = 512,
            PoolingMode = PoolingMode.Cls,
            DoLowerCase = true,
            Description = "BAAI, English",
            Subfolder = "onnx"
        },
        ["bge-base-en-v1.5"] = new ModelInfo
        {
            RepoId = "BAAI/bge-base-en-v1.5",
            Dimensions = 768,
            MaxSequenceLength = 512,
            PoolingMode = PoolingMode.Cls,
            DoLowerCase = true,
            Description = "BAAI, English, higher quality",
            Subfolder = "onnx"
        },
        ["multilingual-e5-small"] = new ModelInfo
        {
            RepoId = "intfloat/multilingual-e5-small",
            Dimensions = 384,
            MaxSequenceLength = 512,
            PoolingMode = PoolingMode.Mean,
            DoLowerCase = false,
            Description = "Multilingual",
            Subfolder = "onnx"
        },
        ["multilingual-e5-base"] = new ModelInfo
        {
            RepoId = "intfloat/multilingual-e5-base",
            Dimensions = 768,
            MaxSequenceLength = 512,
            PoolingMode = PoolingMode.Mean,
            DoLowerCase = false,
            Description = "Multilingual, higher quality",
            Subfolder = "onnx"
        }
    };

    /// <summary>
    /// Tries to get model info by model ID.
    /// </summary>
    public static bool TryGetModel(string modelId, out ModelInfo? info)
    {
        return _models.TryGetValue(modelId, out info);
    }

    /// <summary>
    /// Gets all available model IDs.
    /// </summary>
    public static IEnumerable<string> GetAvailableModels() => _models.Keys;
}

/// <summary>
/// Configuration information for a pre-configured model.
/// </summary>
public sealed record ModelInfo
{
    public required string RepoId { get; init; }
    public required int Dimensions { get; init; }
    public required int MaxSequenceLength { get; init; }
    public required PoolingMode PoolingMode { get; init; }
    public required bool DoLowerCase { get; init; }
    public string? Description { get; init; }
    public string? Subfolder { get; init; }
}
