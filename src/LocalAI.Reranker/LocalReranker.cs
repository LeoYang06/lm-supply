using LocalAI.Reranker.Models;

namespace LocalAI.Reranker;

/// <summary>
/// Main entry point for loading and using reranker models.
/// </summary>
public static class LocalReranker
{
    /// <summary>
    /// Loads a reranker model by name or path.
    /// </summary>
    /// <param name="modelIdOrPath">
    /// Either a model alias (e.g., "default", "quality", "fast"),
    /// a HuggingFace model ID (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    /// or a local path to an ONNX model file.
    /// </param>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="progress">Optional progress reporting for downloads.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A loaded reranker ready for inference.</returns>
    public static async Task<IRerankerModel> LoadAsync(
        string modelIdOrPath,
        RerankerOptions? options = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new RerankerOptions();
        options.ModelId = modelIdOrPath;

        var reranker = new Reranker(options);

        // Eagerly initialize and warm up the model
        await reranker.WarmupAsync(cancellationToken);

        return reranker;
    }

    /// <summary>
    /// Gets a list of pre-configured model aliases available for use.
    /// </summary>
    /// <returns>Available model aliases.</returns>
    public static IEnumerable<string> GetAvailableModels()
    {
        return ModelRegistry.Default.GetAliases();
    }

    /// <summary>
    /// Gets all registered model information.
    /// </summary>
    /// <returns>Collection of model information.</returns>
    public static IEnumerable<ModelInfo> GetAllModels()
    {
        return ModelRegistry.Default.GetAll();
    }
}
