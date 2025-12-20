using LMSupply.ImageGenerator.Models;

namespace LMSupply.ImageGenerator;

/// <summary>
/// Entry point for loading and creating local image generator models.
/// </summary>
public static class LocalImageGenerator
{
    /// <summary>
    /// Loads an image generator model.
    /// </summary>
    /// <param name="modelIdOrPath">
    /// Model identifier. Can be:
    /// - An alias: "default", "fast", "quality"
    /// - A HuggingFace repo ID: "TheyCallMeHex/LCM-Dreamshaper-V7-ONNX"
    /// - A local directory path containing ONNX model files
    /// </param>
    /// <param name="options">Optional model loading options.</param>
    /// <param name="progress">Optional progress reporter for model download.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Loaded image generator model.</returns>
    /// <example>
    /// <code>
    /// // Load default model
    /// await using var generator = await LocalImageGenerator.LoadAsync("default");
    ///
    /// // Generate an image
    /// var result = await generator.GenerateAsync("A sunset over mountains");
    /// await result.SaveAsync("output.png");
    /// </code>
    /// </example>
    public static async Task<IImageGeneratorModel> LoadAsync(
        string modelIdOrPath,
        ImageGeneratorOptions? options = null,
        IProgress<float>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelIdOrPath);

        options ??= new ImageGeneratorOptions();

        // Resolve model alias to actual repo ID
        var modelDefinition = WellKnownImageModels.Resolve(modelIdOrPath);

        // TODO: Implement LCM pipeline loading in Phase 4
        // For now, throw NotImplementedException to indicate work in progress
        throw new NotImplementedException(
            $"LCM pipeline loading not yet implemented. " +
            $"Model: {modelDefinition.RepoId}, Steps: {modelDefinition.RecommendedSteps}");
    }

    /// <summary>
    /// Gets all available model aliases.
    /// </summary>
    /// <returns>Collection of valid model aliases.</returns>
    public static IReadOnlyCollection<string> GetAvailableAliases() =>
        WellKnownImageModels.GetAliases();
}
