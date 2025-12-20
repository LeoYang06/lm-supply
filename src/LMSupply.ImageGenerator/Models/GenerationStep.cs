namespace LMSupply.ImageGenerator.Models;

/// <summary>
/// Represents a step in the image generation process for streaming progress.
/// </summary>
public sealed class GenerationStep
{
    /// <summary>
    /// Current step number (1-based).
    /// </summary>
    public required int StepNumber { get; init; }

    /// <summary>
    /// Total number of steps.
    /// </summary>
    public required int TotalSteps { get; init; }

    /// <summary>
    /// Progress as a percentage (0-100).
    /// </summary>
    public float Progress => TotalSteps > 0 ? (float)StepNumber / TotalSteps * 100 : 0;

    /// <summary>
    /// Whether this is the final step.
    /// </summary>
    public bool IsFinal => StepNumber == TotalSteps;

    /// <summary>
    /// Preview image data (PNG format) if available.
    /// Only populated when GeneratePreviews is enabled.
    /// </summary>
    public byte[]? PreviewData { get; init; }

    /// <summary>
    /// Whether a preview image is available for this step.
    /// </summary>
    public bool HasPreview => PreviewData is { Length: > 0 };

    /// <summary>
    /// The final generated image (only available on the last step).
    /// </summary>
    public GeneratedImage? FinalImage { get; init; }

    /// <summary>
    /// Elapsed time since generation started.
    /// </summary>
    public required TimeSpan Elapsed { get; init; }
}
