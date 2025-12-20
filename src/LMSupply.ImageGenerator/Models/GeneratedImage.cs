namespace LMSupply.ImageGenerator.Models;

/// <summary>
/// Represents a generated image result.
/// </summary>
public sealed class GeneratedImage
{
    /// <summary>
    /// Raw image data in PNG format.
    /// </summary>
    public required byte[] ImageData { get; init; }

    /// <summary>
    /// Width of the generated image in pixels.
    /// </summary>
    public required int Width { get; init; }

    /// <summary>
    /// Height of the generated image in pixels.
    /// </summary>
    public required int Height { get; init; }

    /// <summary>
    /// The seed used for generation (for reproducibility).
    /// </summary>
    public required int Seed { get; init; }

    /// <summary>
    /// Time taken to generate the image.
    /// </summary>
    public required TimeSpan GenerationTime { get; init; }

    /// <summary>
    /// Number of inference steps used.
    /// </summary>
    public required int Steps { get; init; }

    /// <summary>
    /// The prompt used to generate this image.
    /// </summary>
    public required string Prompt { get; init; }

    /// <summary>
    /// Saves the image to a file.
    /// </summary>
    /// <param name="filePath">Path to save the image.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task SaveAsync(string filePath, CancellationToken cancellationToken = default)
    {
        await File.WriteAllBytesAsync(filePath, ImageData, cancellationToken);
    }

    /// <summary>
    /// Gets the image data as a memory stream.
    /// </summary>
    public MemoryStream ToStream() => new(ImageData, writable: false);
}
