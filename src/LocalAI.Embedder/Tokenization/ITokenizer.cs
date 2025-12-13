namespace LocalAI.Embedder.Tokenization;

/// <summary>
/// Internal interface for text tokenization.
/// </summary>
internal interface ITokenizer
{
    /// <summary>
    /// Encodes text into token IDs and attention mask.
    /// </summary>
    /// <param name="text">The text to encode.</param>
    /// <param name="maxLength">Maximum sequence length.</param>
    /// <returns>Token IDs and attention mask arrays.</returns>
    (long[] InputIds, long[] AttentionMask) Encode(string text, int maxLength);

    /// <summary>
    /// Encodes multiple texts into batched token IDs and attention masks.
    /// </summary>
    (long[][] InputIds, long[][] AttentionMasks) EncodeBatch(IReadOnlyList<string> texts, int maxLength);
}
