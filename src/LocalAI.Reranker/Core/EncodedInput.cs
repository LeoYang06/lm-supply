namespace LocalAI.Reranker.Core;

/// <summary>
/// Represents tokenized input for the model.
/// </summary>
internal readonly struct EncodedInput
{
    /// <summary>
    /// Gets the token IDs.
    /// </summary>
    public long[] InputIds { get; init; }

    /// <summary>
    /// Gets the attention mask (1 for real tokens, 0 for padding).
    /// </summary>
    public long[] AttentionMask { get; init; }

    /// <summary>
    /// Gets the token type IDs (0 for query, 1 for document).
    /// </summary>
    public long[] TokenTypeIds { get; init; }

    /// <summary>
    /// Gets the actual sequence length before padding.
    /// </summary>
    public int Length { get; init; }

    /// <summary>
    /// Creates a new encoded input.
    /// </summary>
    public EncodedInput(long[] inputIds, long[] attentionMask, long[] tokenTypeIds, int length)
    {
        InputIds = inputIds;
        AttentionMask = attentionMask;
        TokenTypeIds = tokenTypeIds;
        Length = length;
    }
}

/// <summary>
/// Represents a batch of encoded inputs.
/// </summary>
internal sealed class EncodedBatch
{
    /// <summary>
    /// Gets the flattened input IDs [batch_size * seq_length].
    /// </summary>
    public long[] InputIds { get; }

    /// <summary>
    /// Gets the flattened attention mask [batch_size * seq_length].
    /// </summary>
    public long[] AttentionMask { get; }

    /// <summary>
    /// Gets the flattened token type IDs [batch_size * seq_length].
    /// </summary>
    public long[] TokenTypeIds { get; }

    /// <summary>
    /// Gets the batch size.
    /// </summary>
    public int BatchSize { get; }

    /// <summary>
    /// Gets the sequence length.
    /// </summary>
    public int SequenceLength { get; }

    /// <summary>
    /// Creates a new encoded batch.
    /// </summary>
    public EncodedBatch(int batchSize, int sequenceLength)
    {
        BatchSize = batchSize;
        SequenceLength = sequenceLength;

        var totalSize = batchSize * sequenceLength;
        InputIds = new long[totalSize];
        AttentionMask = new long[totalSize];
        TokenTypeIds = new long[totalSize];
    }

    /// <summary>
    /// Sets the encoded input at the specified batch index.
    /// </summary>
    public void SetInput(int batchIndex, EncodedInput input)
    {
        var offset = batchIndex * SequenceLength;
        Array.Copy(input.InputIds, 0, InputIds, offset, SequenceLength);
        Array.Copy(input.AttentionMask, 0, AttentionMask, offset, SequenceLength);
        Array.Copy(input.TokenTypeIds, 0, TokenTypeIds, offset, SequenceLength);
    }
}
