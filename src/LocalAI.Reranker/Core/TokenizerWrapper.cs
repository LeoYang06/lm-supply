using System.Text;
using System.Text.Json;
using Microsoft.ML.Tokenizers;

namespace LocalAI.Reranker.Core;

/// <summary>
/// Wraps tokenizer functionality for cross-encoder models.
/// </summary>
internal sealed class TokenizerWrapper
{
    private readonly Tokenizer _tokenizer;
    private readonly int _clsTokenId;
    private readonly int _sepTokenId;
    private readonly int _padTokenId;
    private readonly int _maxLength;

    /// <summary>
    /// Gets the maximum sequence length.
    /// </summary>
    public int MaxLength => _maxLength;

    private TokenizerWrapper(Tokenizer tokenizer, int clsTokenId, int sepTokenId, int padTokenId, int maxLength)
    {
        _tokenizer = tokenizer;
        _clsTokenId = clsTokenId;
        _sepTokenId = sepTokenId;
        _padTokenId = padTokenId;
        _maxLength = maxLength;
    }

    /// <summary>
    /// Creates a tokenizer from a tokenizer.json file.
    /// </summary>
    /// <param name="tokenizerPath">Path to tokenizer.json.</param>
    /// <param name="maxLength">Maximum sequence length.</param>
    /// <returns>Configured tokenizer wrapper.</returns>
    public static TokenizerWrapper FromFile(string tokenizerPath, int maxLength = 512)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tokenizerPath);

        if (!File.Exists(tokenizerPath))
        {
            throw new FileNotFoundException($"Tokenizer file not found: {tokenizerPath}", tokenizerPath);
        }

        try
        {
            // Load vocab.txt from the same directory if available
            var directory = Path.GetDirectoryName(tokenizerPath) ?? ".";
            var vocabPath = Path.Combine(directory, "vocab.txt");

            Tokenizer tokenizer;

            if (File.Exists(vocabPath))
            {
                // Use WordPiece tokenizer with vocab.txt
                using var vocabStream = File.OpenRead(vocabPath);
                tokenizer = WordPieceTokenizer.Create(vocabStream);
            }
            else
            {
                // Extract vocab from tokenizer.json and create WordPiece tokenizer
                tokenizer = CreateTokenizerFromJson(tokenizerPath);
            }

            // Extract special token IDs from config
            var (clsId, sepId, padId) = ExtractSpecialTokenIds(tokenizerPath);

            return new TokenizerWrapper(tokenizer, clsId, sepId, padId, maxLength);
        }
        catch (Exception ex) when (ex is not FileNotFoundException)
        {
            throw new TokenizationException($"Failed to load tokenizer from {tokenizerPath}", ex);
        }
    }

    /// <summary>
    /// Creates a WordPiece tokenizer from a HuggingFace tokenizer.json file.
    /// </summary>
    private static Tokenizer CreateTokenizerFromJson(string tokenizerPath)
    {
        var json = File.ReadAllText(tokenizerPath);
        using var doc = JsonDocument.Parse(json);

        // Extract vocab from model.vocab section
        if (!doc.RootElement.TryGetProperty("model", out var model))
        {
            throw new TokenizationException("Invalid tokenizer.json: missing 'model' section");
        }

        if (!model.TryGetProperty("vocab", out var vocab))
        {
            throw new TokenizationException("Invalid tokenizer.json: missing 'model.vocab' section");
        }

        // Build vocab dictionary and sort by ID to create vocab.txt format
        var vocabDict = new SortedDictionary<int, string>();
        foreach (var property in vocab.EnumerateObject())
        {
            var token = property.Name;
            var id = property.Value.GetInt32();
            vocabDict[id] = token;
        }

        // Create vocab.txt content (one token per line, ordered by ID)
        var vocabLines = new StringBuilder();
        for (var i = 0; i < vocabDict.Count; i++)
        {
            if (vocabDict.TryGetValue(i, out var token))
            {
                vocabLines.AppendLine(token);
            }
            else
            {
                // Fill gaps with [unused] placeholder
                vocabLines.AppendLine($"[unused{i}]");
            }
        }

        // Create tokenizer from vocab stream
        var vocabBytes = Encoding.UTF8.GetBytes(vocabLines.ToString());
        using var vocabStream = new MemoryStream(vocabBytes);
        return WordPieceTokenizer.Create(vocabStream);
    }

    /// <summary>
    /// Encodes a query-document pair for cross-encoder input.
    /// Format: [CLS] query_tokens [SEP] document_tokens [SEP] [PAD...]
    /// </summary>
    /// <param name="query">The query text.</param>
    /// <param name="document">The document text.</param>
    /// <returns>Encoded input with input_ids, attention_mask, and token_type_ids.</returns>
    public EncodedInput EncodePair(string query, string document)
    {
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(document);

        // Tokenize query and document separately
        var queryTokens = _tokenizer.EncodeToIds(query);
        var documentTokens = _tokenizer.EncodeToIds(document);

        // Calculate available space: [CLS] query [SEP] doc [SEP]
        // Special tokens: 1 CLS + 1 SEP after query + 1 SEP after doc = 3 tokens
        var availableForContent = _maxLength - 3;

        // Truncate if necessary - prioritize query, then truncate document
        var queryIds = queryTokens.ToArray();
        var docIds = documentTokens.ToArray();

        if (queryIds.Length + docIds.Length > availableForContent)
        {
            // Keep at least half for query, rest for document
            var maxQueryLen = Math.Min(queryIds.Length, availableForContent / 2);
            var maxDocLen = availableForContent - maxQueryLen;

            if (queryIds.Length > maxQueryLen)
            {
                Array.Resize(ref queryIds, maxQueryLen);
            }
            if (docIds.Length > maxDocLen)
            {
                Array.Resize(ref docIds, maxDocLen);
            }
        }

        // Build the sequence: [CLS] query [SEP] document [SEP]
        var totalLength = 1 + queryIds.Length + 1 + docIds.Length + 1;
        var paddingLength = _maxLength - totalLength;

        var inputIds = new long[_maxLength];
        var attentionMask = new long[_maxLength];
        var tokenTypeIds = new long[_maxLength];

        var pos = 0;

        // [CLS]
        inputIds[pos] = _clsTokenId;
        attentionMask[pos] = 1;
        tokenTypeIds[pos] = 0;
        pos++;

        // Query tokens
        foreach (var id in queryIds)
        {
            inputIds[pos] = id;
            attentionMask[pos] = 1;
            tokenTypeIds[pos] = 0;
            pos++;
        }

        // [SEP] after query
        inputIds[pos] = _sepTokenId;
        attentionMask[pos] = 1;
        tokenTypeIds[pos] = 0;
        pos++;

        // Document tokens
        foreach (var id in docIds)
        {
            inputIds[pos] = id;
            attentionMask[pos] = 1;
            tokenTypeIds[pos] = 1; // Token type 1 for document
            pos++;
        }

        // [SEP] after document
        inputIds[pos] = _sepTokenId;
        attentionMask[pos] = 1;
        tokenTypeIds[pos] = 1;
        pos++;

        // Padding
        for (var i = pos; i < _maxLength; i++)
        {
            inputIds[i] = _padTokenId;
            attentionMask[i] = 0;
            tokenTypeIds[i] = 0;
        }

        return new EncodedInput(inputIds, attentionMask, tokenTypeIds, totalLength);
    }

    /// <summary>
    /// Encodes multiple query-document pairs into a batch.
    /// </summary>
    /// <param name="query">The query text.</param>
    /// <param name="documents">The documents to encode.</param>
    /// <returns>Encoded batch ready for inference.</returns>
    public EncodedBatch EncodeBatch(string query, IReadOnlyList<string> documents)
    {
        var batch = new EncodedBatch(documents.Count, _maxLength);

        for (var i = 0; i < documents.Count; i++)
        {
            var encoded = EncodePair(query, documents[i]);
            batch.SetInput(i, encoded);
        }

        return batch;
    }

    private static (int clsId, int sepId, int padId) ExtractSpecialTokenIds(string tokenizerPath)
    {
        try
        {
            var json = File.ReadAllText(tokenizerPath);
            using var doc = JsonDocument.Parse(json);

            var clsId = 101;  // Default BERT [CLS]
            var sepId = 102;  // Default BERT [SEP]
            var padId = 0;    // Default BERT [PAD]

            // Try to extract from added_tokens or model config
            if (doc.RootElement.TryGetProperty("added_tokens", out var addedTokens))
            {
                foreach (var token in addedTokens.EnumerateArray())
                {
                    if (token.TryGetProperty("content", out var content) &&
                        token.TryGetProperty("id", out var id))
                    {
                        var contentStr = content.GetString();
                        var tokenId = id.GetInt32();

                        switch (contentStr)
                        {
                            case "[CLS]":
                                clsId = tokenId;
                                break;
                            case "[SEP]":
                                sepId = tokenId;
                                break;
                            case "[PAD]":
                                padId = tokenId;
                                break;
                        }
                    }
                }
            }

            return (clsId, sepId, padId);
        }
        catch
        {
            // Return BERT defaults if parsing fails
            return (101, 102, 0);
        }
    }

    public void Dispose()
    {
        // Tokenizer doesn't implement IDisposable in current version
        // Kept for future compatibility
    }
}
