namespace LMSupply.Text;

/// <summary>
/// SentencePiece/Unigram tokenizer for pair encoding (cross-encoders/rerankers).
/// Supports models that use Unigram or BPE tokenization instead of WordPiece.
/// </summary>
internal sealed class SentencePiecePairTokenizer : IPairTokenizer
{
    private readonly Tokenizer _tokenizer;
    private readonly SpecialTokens _specialTokens;
    private readonly int _maxSequenceLength;

    public int VocabSize { get; }
    public int PadTokenId => _specialTokens.PadTokenId;
    public int UnkTokenId => _specialTokens.UnkTokenId;
    public int? BosTokenId => _specialTokens.BosTokenId;
    public int? EosTokenId => _specialTokens.EosTokenId;
    public int? ClsTokenId => _specialTokens.ClsTokenId;
    public int? SepTokenId => _specialTokens.SepTokenId;
    public int MaxSequenceLength => _maxSequenceLength;

    public SentencePiecePairTokenizer(
        Tokenizer tokenizer,
        SpecialTokens specialTokens,
        int maxSequenceLength,
        int vocabSize = 32000)
    {
        _tokenizer = tokenizer;
        _specialTokens = specialTokens;
        _maxSequenceLength = maxSequenceLength;
        VocabSize = vocabSize;
    }

    public int[] Encode(string text, bool addSpecialTokens = true)
    {
        var ids = _tokenizer.EncodeToIds(text).ToArray();

        if (!addSpecialTokens)
            return ids;

        // Use CLS/SEP if available (BERT-style), otherwise BOS/EOS
        var startToken = _specialTokens.ClsTokenId ?? _specialTokens.BosTokenId;
        var endToken = _specialTokens.SepTokenId ?? _specialTokens.EosTokenId;

        var hasStart = startToken.HasValue;
        var hasEnd = endToken.HasValue;
        var extraTokens = (hasStart ? 1 : 0) + (hasEnd ? 1 : 0);

        if (extraTokens == 0)
            return ids;

        var result = new int[ids.Length + extraTokens];
        var pos = 0;

        if (hasStart)
        {
            result[pos++] = startToken!.Value;
        }

        Array.Copy(ids, 0, result, pos, ids.Length);
        pos += ids.Length;

        if (hasEnd)
        {
            result[pos] = endToken!.Value;
        }

        return result;
    }

    public string Decode(ReadOnlySpan<int> tokenIds, bool skipSpecialTokens = true)
    {
        var ids = skipSpecialTokens
            ? tokenIds.ToArray().Where(id => !IsSpecialToken(id))
            : tokenIds.ToArray().AsEnumerable();

        var decoded = _tokenizer.Decode(ids);

        // SentencePiece uses ▁ (U+2581) to mark word boundaries, replace with space
        return decoded?.Replace("▁", " ").Trim() ?? string.Empty;
    }

    public bool IsSpecialToken(int tokenId)
    {
        return tokenId == PadTokenId ||
               tokenId == UnkTokenId ||
               tokenId == ClsTokenId ||
               tokenId == SepTokenId ||
               tokenId == BosTokenId ||
               tokenId == EosTokenId;
    }

    public EncodedSequence EncodeSequence(string text, int? maxLength = null)
    {
        var length = maxLength ?? _maxSequenceLength;
        var tokens = _tokenizer.EncodeToIds(text).ToArray();

        var startToken = _specialTokens.ClsTokenId ?? _specialTokens.BosTokenId ?? 0;
        var endToken = _specialTokens.SepTokenId ?? _specialTokens.EosTokenId ?? 2;

        var availableLength = length - 2;
        var contentLength = Math.Min(tokens.Length, availableLength);

        var inputIds = new long[length];
        var attentionMask = new long[length];

        inputIds[0] = startToken;
        attentionMask[0] = 1;

        for (int i = 0; i < contentLength; i++)
        {
            inputIds[i + 1] = tokens[i];
            attentionMask[i + 1] = 1;
        }

        inputIds[contentLength + 1] = endToken;
        attentionMask[contentLength + 1] = 1;

        for (int i = contentLength + 2; i < length; i++)
        {
            inputIds[i] = PadTokenId;
        }

        return new EncodedSequence(inputIds, attentionMask, contentLength + 2);
    }

    public EncodedBatch EncodeBatch(IReadOnlyList<string> texts, int? maxLength = null)
    {
        var length = maxLength ?? _maxSequenceLength;
        var batch = new EncodedBatch(texts.Count, length);

        for (int i = 0; i < texts.Count; i++)
        {
            var encoded = EncodeSequence(texts[i], length);
            batch.SetSequence(i, encoded);
        }

        return batch;
    }

    public EncodedPair EncodePair(string text1, string text2, int? maxLength = null)
    {
        var length = maxLength ?? _maxSequenceLength;

        var tokens1 = _tokenizer.EncodeToIds(text1).ToArray();
        var tokens2 = _tokenizer.EncodeToIds(text2).ToArray();

        var startToken = _specialTokens.ClsTokenId ?? _specialTokens.BosTokenId ?? 0;
        var sepToken = _specialTokens.SepTokenId ?? _specialTokens.EosTokenId ?? 2;

        // Format: [CLS/BOS] text1 [SEP/EOS] text2 [SEP/EOS]
        var availableLength = length - 3; // Reserve 3 for special tokens
        var totalTokens = tokens1.Length + tokens2.Length;

        int len1, len2;
        if (totalTokens <= availableLength)
        {
            len1 = tokens1.Length;
            len2 = tokens2.Length;
        }
        else
        {
            // Truncate proportionally, but ensure at least some tokens from each
            var ratio = (double)availableLength / totalTokens;
            len1 = Math.Max(1, (int)(tokens1.Length * ratio));
            len2 = Math.Max(1, Math.Min(tokens2.Length, availableLength - len1));
            len1 = Math.Min(tokens1.Length, availableLength - len2);
        }

        var inputIds = new long[length];
        var attentionMask = new long[length];
        var tokenTypeIds = new long[length];

        var pos = 0;

        // [CLS/BOS]
        inputIds[pos] = startToken;
        attentionMask[pos] = 1;
        tokenTypeIds[pos] = 0;
        pos++;

        // text1 tokens
        for (int i = 0; i < len1; i++)
        {
            inputIds[pos] = tokens1[i];
            attentionMask[pos] = 1;
            tokenTypeIds[pos] = 0;
            pos++;
        }

        // [SEP/EOS]
        inputIds[pos] = sepToken;
        attentionMask[pos] = 1;
        tokenTypeIds[pos] = 0;
        pos++;

        // text2 tokens
        for (int i = 0; i < len2; i++)
        {
            inputIds[pos] = tokens2[i];
            attentionMask[pos] = 1;
            tokenTypeIds[pos] = 1;
            pos++;
        }

        // [SEP/EOS]
        inputIds[pos] = sepToken;
        attentionMask[pos] = 1;
        tokenTypeIds[pos] = 1;
        pos++;

        // Padding
        for (int i = pos; i < length; i++)
        {
            inputIds[i] = PadTokenId;
            // attentionMask and tokenTypeIds already 0
        }

        return new EncodedPair(inputIds, attentionMask, tokenTypeIds, pos);
    }

    public EncodedPairBatch EncodePairBatch(string text1, IReadOnlyList<string> texts2, int? maxLength = null)
    {
        var length = maxLength ?? _maxSequenceLength;
        var batch = new EncodedPairBatch(texts2.Count, length);

        for (int i = 0; i < texts2.Count; i++)
        {
            var encoded = EncodePair(text1, texts2[i], length);
            batch.SetPair(i, encoded);
        }

        return batch;
    }

    public void Dispose()
    {
        // Tokenizer doesn't implement IDisposable
    }
}
