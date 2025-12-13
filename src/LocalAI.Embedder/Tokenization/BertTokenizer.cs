namespace LocalAI.Embedder.Tokenization;

/// <summary>
/// BERT-compatible WordPiece tokenizer.
/// </summary>
internal sealed class BertTokenizer : ITokenizer
{
    private readonly Dictionary<string, int> _vocab;
    private readonly int _clsTokenId;
    private readonly int _sepTokenId;
    private readonly int _padTokenId;
    private readonly int _unkTokenId;
    private readonly bool _doLowerCase;
    private readonly int _maxInputCharsPerWord;

    private BertTokenizer(
        Dictionary<string, int> vocab,
        bool doLowerCase,
        int clsTokenId,
        int sepTokenId,
        int padTokenId,
        int unkTokenId)
    {
        _vocab = vocab;
        _doLowerCase = doLowerCase;
        _clsTokenId = clsTokenId;
        _sepTokenId = sepTokenId;
        _padTokenId = padTokenId;
        _unkTokenId = unkTokenId;
        _maxInputCharsPerWord = 200;
    }

    /// <summary>
    /// Creates a tokenizer from a vocab.txt file.
    /// </summary>
    public static async Task<BertTokenizer> CreateFromVocabAsync(string vocabPath, bool doLowerCase = true)
    {
        if (!File.Exists(vocabPath))
            throw new TokenizationException($"Vocabulary file not found: {vocabPath}");

        var vocab = await LoadVocabularyAsync(vocabPath);

        int clsId = vocab.GetValueOrDefault("[CLS]", 101);
        int sepId = vocab.GetValueOrDefault("[SEP]", 102);
        int padId = vocab.GetValueOrDefault("[PAD]", 0);
        int unkId = vocab.GetValueOrDefault("[UNK]", 100);

        return new BertTokenizer(vocab, doLowerCase, clsId, sepId, padId, unkId);
    }

    public (long[] InputIds, long[] AttentionMask) Encode(string text, int maxLength)
    {
        // Preprocess text
        var processedText = PreprocessText(text);

        // Tokenize to WordPiece tokens
        var tokens = TokenizeToWordPiece(processedText);

        // Convert tokens to IDs
        var tokenIds = tokens.Select(t => _vocab.GetValueOrDefault(t, _unkTokenId)).ToList();

        // Calculate available space for content tokens (excluding [CLS] and [SEP])
        int availableLength = maxLength - 2;
        int contentLength = Math.Min(tokenIds.Count, availableLength);

        // Build final sequence: [CLS] + tokens + [SEP] + [PAD]...
        var inputIds = new long[maxLength];
        var attentionMask = new long[maxLength];

        // [CLS] token
        inputIds[0] = _clsTokenId;
        attentionMask[0] = 1;

        // Content tokens
        for (int i = 0; i < contentLength; i++)
        {
            inputIds[i + 1] = tokenIds[i];
            attentionMask[i + 1] = 1;
        }

        // [SEP] token
        inputIds[contentLength + 1] = _sepTokenId;
        attentionMask[contentLength + 1] = 1;

        // Padding (already 0 from initialization)
        for (int i = contentLength + 2; i < maxLength; i++)
        {
            inputIds[i] = _padTokenId;
            attentionMask[i] = 0;
        }

        return (inputIds, attentionMask);
    }

    public (long[][] InputIds, long[][] AttentionMasks) EncodeBatch(IReadOnlyList<string> texts, int maxLength)
    {
        var inputIds = new long[texts.Count][];
        var attentionMasks = new long[texts.Count][];

        for (int i = 0; i < texts.Count; i++)
        {
            (inputIds[i], attentionMasks[i]) = Encode(texts[i], maxLength);
        }

        return (inputIds, attentionMasks);
    }

    private string PreprocessText(string text)
    {
        // Clean text
        text = CleanText(text);

        // Lowercase if required
        if (_doLowerCase)
        {
            text = text.ToLowerInvariant();
        }

        return text;
    }

    private static string CleanText(string text)
    {
        var result = new char[text.Length];
        int resultIndex = 0;

        foreach (char c in text)
        {
            // Skip invalid characters
            if (c == 0 || c == 0xFFFD || char.IsControl(c))
            {
                continue;
            }

            // Normalize whitespace
            if (char.IsWhiteSpace(c))
            {
                result[resultIndex++] = ' ';
            }
            else
            {
                result[resultIndex++] = c;
            }
        }

        return new string(result, 0, resultIndex);
    }

    private List<string> TokenizeToWordPiece(string text)
    {
        var outputTokens = new List<string>();
        var words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);

        foreach (var word in words)
        {
            // Handle punctuation
            var subWords = SplitOnPunctuation(word);

            foreach (var subWord in subWords)
            {
                if (subWord.Length > _maxInputCharsPerWord)
                {
                    outputTokens.Add("[UNK]");
                    continue;
                }

                // WordPiece tokenization: greedy longest-match-first
                var wordPieceTokens = WordPieceTokenize(subWord);
                outputTokens.AddRange(wordPieceTokens);
            }
        }

        return outputTokens;
    }

    private List<string> WordPieceTokenize(string word)
    {
        var tokens = new List<string>();
        int start = 0;

        while (start < word.Length)
        {
            int end = word.Length;
            string? currentToken = null;

            // Greedy longest-match-first
            while (start < end)
            {
                var substr = word[start..end];
                if (start > 0)
                {
                    substr = "##" + substr;
                }

                if (_vocab.ContainsKey(substr))
                {
                    currentToken = substr;
                    break;
                }

                end--;
            }

            if (currentToken == null)
            {
                // No match found, use [UNK]
                tokens.Add("[UNK]");
                break;
            }

            tokens.Add(currentToken);
            start = end;
        }

        return tokens;
    }

    private static List<string> SplitOnPunctuation(string text)
    {
        var result = new List<string>();
        var current = new List<char>();

        foreach (char c in text)
        {
            if (char.IsPunctuation(c))
            {
                if (current.Count > 0)
                {
                    result.Add(new string(current.ToArray()));
                    current.Clear();
                }
                result.Add(c.ToString());
            }
            else
            {
                current.Add(c);
            }
        }

        if (current.Count > 0)
        {
            result.Add(new string(current.ToArray()));
        }

        return result;
    }

    private static async Task<Dictionary<string, int>> LoadVocabularyAsync(string vocabPath)
    {
        var vocab = new Dictionary<string, int>();
        var lines = await File.ReadAllLinesAsync(vocabPath);

        for (int i = 0; i < lines.Length; i++)
        {
            var token = lines[i].Trim();
            if (!string.IsNullOrEmpty(token))
            {
                vocab[token] = i;
            }
        }

        return vocab;
    }
}
