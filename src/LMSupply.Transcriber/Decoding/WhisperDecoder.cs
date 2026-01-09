using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LMSupply.Transcriber.Decoding;

/// <summary>
/// Whisper autoregressive decoder using greedy search.
/// </summary>
internal sealed class WhisperDecoder
{
    private readonly InferenceSession _decoderSession;
    private readonly WhisperTokenizer _tokenizer;
    private readonly int _maxLength;

    // Input/output names for onnx-community models
    private const string InputTokensName = "input_ids";
    private const string InputEncoderHiddenStates = "encoder_hidden_states";
    private const string OutputLogitsName = "logits";

    // Alternative names used by some ONNX exports
    private static readonly string[] TokenInputNames = ["input_ids", "tokens", "decoder_input_ids"];
    private static readonly string[] EncoderInputNames = ["encoder_hidden_states", "audio", "encoder_outputs"];

    private readonly string _actualTokenInputName;
    private readonly string _actualEncoderInputName;
    private readonly string _actualLogitsOutputName;

    public WhisperDecoder(
        InferenceSession decoderSession,
        WhisperTokenizer tokenizer,
        int maxLength = 448)
    {
        _decoderSession = decoderSession;
        _tokenizer = tokenizer;
        _maxLength = maxLength;

        // Detect input/output names from session metadata
        var inputNames = _decoderSession.InputMetadata.Keys.ToHashSet(StringComparer.OrdinalIgnoreCase);
        var outputNames = _decoderSession.OutputMetadata.Keys.ToHashSet(StringComparer.OrdinalIgnoreCase);

        _actualTokenInputName = TokenInputNames.FirstOrDefault(n => inputNames.Contains(n))
            ?? throw new InvalidOperationException(
                $"Could not find token input. Available inputs: {string.Join(", ", inputNames)}");

        _actualEncoderInputName = EncoderInputNames.FirstOrDefault(n => inputNames.Contains(n))
            ?? throw new InvalidOperationException(
                $"Could not find encoder input. Available inputs: {string.Join(", ", inputNames)}");

        _actualLogitsOutputName = outputNames.Contains(OutputLogitsName) ? OutputLogitsName
            : outputNames.FirstOrDefault(n => n.Contains("logit", StringComparison.OrdinalIgnoreCase))
            ?? outputNames.First();
    }

    /// <summary>
    /// Decodes encoder output to text using greedy search.
    /// </summary>
    public async Task<DecodingResult> DecodeAsync(
        float[] encoderOutput,
        int encoderSequenceLength,
        int hiddenSize,
        TranscribeOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        return await Task.Run(() =>
        {
            // Initialize tokens with SOT sequence
            var useTimestamps = options?.WordTimestamps ?? false;
            var initialTokens = _tokenizer.GetSotSequence(options?.Language, useTimestamps);
            var tokens = new List<int>(initialTokens);

            var segments = new List<TranscriptionSegment>();
            var currentSegmentTokens = new List<int>();
            var currentSegmentStart = 0.0;

            // Create encoder output tensor [1, seq_len, hidden_size]
            var encoderTensor = new DenseTensor<float>(
                encoderOutput,
                [1, encoderSequenceLength, hiddenSize]);

            string? detectedLanguage = null;
            int segmentId = 0;

            // Autoregressive generation loop
            while (tokens.Count < _maxLength)
            {
                cancellationToken.ThrowIfCancellationRequested();

                // Create input tensor
                var tokenArray = tokens.ToArray();
                var tokenTensor = new DenseTensor<long>(
                    tokenArray.Select(t => (long)t).ToArray(),
                    [1, tokens.Count]);

                // Run decoder
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(_actualTokenInputName, tokenTensor),
                    NamedOnnxValue.CreateFromTensor(_actualEncoderInputName, encoderTensor)
                };

                using var results = _decoderSession.Run(inputs);
                var logits = results.First(r => r.Name == _actualLogitsOutputName).AsTensor<float>();

                // Get logits for last position
                var vocabSize = _tokenizer.VocabSize;
                var lastPosition = tokens.Count - 1;
                var lastLogits = new float[vocabSize];

                for (int i = 0; i < vocabSize; i++)
                {
                    lastLogits[i] = logits[0, lastPosition, i];
                }

                // Apply temperature if specified
                if (options is { Temperature: > 0 and < 1 })
                {
                    for (int i = 0; i < lastLogits.Length; i++)
                    {
                        lastLogits[i] /= options.Temperature;
                    }
                }

                // Greedy selection: argmax
                var nextToken = ArgMax(lastLogits);

                // Detect language from first generated token after SOT
                if (tokens.Count == initialTokens.Length && _tokenizer.IsLanguageToken(nextToken))
                {
                    detectedLanguage = _tokenizer.GetLanguageFromToken(nextToken);
                }

                // Check for end of text
                if (nextToken == WhisperTokenizer.EndOfTextToken)
                {
                    break;
                }

                // Handle timestamp tokens
                if (_tokenizer.IsTimestampToken(nextToken))
                {
                    var timestamp = _tokenizer.TimestampTokenToSeconds(nextToken);

                    // Start timestamp
                    if (currentSegmentTokens.Count == 0)
                    {
                        currentSegmentStart = timestamp;
                    }
                    else
                    {
                        // End timestamp - create segment
                        var segmentText = _tokenizer.Decode(
                            currentSegmentTokens.ToArray().AsSpan(),
                            skipSpecialTokens: true);

                        if (!string.IsNullOrWhiteSpace(segmentText))
                        {
                            segments.Add(new TranscriptionSegment
                            {
                                Id = segmentId++,
                                Start = currentSegmentStart,
                                End = timestamp,
                                Text = segmentText.Trim()
                            });
                        }

                        currentSegmentTokens.Clear();
                    }
                }
                else if (!_tokenizer.IsSpecialToken(nextToken))
                {
                    // Regular text token
                    currentSegmentTokens.Add(nextToken);
                }

                tokens.Add(nextToken);
            }

            // Handle remaining tokens as final segment
            if (currentSegmentTokens.Count > 0)
            {
                var segmentText = _tokenizer.Decode(
                    currentSegmentTokens.ToArray().AsSpan(),
                    skipSpecialTokens: true);

                if (!string.IsNullOrWhiteSpace(segmentText))
                {
                    segments.Add(new TranscriptionSegment
                    {
                        Id = segmentId,
                        Start = currentSegmentStart,
                        End = 30.0, // Default chunk length
                        Text = segmentText.Trim()
                    });
                }
            }

            // If no segments created (no timestamps mode), create single segment
            if (segments.Count == 0)
            {
                var allTokens = tokens.Skip(initialTokens.Length).ToArray();
                var fullText = _tokenizer.Decode(allTokens.AsSpan(), skipSpecialTokens: true);

                if (!string.IsNullOrWhiteSpace(fullText))
                {
                    segments.Add(new TranscriptionSegment
                    {
                        Id = 0,
                        Start = 0,
                        End = 30.0,
                        Text = fullText.Trim()
                    });
                }
            }

            // Combine all segment texts for full transcription
            var fullTranscription = string.Join(" ", segments.Select(s => s.Text));

            return new DecodingResult
            {
                Text = fullTranscription,
                Language = detectedLanguage ?? options?.Language ?? "en",
                Segments = segments,
                TokenCount = tokens.Count - initialTokens.Length
            };
        }, cancellationToken);
    }

    private static int ArgMax(float[] values)
    {
        int maxIndex = 0;
        float maxValue = values[0];

        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > maxValue)
            {
                maxValue = values[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }
}

/// <summary>
/// Result from the Whisper decoder.
/// </summary>
internal sealed class DecodingResult
{
    public required string Text { get; init; }
    public required string Language { get; init; }
    public required List<TranscriptionSegment> Segments { get; init; }
    public int TokenCount { get; init; }
}
