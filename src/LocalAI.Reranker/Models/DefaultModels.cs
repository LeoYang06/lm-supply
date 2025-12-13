namespace LocalAI.Reranker.Models;

/// <summary>
/// Provides definitions for built-in supported models.
/// </summary>
public static class DefaultModels
{
    /// <summary>
    /// Gets the default model (balanced speed and quality).
    /// </summary>
    public static ModelInfo Default => MsMarcoMiniLML6V2;

    /// <summary>
    /// MS MARCO MiniLM L6 v2 - Fast and lightweight model.
    /// ~90MB, 512 tokens, good quality for English.
    /// </summary>
    public static ModelInfo MsMarcoMiniLML6V2 { get; } = new()
    {
        Id = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        Alias = "default",
        DisplayName = "MS MARCO MiniLM L6",
        Parameters = 22_700_000,
        MaxSequenceLength = 512,
        SizeBytes = 90_000_000,
        OnnxFile = "onnx/model.onnx",
        TokenizerFile = "tokenizer.json",
        Description = "Balanced speed and quality for English text",
        IsMultilingual = false,
        Architecture = ModelArchitecture.Bert,
        OutputShape = OutputShape.SingleLogit
    };

    /// <summary>
    /// MS MARCO MiniLM L12 v2 - Higher quality model.
    /// ~134MB, 512 tokens, better accuracy.
    /// </summary>
    public static ModelInfo MsMarcoMiniLML12V2 { get; } = new()
    {
        Id = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        Alias = "quality",
        DisplayName = "MS MARCO MiniLM L12",
        Parameters = 33_400_000,
        MaxSequenceLength = 512,
        SizeBytes = 134_000_000,
        OnnxFile = "onnx/model.onnx",
        TokenizerFile = "tokenizer.json",
        Description = "Higher quality at the cost of speed",
        IsMultilingual = false,
        Architecture = ModelArchitecture.Bert,
        OutputShape = OutputShape.SingleLogit
    };

    /// <summary>
    /// MS MARCO TinyBERT L2 v2 - Ultra-fast lightweight model.
    /// ~18MB, 512 tokens, basic quality.
    /// </summary>
    public static ModelInfo MsMarcoTinyBertL2V2 { get; } = new()
    {
        Id = "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        Alias = "fast",
        DisplayName = "MS MARCO TinyBERT L2",
        Parameters = 4_400_000,
        MaxSequenceLength = 512,
        SizeBytes = 18_000_000,
        OnnxFile = "onnx/model.onnx",
        TokenizerFile = "tokenizer.json",
        Description = "Ultra-fast for latency-critical applications",
        IsMultilingual = false,
        Architecture = ModelArchitecture.Bert,
        OutputShape = OutputShape.SingleLogit
    };

    /// <summary>
    /// BGE Reranker v2 M3 - Best quality multilingual model.
    /// ~1.1GB, 8192 tokens, 100+ languages.
    /// </summary>
    public static ModelInfo BgeRerankerV2M3 { get; } = new()
    {
        Id = "BAAI/bge-reranker-v2-m3",
        Alias = "multilingual",
        DisplayName = "BGE Reranker v2 M3",
        Parameters = 568_000_000,
        MaxSequenceLength = 8192,
        SizeBytes = 1_100_000_000,
        OnnxFile = "onnx/model.onnx",
        TokenizerFile = "tokenizer.json",
        Description = "Best quality, multilingual support, long context",
        IsMultilingual = true,
        Architecture = ModelArchitecture.XlmRoberta,
        OutputShape = OutputShape.SingleLogit
    };

    /// <summary>
    /// BGE Reranker Base - Good multilingual model.
    /// ~440MB, 512 tokens, multilingual.
    /// </summary>
    public static ModelInfo BgeRerankerBase { get; } = new()
    {
        Id = "BAAI/bge-reranker-base",
        Alias = "bge-base",
        DisplayName = "BGE Reranker Base",
        Parameters = 278_000_000,
        MaxSequenceLength = 512,
        SizeBytes = 440_000_000,
        OnnxFile = "onnx/model.onnx",
        TokenizerFile = "tokenizer.json",
        Description = "Good quality multilingual model",
        IsMultilingual = true,
        Architecture = ModelArchitecture.XlmRoberta,
        OutputShape = OutputShape.SingleLogit
    };

    /// <summary>
    /// Gets all built-in models.
    /// </summary>
    public static IReadOnlyList<ModelInfo> All { get; } =
    [
        MsMarcoMiniLML6V2,
        MsMarcoMiniLML12V2,
        MsMarcoTinyBertL2V2,
        BgeRerankerV2M3,
        BgeRerankerBase
    ];
}
