namespace LocalAI.Generator;

/// <summary>
/// Well-known model identifiers for LocalAI components.
/// </summary>
public static class WellKnownModels
{
    /// <summary>
    /// Text generation models.
    /// </summary>
    public static class Generator
    {
        /// <summary>
        /// Default balanced model - Microsoft Phi-3.5 Mini (MIT license).
        /// 3.8B parameters, 128K context, excellent reasoning.
        /// </summary>
        public const string Default = "microsoft/Phi-3.5-mini-instruct-onnx";

        /// <summary>
        /// Fast/small model - Meta Llama 3.2 1B.
        /// 1B parameters, fast inference, good for simple tasks.
        /// Note: Llama Community License (700M MAU limit).
        /// </summary>
        public const string Fast = "onnx-community/Llama-3.2-1B-Instruct-ONNX";

        /// <summary>
        /// Small model - same as Fast, alias for clarity.
        /// </summary>
        public const string Small = Fast;

        /// <summary>
        /// Quality model - Microsoft Phi-4 (MIT license).
        /// 14B parameters, highest quality reasoning.
        /// </summary>
        public const string Quality = "microsoft/phi-4-onnx";

        /// <summary>
        /// Medium model - Meta Llama 3.2 3B.
        /// 3B parameters, balance of speed and quality.
        /// Note: Llama Community License (700M MAU limit).
        /// </summary>
        public const string Medium = "onnx-community/Llama-3.2-3B-Instruct-ONNX";

        /// <summary>
        /// Multilingual model - Google Gemma 2 2B.
        /// 2B parameters, good multilingual support.
        /// Note: Gemma Terms of Use apply.
        /// </summary>
        public const string Multilingual = "google/gemma-2-2b-it-onnx";
    }

    /// <summary>
    /// Embedding models (from LocalAI.Embedder).
    /// </summary>
    public static class Embedder
    {
        /// <summary>
        /// Default embedding model - bge-small-en-v1.5.
        /// </summary>
        public const string Default = "BAAI/bge-small-en-v1.5";

        /// <summary>
        /// Multilingual embedding model - bge-m3.
        /// </summary>
        public const string Multilingual = "BAAI/bge-m3";

        /// <summary>
        /// Large English embedding model - bge-large-en-v1.5.
        /// </summary>
        public const string Large = "BAAI/bge-large-en-v1.5";
    }

    /// <summary>
    /// Reranking models (from LocalAI.Reranker).
    /// </summary>
    public static class Reranker
    {
        /// <summary>
        /// Default reranker - bge-reranker-base.
        /// </summary>
        public const string Default = "BAAI/bge-reranker-base";

        /// <summary>
        /// Large reranker - bge-reranker-large.
        /// </summary>
        public const string Large = "BAAI/bge-reranker-large";

        /// <summary>
        /// Multilingual reranker - bge-reranker-v2-m3.
        /// </summary>
        public const string Multilingual = "BAAI/bge-reranker-v2-m3";
    }

    /// <summary>
    /// Gets license information for a model.
    /// </summary>
    /// <param name="modelId">The model identifier.</param>
    /// <returns>License tier classification.</returns>
    public static LicenseTier GetLicenseTier(string modelId)
    {
        var info = ModelRegistry.GetModel(modelId);
        return info?.License ?? LicenseTier.Conditional;
    }

    /// <summary>
    /// Checks if a model has usage restrictions.
    /// </summary>
    /// <param name="modelId">The model identifier.</param>
    /// <returns>True if the model has restrictions (non-MIT license).</returns>
    public static bool HasRestrictions(string modelId)
    {
        return GetLicenseTier(modelId) != LicenseTier.MIT;
    }

    /// <summary>
    /// Gets MIT-licensed models only (no usage restrictions).
    /// </summary>
    public static IReadOnlyList<string> GetUnrestrictedModels() =>
    [
        Generator.Default,
        Generator.Quality
    ];
}
