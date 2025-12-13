using Microsoft.ML.OnnxRuntime;

namespace LocalAI.Inference;

/// <summary>
/// Factory for creating ONNX Runtime inference sessions with proper execution provider configuration.
/// </summary>
public static class OnnxSessionFactory
{
    /// <summary>
    /// Creates an ONNX Runtime inference session with the specified execution provider.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="provider">The execution provider to use.</param>
    /// <param name="configureOptions">Optional callback to configure additional session options.</param>
    /// <returns>A configured inference session.</returns>
    public static InferenceSession Create(
        string modelPath,
        ExecutionProvider provider = ExecutionProvider.Auto,
        Action<SessionOptions>? configureOptions = null)
    {
        var options = new SessionOptions();

        // Apply user configuration first
        configureOptions?.Invoke(options);

        // Configure execution provider
        ConfigureExecutionProvider(options, provider);

        return new InferenceSession(modelPath, options);
    }

    /// <summary>
    /// Configures the execution provider for the session options.
    /// </summary>
    public static void ConfigureExecutionProvider(SessionOptions options, ExecutionProvider provider)
    {
        switch (provider)
        {
            case ExecutionProvider.Auto:
                TryAddBestAvailableProvider(options);
                break;

            case ExecutionProvider.Cuda:
                TryAddCuda(options);
                break;

            case ExecutionProvider.DirectML:
                TryAddDirectML(options);
                break;

            case ExecutionProvider.CoreML:
                TryAddCoreML(options);
                break;

            case ExecutionProvider.Cpu:
                // CPU is always available as fallback
                break;

            default:
                throw new ArgumentOutOfRangeException(nameof(provider), provider, "Unknown execution provider");
        }
    }

    /// <summary>
    /// Tries to add the best available GPU provider, falls back to CPU.
    /// </summary>
    private static void TryAddBestAvailableProvider(SessionOptions options)
    {
        // Try providers in order of preference
        if (TryAddCuda(options)) return;
        if (TryAddDirectML(options)) return;
        if (TryAddCoreML(options)) return;
        // CPU fallback is automatic
    }

    private static bool TryAddCuda(SessionOptions options)
    {
        try
        {
            options.AppendExecutionProvider_CUDA();
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static bool TryAddDirectML(SessionOptions options)
    {
        try
        {
            options.AppendExecutionProvider_DML();
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static bool TryAddCoreML(SessionOptions options)
    {
        try
        {
            options.AppendExecutionProvider_CoreML();
            return true;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Gets the list of available execution providers on the current system.
    /// </summary>
    public static IEnumerable<ExecutionProvider> GetAvailableProviders()
    {
        // CPU is always available
        yield return ExecutionProvider.Cpu;

        // Check GPU providers
        var testOptions = new SessionOptions();

        if (TryAddCuda(testOptions))
            yield return ExecutionProvider.Cuda;

        testOptions = new SessionOptions();
        if (TryAddDirectML(testOptions))
            yield return ExecutionProvider.DirectML;

        testOptions = new SessionOptions();
        if (TryAddCoreML(testOptions))
            yield return ExecutionProvider.CoreML;
    }
}
