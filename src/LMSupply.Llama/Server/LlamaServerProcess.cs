using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using ChildProcessGuard;

namespace LMSupply.Llama.Server;

/// <summary>
/// Configuration for llama-server process.
/// </summary>
public sealed class LlamaServerConfig
{
    /// <summary>
    /// Path to the GGUF model file.
    /// </summary>
    public required string ModelPath { get; init; }

    /// <summary>
    /// Port to run the server on (0 for auto-assign).
    /// </summary>
    public int Port { get; init; } = 0;

    /// <summary>
    /// Context size.
    /// </summary>
    public int ContextSize { get; init; } = 4096;

    /// <summary>
    /// Number of GPU layers to offload (-1 for all).
    /// </summary>
    public int GpuLayers { get; init; } = -1;

    /// <summary>
    /// Batch size for prompt processing.
    /// </summary>
    public int BatchSize { get; init; } = 512;

    /// <summary>
    /// Number of parallel sequences.
    /// </summary>
    public int Parallel { get; init; } = 1;

    /// <summary>
    /// Enable Flash Attention.
    /// </summary>
    public bool FlashAttention { get; init; } = false;

    /// <summary>
    /// Additional command line arguments.
    /// </summary>
    public IReadOnlyList<string>? AdditionalArgs { get; init; }

    /// <summary>
    /// Timeout for server startup.
    /// </summary>
    public TimeSpan StartupTimeout { get; init; } = TimeSpan.FromSeconds(60);

    /// <summary>
    /// Timeout for graceful shutdown.
    /// </summary>
    public TimeSpan ShutdownTimeout { get; init; } = TimeSpan.FromSeconds(10);
}

/// <summary>
/// Information about a running llama-server process.
/// </summary>
public sealed record LlamaServerInfo
{
    /// <summary>
    /// Process ID.
    /// </summary>
    public required int ProcessId { get; init; }

    /// <summary>
    /// Port the server is listening on.
    /// </summary>
    public required int Port { get; init; }

    /// <summary>
    /// Base URL for API calls.
    /// </summary>
    public string BaseUrl => $"http://localhost:{Port}";

    /// <summary>
    /// Model path being served.
    /// </summary>
    public required string ModelPath { get; init; }

    /// <summary>
    /// llama-server version.
    /// </summary>
    public string? Version { get; init; }

    /// <summary>
    /// GPU backend being used.
    /// </summary>
    public LlamaServerBackend Backend { get; init; }

    /// <summary>
    /// Time when server was started.
    /// </summary>
    public DateTimeOffset StartTime { get; init; }

    /// <summary>
    /// Startup log from the server (stderr output during initialization).
    /// </summary>
    public string? StartupLog { get; init; }
}

/// <summary>
/// Manages llama-server process lifecycle with automatic cleanup.
/// </summary>
public sealed class LlamaServerProcess : IAsyncDisposable
{
    private readonly ProcessGuardian _guardian;
    private readonly LlamaServerConfig _config;
    private readonly string _serverPath;
    private readonly LlamaServerBackend _backend;
    private readonly HttpClient _httpClient;

    private Process? _process;
    private int _port;
    private bool _disposed;

    /// <summary>
    /// Gets information about the running server.
    /// </summary>
    public LlamaServerInfo? Info { get; private set; }

    /// <summary>
    /// Gets whether the server is running.
    /// </summary>
    public bool IsRunning => _process is { HasExited: false };

    private LlamaServerProcess(
        string serverPath,
        LlamaServerConfig config,
        LlamaServerBackend backend)
    {
        _serverPath = serverPath;
        _config = config;
        _backend = backend;

        _guardian = new ProcessGuardian(new ProcessGuardianOptions
        {
            ProcessKillTimeout = config.ShutdownTimeout
        });

        _httpClient = new HttpClient
        {
            Timeout = TimeSpan.FromSeconds(5)
        };
    }

    /// <summary>
    /// Starts a new llama-server process.
    /// </summary>
    public static async Task<LlamaServerProcess> StartAsync(
        string serverPath,
        LlamaServerConfig config,
        LlamaServerBackend backend,
        CancellationToken cancellationToken = default)
    {
        var server = new LlamaServerProcess(serverPath, config, backend);

        try
        {
            await server.StartInternalAsync(cancellationToken);
            return server;
        }
        catch
        {
            await server.DisposeAsync();
            throw;
        }
    }

    private async Task StartInternalAsync(CancellationToken cancellationToken)
    {
        // Find available port if not specified
        _port = _config.Port > 0 ? _config.Port : FindAvailablePort();

        // Build arguments
        var args = BuildArguments();

        // Get the directory containing llama-server for DLL resolution
        var workingDir = Path.GetDirectoryName(_serverPath)!;

        // Start process via guardian
        var startInfo = new ProcessStartInfo
        {
            FileName = _serverPath,
            Arguments = args,
            WorkingDirectory = workingDir,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true
        };

        _process = _guardian.StartProcessWithStartInfo(startInfo);

        // Capture stderr output for diagnostics
        var stderrBuilder = new System.Text.StringBuilder();
        _process.ErrorDataReceived += (_, e) =>
        {
            if (e.Data != null)
            {
                stderrBuilder.AppendLine(e.Data);
            }
        };
        _process.BeginErrorReadLine();

        // Wait for server to be ready
        var startTime = DateTimeOffset.UtcNow;
        var ready = await WaitForServerReadyAsync(_config.StartupTimeout, cancellationToken);

        if (!ready)
        {
            // Collect error output
            var error = stderrBuilder.ToString();
            if (string.IsNullOrEmpty(error) && _process.HasExited)
            {
                error = "Process exited without error output";
            }

            throw new InvalidOperationException(
                $"llama-server failed to start within {_config.StartupTimeout.TotalSeconds}s. " +
                $"Exit code: {(_process.HasExited ? _process.ExitCode : "still running")}. " +
                $"Error: {error}");
        }

        Info = new LlamaServerInfo
        {
            ProcessId = _process.Id,
            Port = _port,
            ModelPath = _config.ModelPath,
            Backend = _backend,
            StartTime = startTime,
            StartupLog = stderrBuilder.ToString()
        };
    }

    private string BuildArguments()
    {
        var args = new List<string>
        {
            "--model", $"\"{_config.ModelPath}\"",
            "--port", _port.ToString(),
            "--ctx-size", _config.ContextSize.ToString(),
            "--n-gpu-layers", _config.GpuLayers.ToString(),
            "--batch-size", _config.BatchSize.ToString(),
            "--parallel", _config.Parallel.ToString(),
            "--host", "127.0.0.1", // Only listen on localhost for security
            "--cont-batching"      // Enable continuous batching for better throughput
        };

        if (_config.FlashAttention)
        {
            args.Add("--flash-attn");
        }

        if (_config.AdditionalArgs != null)
        {
            args.AddRange(_config.AdditionalArgs);
        }

        return string.Join(" ", args);
    }

    private async Task<bool> WaitForServerReadyAsync(TimeSpan timeout, CancellationToken cancellationToken)
    {
        var deadline = DateTime.UtcNow + timeout;

        while (DateTime.UtcNow < deadline)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (_process?.HasExited == true)
            {
                return false;
            }

            try
            {
                var response = await _httpClient.GetAsync($"http://localhost:{_port}/health", cancellationToken);
                if (response.IsSuccessStatusCode)
                {
                    return true;
                }
            }
            catch (HttpRequestException)
            {
                // Server not ready yet
            }
            catch (TaskCanceledException)
            {
                // Timeout, retry
            }

            await Task.Delay(100, cancellationToken);
        }

        return false;
    }

    /// <summary>
    /// Checks if the server is healthy.
    /// </summary>
    public async Task<bool> CheckHealthAsync(CancellationToken cancellationToken = default)
    {
        if (!IsRunning)
            return false;

        try
        {
            var response = await _httpClient.GetAsync($"http://localhost:{_port}/health", cancellationToken);
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// Stops the server gracefully.
    /// </summary>
    public async Task StopAsync(CancellationToken cancellationToken = default)
    {
        if (_process == null || _process.HasExited)
            return;

        try
        {
            // Try graceful shutdown first
            _process.Kill(entireProcessTree: true);
            await _process.WaitForExitAsync(cancellationToken);
        }
        catch (InvalidOperationException)
        {
            // Process already exited
        }
    }

    private static int FindAvailablePort()
    {
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        var port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed)
            return;

        _disposed = true;

        await StopAsync();
        _httpClient.Dispose();
        _guardian.Dispose();
        _process?.Dispose();
    }
}
