using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace LMSupply.Llama.Server;

/// <summary>
/// OpenAI-compatible HTTP client for llama-server.
/// </summary>
public sealed class LlamaServerClient : IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly string _baseUrl;
    private readonly bool _ownsHttpClient;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    /// <summary>
    /// Creates a new client for the specified server URL.
    /// </summary>
    public LlamaServerClient(string baseUrl, HttpClient? httpClient = null)
    {
        _baseUrl = baseUrl.TrimEnd('/');

        if (httpClient != null)
        {
            _httpClient = httpClient;
            _ownsHttpClient = false;
        }
        else
        {
            _httpClient = new HttpClient();
            _ownsHttpClient = true;
        }
    }

    /// <summary>
    /// Generates a streaming chat completion.
    /// </summary>
    public async IAsyncEnumerable<string> GenerateChatAsync(
        IEnumerable<ChatCompletionMessage> messages,
        ChatCompletionOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        options ??= new ChatCompletionOptions();

        var request = new ChatCompletionRequest
        {
            Messages = messages.ToList(),
            MaxTokens = options.MaxTokens,
            Temperature = options.Temperature,
            TopP = options.TopP,
            Stream = true,
            Stop = options.StopSequences?.ToList()
        };

        var json = JsonSerializer.Serialize(request, JsonOptions);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        using var httpRequest = new HttpRequestMessage(HttpMethod.Post, $"{_baseUrl}/v1/chat/completions")
        {
            Content = content
        };

        using var response = await _httpClient.SendAsync(
            httpRequest,
            HttpCompletionOption.ResponseHeadersRead,
            cancellationToken);

        response.EnsureSuccessStatusCode();

        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
        using var reader = new StreamReader(stream);

        while (!reader.EndOfStream)
        {
            var line = await reader.ReadLineAsync(cancellationToken);

            if (string.IsNullOrEmpty(line))
                continue;

            if (!line.StartsWith("data: "))
                continue;

            var data = line[6..];

            if (data == "[DONE]")
                break;

            ChatCompletionChunk? chunk;
            try
            {
                chunk = JsonSerializer.Deserialize<ChatCompletionChunk>(data, JsonOptions);
            }
            catch
            {
                continue;
            }

            var delta = chunk?.Choices?.FirstOrDefault()?.Delta?.Content;
            if (!string.IsNullOrEmpty(delta))
            {
                yield return delta;
            }
        }
    }

    /// <summary>
    /// Generates a non-streaming chat completion.
    /// </summary>
    public async Task<string> GenerateChatCompleteAsync(
        IEnumerable<ChatCompletionMessage> messages,
        ChatCompletionOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var sb = new StringBuilder();

        await foreach (var token in GenerateChatAsync(messages, options, cancellationToken))
        {
            sb.Append(token);
        }

        return sb.ToString();
    }

    /// <summary>
    /// Generates a streaming text completion.
    /// </summary>
    public async IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        CompletionOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        options ??= new CompletionOptions();

        var request = new CompletionRequest
        {
            Prompt = prompt,
            NPredict = options.MaxTokens,
            Temperature = options.Temperature,
            TopP = options.TopP,
            Stream = true,
            Stop = options.StopSequences?.ToList()
        };

        var json = JsonSerializer.Serialize(request, JsonOptions);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");

        using var httpRequest = new HttpRequestMessage(HttpMethod.Post, $"{_baseUrl}/completion")
        {
            Content = content
        };

        using var response = await _httpClient.SendAsync(
            httpRequest,
            HttpCompletionOption.ResponseHeadersRead,
            cancellationToken);

        response.EnsureSuccessStatusCode();

        await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
        using var reader = new StreamReader(stream);

        while (!reader.EndOfStream)
        {
            var line = await reader.ReadLineAsync(cancellationToken);

            if (string.IsNullOrEmpty(line))
                continue;

            if (!line.StartsWith("data: "))
                continue;

            var data = line[6..];

            CompletionChunk? chunk;
            try
            {
                chunk = JsonSerializer.Deserialize<CompletionChunk>(data, JsonOptions);
            }
            catch
            {
                continue;
            }

            if (chunk?.Stop == true)
                break;

            if (!string.IsNullOrEmpty(chunk?.Content))
            {
                yield return chunk.Content;
            }
        }
    }

    /// <summary>
    /// Checks if the server is healthy.
    /// </summary>
    public async Task<bool> CheckHealthAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/health", cancellationToken);
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    public void Dispose()
    {
        if (_ownsHttpClient)
        {
            _httpClient.Dispose();
        }
    }
}

#region Request/Response Models

/// <summary>
/// Chat completion message.
/// </summary>
public sealed class ChatCompletionMessage
{
    [JsonPropertyName("role")]
    public required string Role { get; init; }

    [JsonPropertyName("content")]
    public required string Content { get; init; }

    public static ChatCompletionMessage System(string content) => new() { Role = "system", Content = content };
    public static ChatCompletionMessage User(string content) => new() { Role = "user", Content = content };
    public static ChatCompletionMessage Assistant(string content) => new() { Role = "assistant", Content = content };
}

/// <summary>
/// Options for chat completion.
/// </summary>
public sealed class ChatCompletionOptions
{
    public int MaxTokens { get; init; } = 256;
    public float Temperature { get; init; } = 0.7f;
    public float TopP { get; init; } = 0.9f;
    public IReadOnlyList<string>? StopSequences { get; init; }
}

/// <summary>
/// Options for text completion.
/// </summary>
public sealed class CompletionOptions
{
    public int MaxTokens { get; init; } = 256;
    public float Temperature { get; init; } = 0.7f;
    public float TopP { get; init; } = 0.9f;
    public IReadOnlyList<string>? StopSequences { get; init; }
}

internal sealed class ChatCompletionRequest
{
    public List<ChatCompletionMessage>? Messages { get; set; }
    public int? MaxTokens { get; set; }
    public float? Temperature { get; set; }
    public float? TopP { get; set; }
    public bool Stream { get; set; }
    public List<string>? Stop { get; set; }
}

internal sealed class CompletionRequest
{
    public string? Prompt { get; set; }
    public int? NPredict { get; set; }
    public float? Temperature { get; set; }
    public float? TopP { get; set; }
    public bool Stream { get; set; }
    public List<string>? Stop { get; set; }
}

internal sealed class ChatCompletionChunk
{
    public List<ChatCompletionChoice>? Choices { get; set; }
}

internal sealed class ChatCompletionChoice
{
    public ChatCompletionDelta? Delta { get; set; }
    public string? FinishReason { get; set; }
}

internal sealed class ChatCompletionDelta
{
    public string? Content { get; set; }
}

internal sealed class CompletionChunk
{
    public string? Content { get; set; }
    public bool Stop { get; set; }
}

#endregion
