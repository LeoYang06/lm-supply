using System.Text.Json.Serialization;

namespace LMSupply.Llama.Server;

/// <summary>
/// Persisted state for a llama-server installation.
/// </summary>
public sealed class LlamaServerVersionState
{
    /// <summary>
    /// Currently installed version (e.g., "b7898").
    /// </summary>
    [JsonPropertyName("installedVersion")]
    public required string InstalledVersion { get; set; }

    /// <summary>
    /// Path to the installed version's directory.
    /// </summary>
    [JsonPropertyName("installedPath")]
    public required string InstalledPath { get; set; }

    /// <summary>
    /// Latest known version from GitHub (e.g., "b7900").
    /// </summary>
    [JsonPropertyName("latestKnownVersion")]
    public string? LatestKnownVersion { get; set; }

    /// <summary>
    /// When the latest version was last checked.
    /// </summary>
    [JsonPropertyName("lastVersionCheck")]
    public DateTimeOffset LastVersionCheck { get; set; }

    /// <summary>
    /// Version that has been downloaded but not yet activated.
    /// </summary>
    [JsonPropertyName("pendingVersion")]
    public string? PendingVersion { get; set; }

    /// <summary>
    /// Path to the pending version's directory.
    /// </summary>
    [JsonPropertyName("pendingPath")]
    public string? PendingPath { get; set; }

    /// <summary>
    /// Whether an update is ready to be applied.
    /// </summary>
    [JsonPropertyName("updateReady")]
    public bool UpdateReady { get; set; }

    /// <summary>
    /// Previous versions kept for rollback (newest first).
    /// </summary>
    [JsonPropertyName("previousVersions")]
    public List<VersionEntry> PreviousVersions { get; set; } = [];

    /// <summary>
    /// Versions that failed to load (will be skipped in future checks).
    /// </summary>
    [JsonPropertyName("failedVersions")]
    public HashSet<string> FailedVersions { get; set; } = [];

    /// <summary>
    /// The GPU backend this state applies to.
    /// </summary>
    [JsonPropertyName("backend")]
    public string Backend { get; set; } = "vulkan";
}

/// <summary>
/// Entry for a previous version kept for rollback.
/// </summary>
public sealed class VersionEntry
{
    /// <summary>
    /// Version string (e.g., "b7898").
    /// </summary>
    [JsonPropertyName("version")]
    public required string Version { get; set; }

    /// <summary>
    /// Path to the version's directory.
    /// </summary>
    [JsonPropertyName("path")]
    public required string Path { get; set; }

    /// <summary>
    /// When this version was installed.
    /// </summary>
    [JsonPropertyName("installedAt")]
    public DateTimeOffset InstalledAt { get; set; }
}

/// <summary>
/// Root object for the persisted state file.
/// </summary>
public sealed class LlamaServerStateFile
{
    /// <summary>
    /// Schema version for forward compatibility.
    /// </summary>
    [JsonPropertyName("schemaVersion")]
    public int SchemaVersion { get; set; } = 1;

    /// <summary>
    /// State entries keyed by "{backend}|{platform}" (e.g., "vulkan|win-x64").
    /// </summary>
    [JsonPropertyName("entries")]
    public Dictionary<string, LlamaServerVersionState> Entries { get; set; } = [];
}
