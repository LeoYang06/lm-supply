namespace LMSupply.Runtime;

/// <summary>
/// Configuration options for runtime auto-update behavior.
/// </summary>
public sealed class RuntimeUpdateOptions
{
    /// <summary>
    /// Default singleton instance with standard settings.
    /// </summary>
    public static RuntimeUpdateOptions Default { get; } = new();

    /// <summary>
    /// Gets or sets the interval between version checks.
    /// Default is 24 hours to avoid excessive NuGet API requests.
    /// </summary>
    public TimeSpan VersionCheckInterval { get; set; } = TimeSpan.FromHours(24);

    /// <summary>
    /// Gets or sets whether to automatically download updates in the background.
    /// When true, updates are downloaded but not applied until next load.
    /// Default is true.
    /// </summary>
    public bool AutoDownloadUpdates { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to check and apply updates during WarmupAsync.
    /// When true, WarmupAsync will block until any available update is downloaded and applied.
    /// Default is true.
    /// </summary>
    public bool UpdateOnWarmup { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to include prerelease versions.
    /// Default is false (stable versions only).
    /// </summary>
    public bool IncludePrerelease { get; set; }

    /// <summary>
    /// Gets or sets the maximum number of previous versions to keep for rollback.
    /// Older versions are cleaned up to save disk space.
    /// Default is 2.
    /// </summary>
    public int MaxVersionsToKeep { get; set; } = 2;

    /// <summary>
    /// Gets or sets the timeout for version check operations.
    /// Default is 30 seconds.
    /// </summary>
    public TimeSpan VersionCheckTimeout { get; set; } = TimeSpan.FromSeconds(30);

    /// <summary>
    /// Gets or sets the cache directory for runtime binaries.
    /// If null, uses the default LMSupply cache directory.
    /// </summary>
    public string? CacheDirectory { get; set; }
}
