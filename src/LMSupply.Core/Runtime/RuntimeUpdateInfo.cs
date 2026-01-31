namespace LMSupply.Runtime;

/// <summary>
/// Information about the runtime update status.
/// Returned by GetRuntimeUpdateInfo() for diagnostics.
/// </summary>
public sealed record RuntimeUpdateInfo
{
    /// <summary>
    /// Gets the currently installed version.
    /// </summary>
    public required string InstalledVersion { get; init; }

    /// <summary>
    /// Gets the latest known version from NuGet (may be null if not checked yet).
    /// </summary>
    public string? LatestVersion { get; init; }

    /// <summary>
    /// Gets whether an update is available.
    /// </summary>
    public bool UpdateAvailable => LatestVersion is not null &&
        !string.Equals(InstalledVersion, LatestVersion, StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// Gets whether an update has been downloaded and is ready to apply on next load.
    /// </summary>
    public bool UpdateReady { get; init; }

    /// <summary>
    /// Gets the path to the ready update (if UpdateReady is true).
    /// </summary>
    public string? UpdateReadyPath { get; init; }

    /// <summary>
    /// Gets the timestamp of the last version check.
    /// </summary>
    public DateTimeOffset LastChecked { get; init; }

    /// <summary>
    /// Gets the package identifier.
    /// </summary>
    public string? PackageId { get; init; }

    /// <summary>
    /// Gets the provider/backend name.
    /// </summary>
    public string? Provider { get; init; }
}

/// <summary>
/// Result of an update operation.
/// </summary>
public sealed record RuntimeUpdateResult
{
    /// <summary>
    /// Gets whether an update was applied.
    /// </summary>
    public bool Updated { get; init; }

    /// <summary>
    /// Gets the previous version (if updated).
    /// </summary>
    public string? PreviousVersion { get; init; }

    /// <summary>
    /// Gets the new version (if updated).
    /// </summary>
    public string? NewVersion { get; init; }

    /// <summary>
    /// Gets the path to the runtime binaries.
    /// </summary>
    public string? RuntimePath { get; init; }

    /// <summary>
    /// Gets whether this was a rollback operation.
    /// </summary>
    public bool WasRollback { get; init; }

    /// <summary>
    /// Gets any error message if the operation failed.
    /// </summary>
    public string? ErrorMessage { get; init; }

    /// <summary>
    /// Creates a successful result for no update needed.
    /// </summary>
    public static RuntimeUpdateResult NoUpdateNeeded(string version, string path) => new()
    {
        Updated = false,
        NewVersion = version,
        RuntimePath = path
    };

    /// <summary>
    /// Creates a successful result for an applied update.
    /// </summary>
    public static RuntimeUpdateResult UpdateApplied(string previous, string current, string path) => new()
    {
        Updated = true,
        PreviousVersion = previous,
        NewVersion = current,
        RuntimePath = path
    };

    /// <summary>
    /// Creates a result for a rollback operation.
    /// </summary>
    public static RuntimeUpdateResult Rollback(string failedVersion, string rolledBackTo, string path) => new()
    {
        Updated = true,
        WasRollback = true,
        PreviousVersion = failedVersion,
        NewVersion = rolledBackTo,
        RuntimePath = path
    };

    /// <summary>
    /// Creates a failed result.
    /// </summary>
    public static RuntimeUpdateResult Failed(string error) => new()
    {
        Updated = false,
        ErrorMessage = error
    };
}
