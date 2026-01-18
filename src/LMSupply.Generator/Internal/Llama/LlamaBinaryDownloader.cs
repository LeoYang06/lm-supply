using System.IO.Compression;
using System.Runtime.InteropServices;
using LMSupply.Runtime;

namespace LMSupply.Generator.Internal.Llama;

/// <summary>
/// Downloads llama.cpp native binaries from GitHub releases.
/// Implements on-demand downloading with caching.
/// </summary>
public sealed class LlamaBinaryDownloader : IDisposable
{
    private const string GitHubApiBase = "https://api.github.com/repos/ggml-org/llama.cpp/releases";
    private const string GitHubDownloadBase = "https://github.com/ggml-org/llama.cpp/releases/download";

    // LLamaSharp v0.25.0 compatible version (approximate - use latest stable for best compatibility)
    // The exact compatible version should be determined from LLamaSharp release notes
    private const string DefaultVersion = "b4000"; // Placeholder - update based on LLamaSharp compatibility

    private readonly HttpClient _httpClient;
    private readonly string _cacheDirectory;

    public LlamaBinaryDownloader() : this(null)
    {
    }

    public LlamaBinaryDownloader(string? cacheDirectory)
    {
        _cacheDirectory = cacheDirectory ?? GetDefaultCacheDirectory();
        _httpClient = new HttpClient();
        _httpClient.DefaultRequestHeaders.Add("User-Agent", "LMSupply/1.0");
    }

    /// <summary>
    /// Downloads the llama.cpp native binary for the specified backend and platform.
    /// Returns the path to the extracted binary directory.
    /// </summary>
    public async Task<string> DownloadAsync(
        LlamaBackend backend,
        PlatformInfo platform,
        string? version = null,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        version ??= DefaultVersion;

        // Check cache first
        var cachePath = GetCachePath(version, backend, platform);
        if (Directory.Exists(cachePath) && IsValidCache(cachePath))
        {
            progress?.Report(new DownloadProgress
            {
                FileName = "llama.cpp",
                BytesDownloaded = 0,
                TotalBytes = 0
            });
            return cachePath;
        }

        // Determine asset name
        var assetName = GetAssetName(backend, platform, version);
        if (assetName is null)
        {
            throw new NotSupportedException(
                $"No llama.cpp binary available for {backend} on {platform.RuntimeIdentifier}");
        }

        // Download
        progress?.Report(new DownloadProgress
        {
            FileName = assetName,
            BytesDownloaded = 0,
            TotalBytes = 0
        });

        var downloadUrl = $"{GitHubDownloadBase}/{version}/{assetName}";
        var tempPath = Path.Combine(Path.GetTempPath(), $"llama-{Guid.NewGuid()}");

        try
        {
            Directory.CreateDirectory(tempPath);

            // Download archive
            var archivePath = Path.Combine(tempPath, assetName);
            await DownloadFileAsync(downloadUrl, archivePath, assetName, progress, cancellationToken);

            // Extract
            progress?.Report(new DownloadProgress
            {
                FileName = $"Extracting {assetName}...",
                BytesDownloaded = 0,
                TotalBytes = 0
            });

            var extractPath = Path.Combine(tempPath, "extracted");
            Directory.CreateDirectory(extractPath);

            if (assetName.EndsWith(".zip", StringComparison.OrdinalIgnoreCase))
            {
                ZipFile.ExtractToDirectory(archivePath, extractPath);
            }
            else if (assetName.EndsWith(".tar.gz", StringComparison.OrdinalIgnoreCase))
            {
                await ExtractTarGzAsync(archivePath, extractPath, cancellationToken);
            }

            // Move to cache
            Directory.CreateDirectory(Path.GetDirectoryName(cachePath)!);
            if (Directory.Exists(cachePath))
            {
                Directory.Delete(cachePath, recursive: true);
            }

            // Find the actual binary directory (might be nested)
            var binaryDir = FindBinaryDirectory(extractPath, platform);
            if (binaryDir is null)
            {
                throw new InvalidOperationException("Could not find llama.cpp binaries in downloaded archive");
            }

            Directory.Move(binaryDir, cachePath);

            progress?.Report(new DownloadProgress
            {
                FileName = "llama.cpp ready",
                BytesDownloaded = 1,
                TotalBytes = 1
            });

            return cachePath;
        }
        finally
        {
            // Cleanup temp directory
            try
            {
                if (Directory.Exists(tempPath))
                {
                    Directory.Delete(tempPath, recursive: true);
                }
            }
            catch
            {
                // Ignore cleanup errors
            }
        }
    }

    /// <summary>
    /// Gets the asset name for the specified backend and platform.
    /// </summary>
    private static string? GetAssetName(LlamaBackend backend, PlatformInfo platform, string version)
    {
        // Asset naming pattern: llama-{version}-bin-{os}-{variant}-{arch}.{ext}
        // Windows: .zip, Linux/macOS: .tar.gz

        var isArm64 = platform.Architecture == Architecture.Arm64;
        var arch = isArm64 ? "arm64" : "x64";
        var ext = platform.IsWindows ? "zip" : "tar.gz";

        return (platform.IsWindows, platform.IsLinux, platform.IsMacOS, backend) switch
        {
            // Windows
            (true, _, _, LlamaBackend.Cpu) => $"llama-{version}-bin-win-cpu-{arch}.{ext}",
            (true, _, _, LlamaBackend.Cuda12) => $"llama-{version}-bin-win-cuda-12.4-x64.{ext}",
            (true, _, _, LlamaBackend.Cuda13) => $"llama-{version}-bin-win-cuda-13.1-x64.{ext}",
            (true, _, _, LlamaBackend.Vulkan) => $"llama-{version}-bin-win-vulkan-x64.{ext}",
            (true, _, _, LlamaBackend.Rocm) => $"llama-{version}-bin-win-hip-radeon-x64.{ext}",

            // Linux
            (_, true, _, LlamaBackend.Cpu) => $"llama-{version}-bin-ubuntu-x64.{ext}",
            (_, true, _, LlamaBackend.Vulkan) => $"llama-{version}-bin-ubuntu-vulkan-x64.{ext}",
            // Note: Linux CUDA builds require compilation or use ubuntu variants

            // macOS
            (_, _, true, LlamaBackend.Metal) when isArm64 => $"llama-{version}-bin-macos-arm64.{ext}",
            (_, _, true, LlamaBackend.Cpu) when isArm64 => $"llama-{version}-bin-macos-arm64.{ext}",
            (_, _, true, _) when !isArm64 => $"llama-{version}-bin-macos-x64.{ext}",

            _ => null
        };
    }

    private string GetCachePath(string version, LlamaBackend backend, PlatformInfo platform)
    {
        var backendStr = backend.ToString().ToLowerInvariant();
        return Path.Combine(_cacheDirectory, "llama.cpp", version, platform.RuntimeIdentifier, backendStr);
    }

    private static bool IsValidCache(string path)
    {
        if (!Directory.Exists(path))
            return false;

        // Check for expected llama library file
        var platform = EnvironmentDetector.DetectPlatform();
        var expectedLib = platform.IsWindows ? "llama.dll" :
                          platform.IsMacOS ? "libllama.dylib" :
                          "libllama.so";

        // Check in root and common subdirectories
        var searchPaths = new[]
        {
            path,
            Path.Combine(path, "bin"),
            Path.Combine(path, "lib")
        };

        return searchPaths.Any(p =>
            Directory.Exists(p) &&
            Directory.EnumerateFiles(p, expectedLib.Replace(".so", ".so*")).Any());
    }

    private static string? FindBinaryDirectory(string extractPath, PlatformInfo platform)
    {
        var expectedLib = platform.IsWindows ? "llama.dll" :
                          platform.IsMacOS ? "libllama.dylib" :
                          "libllama.so";

        // Search for the directory containing the main library
        var searchPattern = platform.IsWindows ? expectedLib : expectedLib.Replace(".so", ".so*");

        foreach (var file in Directory.EnumerateFiles(extractPath, searchPattern, SearchOption.AllDirectories))
        {
            return Path.GetDirectoryName(file);
        }

        // Fallback: return the first non-empty directory
        foreach (var dir in Directory.EnumerateDirectories(extractPath, "*", SearchOption.AllDirectories))
        {
            if (Directory.EnumerateFiles(dir).Any())
                return dir;
        }

        // Last resort: return extractPath if it has files
        if (Directory.EnumerateFiles(extractPath).Any())
            return extractPath;

        return null;
    }

    private async Task DownloadFileAsync(
        string url,
        string destinationPath,
        string fileName,
        IProgress<DownloadProgress>? progress,
        CancellationToken cancellationToken)
    {
        using var response = await _httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        response.EnsureSuccessStatusCode();

        var totalBytes = response.Content.Headers.ContentLength ?? 0;
        var downloadedBytes = 0L;

        await using var contentStream = await response.Content.ReadAsStreamAsync(cancellationToken);
        await using var fileStream = new FileStream(destinationPath, FileMode.Create, FileAccess.Write, FileShare.None, 8192, true);

        var buffer = new byte[8192];
        int bytesRead;

        while ((bytesRead = await contentStream.ReadAsync(buffer, cancellationToken)) > 0)
        {
            await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), cancellationToken);
            downloadedBytes += bytesRead;

            progress?.Report(new DownloadProgress
            {
                FileName = fileName,
                TotalBytes = totalBytes,
                BytesDownloaded = downloadedBytes
            });
        }
    }

    private static async Task ExtractTarGzAsync(string archivePath, string extractPath, CancellationToken cancellationToken)
    {
        // Use System.Formats.Tar for .tar.gz extraction
        await using var fileStream = new FileStream(archivePath, FileMode.Open, FileAccess.Read);
        await using var gzipStream = new GZipStream(fileStream, CompressionMode.Decompress);

        await System.Formats.Tar.TarFile.ExtractToDirectoryAsync(
            gzipStream,
            extractPath,
            overwriteFiles: true,
            cancellationToken: cancellationToken);
    }

    private static string GetDefaultCacheDirectory()
    {
        var baseDir = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        return Path.Combine(baseDir, "LMSupply", "cache", "runtimes");
    }

    public void Dispose()
    {
        _httpClient.Dispose();
    }
}
