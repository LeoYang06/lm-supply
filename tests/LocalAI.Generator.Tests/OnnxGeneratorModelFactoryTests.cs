using FluentAssertions;

namespace LocalAI.Generator.Tests;

public class OnnxGeneratorModelFactoryTests
{
    [Fact]
    public void Constructor_Default_CreatesFactory()
    {
        // Act
        var factory = new OnnxGeneratorModelFactory();

        // Assert
        factory.Should().NotBeNull();
    }

    [Fact]
    public void Constructor_WithParameters_CreatesFactory()
    {
        // Arrange
        var cacheDir = Path.GetTempPath();

        // Act
        var factory = new OnnxGeneratorModelFactory(cacheDir, ExecutionProvider.Cpu);

        // Assert
        factory.Should().NotBeNull();
    }

    [Fact]
    public void IsModelAvailable_NonExistentModel_ReturnsFalse()
    {
        // Arrange
        var factory = new OnnxGeneratorModelFactory();

        // Act
        var isAvailable = factory.IsModelAvailable("nonexistent/model");

        // Assert
        isAvailable.Should().BeFalse();
    }

    [Fact]
    public void GetModelCachePath_ReturnsExpectedFormat()
    {
        // Arrange
        var cacheDir = "/test/cache";
        var factory = new OnnxGeneratorModelFactory(cacheDir, ExecutionProvider.Auto);

        // Act
        var path = factory.GetModelCachePath("microsoft/Phi-3.5-mini-instruct-onnx");

        // Assert
        path.Should().Contain("models--microsoft--Phi-3.5-mini-instruct-onnx");
    }

    [Fact]
    public void GetAvailableModels_EmptyCache_ReturnsEmpty()
    {
        // Arrange
        var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        var factory = new OnnxGeneratorModelFactory(tempDir, ExecutionProvider.Auto);

        // Act
        var models = factory.GetAvailableModels();

        // Assert
        models.Should().BeEmpty();
    }

    [Fact]
    public async Task CreateAsync_NonExistentModel_ThrowsException()
    {
        // Arrange
        var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        var factory = new OnnxGeneratorModelFactory(tempDir, ExecutionProvider.Auto);

        // Act & Assert
        var action = () => factory.CreateAsync("nonexistent/model");
        await action.Should().ThrowAsync<NotSupportedException>()
            .WithMessage("*Automatic model download not yet implemented*");
    }

    [Fact]
    public async Task DownloadModelAsync_NotImplemented_ThrowsNotSupportedException()
    {
        // Arrange
        var factory = new OnnxGeneratorModelFactory();

        // Act & Assert
        var action = () => factory.DownloadModelAsync("test/model");
        await action.Should().ThrowAsync<NotSupportedException>();
    }
}
