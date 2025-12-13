using FluentAssertions;
using LocalAI.Generator.Models;

namespace LocalAI.Generator.Tests;

public class GeneratorOptionsTests
{
    [Fact]
    public void Default_ReturnsExpectedValues()
    {
        // Act
        var options = GeneratorOptions.Default;

        // Assert
        options.MaxTokens.Should().Be(512);
        options.Temperature.Should().Be(0.7f);
        options.TopP.Should().Be(0.9f);
        options.TopK.Should().Be(50);
        options.RepetitionPenalty.Should().Be(1.1f);
    }

    [Fact]
    public void Creative_HasHigherTemperature()
    {
        // Act
        var options = GeneratorOptions.Creative;

        // Assert
        options.Temperature.Should().Be(0.9f);
        options.TopP.Should().Be(0.95f);
        options.TopK.Should().Be(100);
    }

    [Fact]
    public void Precise_HasLowerTemperature()
    {
        // Act
        var options = GeneratorOptions.Precise;

        // Assert
        options.Temperature.Should().Be(0.1f);
        options.TopP.Should().Be(0.5f);
        options.TopK.Should().Be(10);
    }

    [Fact]
    public void Default_HasExpectedSamplingOptions()
    {
        // Act
        var options = GeneratorOptions.Default;

        // Assert - New options from research-05
        options.DoSample.Should().BeTrue();
        options.NumBeams.Should().Be(1);
        options.PastPresentShareBuffer.Should().BeTrue();
        options.MaxNewTokens.Should().BeNull();
    }

    [Fact]
    public void BeamSearch_Configuration()
    {
        // Arrange
        var options = new GeneratorOptions
        {
            NumBeams = 4,
            DoSample = false
        };

        // Assert
        options.NumBeams.Should().Be(4);
        options.DoSample.Should().BeFalse();
    }

    [Fact]
    public void MaxNewTokens_CanBeLimited()
    {
        // Arrange
        var options = new GeneratorOptions
        {
            MaxTokens = 2048,
            MaxNewTokens = 100
        };

        // Assert
        options.MaxTokens.Should().Be(2048);
        options.MaxNewTokens.Should().Be(100);
    }
}
