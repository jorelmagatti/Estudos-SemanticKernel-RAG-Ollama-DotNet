using Microsoft.Extensions.VectorData;

namespace FutbolRulesRAGSemanticKernel;


public sealed class DocumentRecord
{
    [VectorStoreKey]
    public string Id { get; set; } = Guid.NewGuid().ToString();

    [VectorStoreData]
    public string Content { get; set; } = string.Empty;

    [VectorStoreData]
    public int PageNumber { get; set; }

    [VectorStoreData]
    public string Source { get; set; } = string.Empty;

    // Dimensão do vetor deve bater com o modelo de embedding:
    // nomic-embed-text → 768 dimensões
    [VectorStoreVector(Dimensions: 768, DistanceFunction = DistanceFunction.CosineSimilarity)]
    public ReadOnlyMemory<float> Embedding { get; set; }
}