
namespace FarmaceuticAgentRagSemantickernel;

using Microsoft.Extensions.VectorData;

public class DocumentRecord
{
    [VectorStoreKey]
    public string Id { get; set; } = Guid.NewGuid().ToString();

    [VectorStoreData]
    public string Content { get; set; } = string.Empty;

    [VectorStoreData]
    public string Source { get; set; } = string.Empty;

    [VectorStoreData]
    public int Page { get; set; }

    [VectorStoreData]
    public string Medicamento { get; set; } = string.Empty;

    [VectorStoreData]
    public string Categoria { get; set; } = "geral";

    [VectorStoreVector(Dimensions: 768, DistanceFunction = DistanceFunction.CosineSimilarity)]
    public ReadOnlyMemory<float> Embedding { get; set; }
}