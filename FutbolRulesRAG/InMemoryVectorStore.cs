namespace RaglangChainTestes;

public class InMemoryVectorStore
{
    private readonly record struct Entry(
        float[] Embedding,
        string Content,
        int PageNumber,
        string Source);

    private readonly List<Entry> _store = [];

    public void AddChunks(IEnumerable<(float[] Embedding, TextSplitter.Chunk Chunk)> items)
    {
        foreach (var (emb, chunk) in items)
        {
            _store.Add(new Entry(emb, chunk.Content, chunk.PageNumber, chunk.Source));
        }
    }

    /// <summary>
    /// Retorna os k chunks mais similares à query embedding.
    /// </summary>
    public List<TextSplitter.Chunk> SimilaritySearch(float[] queryEmbedding, int k = 3)
    {
        return _store
            .Select(entry => (Score: CosineSimilarity(queryEmbedding, entry.Embedding), Entry: entry))
            .OrderByDescending(x => x.Score)
            .Take(k)
            .Select(x => new TextSplitter.Chunk(x.Entry.Content, x.Entry.PageNumber, x.Entry.Source))
            .ToList();
    }

    private static float CosineSimilarity(float[] a, float[] b)
    {
        float dot = 0f, normA = 0f, normB = 0f;
        int len = Math.Min(a.Length, b.Length);

        for (int i = 0; i < len; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        float denom = MathF.Sqrt(normA) * MathF.Sqrt(normB);
        return denom == 0f ? 0f : dot / denom;
    }

    // 1. Dot Product (Inner Product) – muito usado e mais rápido
    private static float DotProduct(float[] a, float[] b)
    {
        float dot = 0f;
        int len = Math.Min(a.Length, b.Length);
        for (int i = 0; i < len; i++)
            dot += a[i] * b[i];
        return dot;
    }

    // 2. Euclidean Distance (L2) – quanto menor, mais similar
    private static float EuclideanDistance(float[] a, float[] b)
    {
        float sum = 0f;
        int len = Math.Min(a.Length, b.Length);
        for (int i = 0; i < len; i++)
        {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return MathF.Sqrt(sum);
    }

    // 3. Squared Euclidean – evita a raiz quadrada (mais rápido para comparação/ranking)
    private static float SquaredEuclidean(float[] a, float[] b)
    {
        float sum = 0f;
        int len = Math.Min(a.Length, b.Length);
        for (int i = 0; i < len; i++)
        {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    // 4. Manhattan (L1) – soma das diferenças absolutas
    private static float ManhattanDistance(float[] a, float[] b)
    {
        float sum = 0f;
        int len = Math.Min(a.Length, b.Length);
        for (int i = 0; i < len; i++)
            sum += MathF.Abs(a[i] - b[i]);
        return sum;
    }
}
