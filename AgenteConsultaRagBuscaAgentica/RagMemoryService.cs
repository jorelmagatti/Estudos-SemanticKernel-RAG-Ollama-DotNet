namespace AgenteConsultaRagBuscaAgentica;

using Microsoft.SemanticKernel.Embeddings;

/// <summary>
/// Serviço de memória RAG com vetor store em memória.
/// Armazena chunks de documentos e recupera os mais relevantes via similaridade coseno,
/// replicando a estratégia de recuperação de contexto do notebook Python.
/// </summary>
public class RagMemoryService
{
    private readonly ITextEmbeddingGenerationService _embeddingService;
    private readonly List<DocumentChunk> _vectorStore = new();
    private readonly int _chunkSize;
    private readonly int _chunkOverlap;

    public RagMemoryService(
        ITextEmbeddingGenerationService embeddingService,
        int chunkSize = 800,
        int chunkOverlap = 150)
    {
        _embeddingService = embeddingService;
        _chunkSize = chunkSize;
        _chunkOverlap = chunkOverlap;
    }

    /// <summary>
    /// Ingere texto de uma URL, divide em chunks e gera embeddings.
    /// Equivalente ao processo de indexação do RAG.
    /// </summary>
    public async Task IngestAsync(string text, string sourceUrl)
    {
        if (string.IsNullOrWhiteSpace(text)) return;

        var chunks = SplitIntoChunks(text, sourceUrl);

        Console.ForegroundColor = ConsoleColor.DarkCyan;
        Console.WriteLine($"  📥 [RAG] Indexando {chunks.Count} chunk(s) de: {sourceUrl[..Math.Min(60, sourceUrl.Length)]}...");
        Console.ResetColor();

        foreach (var chunk in chunks)
        {
            try
            {
                var embedding = await _embeddingService.GenerateEmbeddingAsync(chunk.Content);
                chunk.Embedding = embedding.ToArray();
                _vectorStore.Add(chunk);
            }
            catch (Exception ex)
            {
                // Se embeddings não disponíveis, armazena sem vetor (busca por keyword como fallback)
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"  ⚠️  [RAG] Embedding falhou, usando fallback keyword: {ex.Message}");
                Console.ResetColor();
                chunk.Embedding = null;
                _vectorStore.Add(chunk);
            }
        }

        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"  ✅ [RAG] Vector store: {_vectorStore.Count} chunk(s) total");
        Console.ResetColor();
    }

    /// <summary>
    /// Recupera os chunks mais relevantes para a query.
    /// Usa similaridade coseno se embeddings disponíveis, ou busca por keyword como fallback.
    /// </summary>
    public async Task<List<DocumentChunk>> RetrieveAsync(string query, int topK = 5)
    {
        if (_vectorStore.Count == 0) return new List<DocumentChunk>();

        // Tenta similaridade vetorial
        var chunksWithEmbeddings = _vectorStore.Where(c => c.Embedding != null).ToList();

        if (chunksWithEmbeddings.Count > 0)
        {
            try
            {
                var queryEmbedding = await _embeddingService.GenerateEmbeddingAsync(query);
                var queryVector = queryEmbedding.ToArray();

                return chunksWithEmbeddings
                    .Select(c => (chunk: c, score: CosineSimilarity(queryVector, c.Embedding!)))
                    .OrderByDescending(x => x.score)
                    .Take(topK)
                    .Select(x => x.chunk)
                    .ToList();
            }
            catch
            {
                // fallback para keyword
            }
        }

        // Fallback: busca por presença de palavras da query
        var queryWords = query.ToLower().Split(' ', StringSplitOptions.RemoveEmptyEntries);
        return _vectorStore
            .Select(c => (chunk: c, score: queryWords.Count(w => c.Content.ToLower().Contains(w))))
            .Where(x => x.score > 0)
            .OrderByDescending(x => x.score)
            .Take(topK)
            .Select(x => x.chunk)
            .ToList();
    }

    public int ChunkCount => _vectorStore.Count;

    // ─── helpers privados ───────────────────────────────────────────────────

    private List<DocumentChunk> SplitIntoChunks(string text, string sourceUrl)
    {
        var chunks = new List<DocumentChunk>();
        var lines = text.Split('\n', StringSplitOptions.RemoveEmptyEntries);
        var current = new System.Text.StringBuilder();

        foreach (var line in lines)
        {
            current.AppendLine(line);

            if (current.Length >= _chunkSize)
            {
                var chunkText = current.ToString().Trim();
                if (!string.IsNullOrWhiteSpace(chunkText))
                    chunks.Add(new DocumentChunk { SourceUrl = sourceUrl, Content = chunkText });

                // Overlap: mantém as últimas N linhas para continuidade
                var overlapText = string.Join("\n",
                    current.ToString().Split('\n').TakeLast(3));
                current.Clear();
                current.AppendLine(overlapText);
            }
        }

        // Último chunk residual
        var last = current.ToString().Trim();
        if (!string.IsNullOrWhiteSpace(last))
            chunks.Add(new DocumentChunk { SourceUrl = sourceUrl, Content = last });

        return chunks;
    }

    private static float CosineSimilarity(float[] a, float[] b)
    {
        if (a.Length != b.Length) return 0f;
        float dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return (normA == 0 || normB == 0) ? 0f : dot / (MathF.Sqrt(normA) * MathF.Sqrt(normB));
    }
}
