namespace AgenteConsultaRagBuscaAgentica;

// <summary>
/// Representa um resultado de busca web (equivalente ao retorno do DuckDuckGo/Tavily no Python).
/// </summary>
public class SearchResult
{
    public string Title { get; set; } = string.Empty;
    public string Url { get; set; } = string.Empty;
    public string Snippet { get; set; } = string.Empty;
}

/// <summary>
/// Representa um chunk de texto extraído de uma página web para o vetor de contexto RAG.
/// </summary>
public class DocumentChunk
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string SourceUrl { get; set; } = string.Empty;
    public string Content { get; set; } = string.Empty;
    public float[]? Embedding { get; set; }
}

/// <summary>
/// Estado do agente durante o ciclo de raciocínio (equivalente ao State do LangGraph).
/// </summary>
public class AgentState
{
    public string OriginalQuery { get; set; } = string.Empty;
    public List<string> SearchQueries { get; set; } = new();
    public List<SearchResult> SearchResults { get; set; } = new();
    public List<DocumentChunk> RetrievedChunks { get; set; } = new();
    public string? FinalAnswer { get; set; }
    public int Iterations { get; set; } = 0;
    public bool IsComplete { get; set; } = false;
    public List<string> ThoughtProcess { get; set; } = new(); // log do raciocínio do agente
}
