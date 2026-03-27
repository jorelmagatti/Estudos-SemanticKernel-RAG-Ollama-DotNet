using Tavily;

namespace MultAgentConsultaRagGrafoSemanticKernel;

/// <summary>
/// Serviço de busca usando o SDK oficial do Tavily para .NET.
/// Equivalente ao tavily = TavilyClient(api_key=...) do notebook Python.
/// </summary>
public class TavilySearchService : IDisposable
{
    private readonly TavilyClient _client;
    private readonly string _apiKey;
    public TavilySearchService()
    {
        _client = new TavilyClient();
        _apiKey = "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";
    }

    /// <summary>
    /// Busca por uma query e retorna os conteúdos dos resultados.
    /// Equivalente ao tavily.search(query=q, max_results=2) do Python.
    /// </summary>
    public async Task<List<string>> SearchAsync(string query, int maxResults = 2)
    {
        try
        {
            var response = await _client.SearchAsync(_apiKey,
                query: query, maxResults: maxResults);

            return (response.Results ?? [])
                .Select(r => r.Content ?? string.Empty)
                .Where(c => !string.IsNullOrWhiteSpace(c))
                .ToList();
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"  ⚠️  Tavily erro para '{query}': {ex.Message}");
            Console.ResetColor();
            return new List<string>();
        }
    }

    /// <summary>
    /// Executa múltiplas queries e agrega os resultados.
    /// Equivalente ao loop for q in queries.queries do notebook Python.
    /// </summary>
    public async Task<List<string>> SearchMultipleAsync(IEnumerable<string> queries, int maxResultsEach = 2)
    {
        var allContent = new List<string>();
        foreach (var query in queries)
        {
            var results = await SearchAsync(query, maxResultsEach);
            allContent.AddRange(results);
        }
        return allContent;
    }

    public void Dispose() => _client.Dispose();
}
