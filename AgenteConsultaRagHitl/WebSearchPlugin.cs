using Microsoft.SemanticKernel;
using System.ComponentModel;
using System.Text.Json;
using Tavily;

namespace AgenteConsultaRagHitl;

/// <summary>
/// Plugin de busca usando o SDK oficial do Tavily para .NET.
/// Equivalente ao TavilySearchResults do notebook Python.
/// </summary>
public class WebSearchPlugin : IDisposable
{
    private readonly TavilyClient _tavily;
    private readonly string _apiKey;

    public WebSearchPlugin()
    {
        _tavily = new TavilyClient();
        _apiKey = "";
    }

    [KernelFunction("search_web")]
    [Description("Busca informações atuais na web. Use para dados recentes ou em tempo real.")]
    public async Task<string> SearchWebAsync(
        [Description("Termo de busca")] string query,
        [Description("Máximo de resultados")] int maxResults = 4)
    {
        var results = new List<WebSearchResult>();
        try
        {
            var response = await _tavily.SearchAsync(apiKey: _apiKey,
                query: query, maxResults: maxResults);

            foreach (var item in response.Results ?? [])
                results.Add(new WebSearchResult
                {
                    Title = item.Title ?? string.Empty,
                    Url = item.Url ?? string.Empty,
                    Snippet = item.Content ?? string.Empty
                });
        }
        catch (Exception ex)
        {
            results.Add(new WebSearchResult
            {
                Title = "Erro",
                Snippet = $"Falha ao buscar '{query}': {ex.Message}"
            });
        }
        return JsonSerializer.Serialize(results, new JsonSerializerOptions { WriteIndented = true });
    }

    /// <summary>Executa a busca diretamente (sem ser via KernelFunction).</summary>
    public async Task<string> SearchDirectAsync(string query, int maxResults = 4) =>
        await SearchWebAsync(query, maxResults);

    public void Dispose() => _tavily.Dispose();
}

