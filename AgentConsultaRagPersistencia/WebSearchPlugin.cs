using Microsoft.SemanticKernel;
using System.ComponentModel;
using System.Text.Json;
using Tavily;

namespace AgentConsultaRagPersistencia;


/// <summary>
/// Plugin de busca web usando o SDK oficial do Tavily para .NET (tryAGI).
/// Equivalente ao TavilySearch do notebook Python:
///   tool = TavilySearch(max_results=3, tavily_api_key=...)
///
/// A API key é lida da variável de ambiente TAVILY_API_KEY.
/// </summary>
public class WebSearchPlugin : IDisposable
{
    private readonly TavilyClient _tavily;
    private readonly HttpClient _http;
    private readonly string _apiKey;

    public WebSearchPlugin(HttpClient http)
    {
        _http = http;

        _apiKey = "";

        // Equivalente ao: TavilyClient(api_key=...) do Python
        _tavily = new TavilyClient();
    }

    [KernelFunction("search_web")]
    [Description("Busca informações atuais na web usando Tavily. Use quando precisar de dados recentes ou que possam ter mudado.")]
    public async Task<string> SearchWebAsync(
        [Description("Termo de busca")] string query,
        [Description("Número máximo de resultados (padrão 4)")] int maxResults = 4)
    {
        var results = new List<WebSearchResult>();

        try
        {
            // Equivalente ao: client.search(query, max_results=N) do Python
            var response = await _tavily.SearchAsync(_apiKey,
                query: query,
                maxResults: maxResults);

            foreach (var item in response.Results ?? [])
            {
                results.Add(new WebSearchResult
                {
                    Title = item.Title ?? string.Empty,
                    Url = item.Url ?? string.Empty,
                    Snippet = item.Content ?? string.Empty
                });
            }
        }
        catch (Exception ex)
        {
            results.Add(new WebSearchResult
            {
                Title = "Erro na busca",
                Url = string.Empty,
                Snippet = $"Falha ao buscar '{query}' via Tavily: {ex.Message}"
            });
        }

        return JsonSerializer.Serialize(results, new JsonSerializerOptions { WriteIndented = true });
    }

    [KernelFunction("fetch_page")]
    [Description("Acessa uma URL e retorna o conteúdo de texto da página.")]
    public async Task<string> FetchPageAsync(
        [Description("URL completa da página")] string url,
        [Description("Máximo de caracteres a retornar")] int maxChars = 3000)
    {
        try
        {
            _http.DefaultRequestHeaders.Clear();
            _http.DefaultRequestHeaders.Add("User-Agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36");

            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(12));
            var html = await _http.GetStringAsync(url, cts.Token);
            var text = ExtractText(html);

            return text.Length > maxChars
                ? text[..maxChars] + "\n[... truncado ...]"
                : text;
        }
        catch (Exception ex)
        {
            return $"Não foi possível acessar {url}: {ex.Message}";
        }
    }

    public void Dispose() => _tavily.Dispose();

    // ── helper: extrai texto limpo do HTML ────────────────────────────────────

    private static string ExtractText(string html)
    {
        var doc = new HtmlAgilityPack.HtmlDocument();
        doc.LoadHtml(html);

        var remove = doc.DocumentNode
            .SelectNodes("//script|//style|//nav|//header|//footer|//iframe");
        if (remove != null)
            foreach (var n in remove.ToList()) n.Remove();

        var nodes = doc.DocumentNode.SelectNodes("//p|//h1|//h2|//h3|//li");
        if (nodes == null) return doc.DocumentNode.InnerText;

        return string.Join("\n",
            nodes.Select(n => HtmlAgilityPack.HtmlEntity.DeEntitize(n.InnerText).Trim())
                 .Where(t => t.Length > 20)
                 .Distinct());
    }
}
