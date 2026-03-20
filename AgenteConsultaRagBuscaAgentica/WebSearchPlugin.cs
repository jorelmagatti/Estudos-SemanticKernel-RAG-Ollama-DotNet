using Microsoft.SemanticKernel;
using System.ComponentModel;
using System.Text.Json;

namespace AgenteConsultaRagBuscaAgentica;

/// <summary>
/// Plugin de Busca Web para o Semantic Kernel.
/// Equivalente ao DuckDuckGo/Tavily do notebook Python.
/// Usa a API do DuckDuckGo (sem chave) para buscar URLs e snippets.
/// </summary>
public class WebSearchPlugin
{
    private readonly HttpClient _httpClient;

    public WebSearchPlugin(HttpClient httpClient)
    {
        _httpClient = httpClient;
        _httpClient.Timeout = TimeSpan.FromMinutes(1);
    }

    /// <summary>
    /// Busca resultados na web usando DuckDuckGo Instant Answer API.
    /// Retorna JSON com lista de resultados (título, URL, snippet).
    /// </summary>
    [KernelFunction("search_web")]
    [Description("Busca informações na web sobre um tópico. Retorna uma lista de resultados com título, URL e resumo do conteúdo.")]
    public async Task<string> SearchWebAsync(
        [Description("A query de busca para encontrar informações relevantes")] string query,
        [Description("Número máximo de resultados (padrão: 5)")] int maxResults = 5)
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine($"  🔍 [WebSearch] Buscando: \"{query}\"");
        Console.ResetColor();

        var results = new List<SearchResult>();

        try
        {
            // DuckDuckGo HTML search (sem necessidade de API key, igual ao notebook)
            var encodedQuery = Uri.EscapeDataString(query);
            var url = $"https://html.duckduckgo.com/html/?q={encodedQuery}";

            _httpClient.DefaultRequestHeaders.Clear();
            _httpClient.DefaultRequestHeaders.Add("User-Agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36");
            _httpClient.DefaultRequestHeaders.Add("Accept-Language", "pt-BR,pt;q=0.9,en;q=0.8");
            
            var response = await _httpClient.GetStringAsync(url);

            // Parse simples do HTML do DuckDuckGo
            results = ParseDuckDuckGoResults(response, maxResults);
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"  ⚠️  [WebSearch] Erro na busca: {ex.Message}. Retornando resultado simulado.");
            Console.ResetColor();

            // Fallback: retorna resultado indicando falha para o agente decidir o próximo passo
            results.Add(new SearchResult
            {
                Title = "Erro na busca web",
                Url = string.Empty,
                Snippet = $"Não foi possível realizar a busca para '{query}'. Erro: {ex.Message}"
            });
        }

        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"  ✅ [WebSearch] {results.Count} resultado(s) encontrado(s)");
        Console.ResetColor();

        return JsonSerializer.Serialize(results, new JsonSerializerOptions { WriteIndented = true });
    }

    /// <summary>
    /// Faz scraping de uma URL e retorna o texto limpo da página.
    /// Equivalente ao scrape_restaurantes_info() do notebook Python.
    /// </summary>
    [KernelFunction("fetch_page_content")]
    [Description("Acessa uma URL e extrai o conteúdo de texto da página web. Use para obter detalhes de um resultado de busca.")]
    public async Task<string> FetchPageContentAsync(
        [Description("A URL completa da página a ser acessada")] string url,
        [Description("Número máximo de caracteres a retornar (padrão: 4000)")] int maxChars = 4000)
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine($"  🌐 [WebScraper] Acessando: {url}");
        Console.ResetColor();

        try
        {
            _httpClient.DefaultRequestHeaders.Clear();
            _httpClient.DefaultRequestHeaders.Add("User-Agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36");
            _httpClient.DefaultRequestHeaders.Add("Accept-Language", "pt-BR,pt;q=0.9,en;q=0.8");

            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(15));
            var html = await _httpClient.GetStringAsync(url, cts.Token);

            var cleanText = ExtractTextFromHtml(html);

            // Limita o tamanho para não sobrecarregar o contexto do LLM
            if (cleanText.Length > maxChars)
                cleanText = cleanText[..maxChars] + "\n[... conteúdo truncado ...]";

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"  ✅ [WebScraper] {cleanText.Length} caracteres extraídos");
            Console.ResetColor();

            return $"URL: {url}\n\nCONTEÚDO:\n{cleanText}";
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"  ⚠️  [WebScraper] Falha ao acessar {url}: {ex.Message}");
            Console.ResetColor();
            return $"Não foi possível acessar a página {url}. Erro: {ex.Message}";
        }
    }

    // ─── helpers privados ───────────────────────────────────────────────────

    private static List<SearchResult> ParseDuckDuckGoResults(string html, int maxResults)
    {
        var results = new List<SearchResult>();

        // Usa HtmlAgilityPack via reflexão dinâmica para evitar dependência direta
        // Aqui fazemos parse manual simples baseado em padrões do DuckDuckGo HTML
        var doc = new HtmlAgilityPack.HtmlDocument();
        doc.LoadHtml(html);

        var resultNodes = doc.DocumentNode
            .SelectNodes("//div[@class='result results_links results_links_deep web-result']" +
                         " | //div[contains(@class,'result__body')]" +
                         " | //div[@class='result__body']");

        if (resultNodes == null)
        {
            // Fallback: tenta pegar todos os links com classe result__a
            var links = doc.DocumentNode.SelectNodes("//a[@class='result__a']");
            if (links != null)
            {
                foreach (var link in links.Take(maxResults))
                {
                    var href = link.GetAttributeValue("href", "");
                    // DuckDuckGo wraps URLs — decodifica
                    if (href.Contains("uddg="))
                    {
                        var encoded = System.Web.HttpUtility.ParseQueryString(
                            new Uri("https://duckduckgo.com" + href).Query)["uddg"];
                        if (!string.IsNullOrEmpty(encoded))
                            href = Uri.UnescapeDataString(encoded);
                    }
                    results.Add(new SearchResult
                    {
                        Title = HtmlAgilityPack.HtmlEntity.DeEntitize(link.InnerText.Trim()),
                        Url = href,
                        Snippet = string.Empty
                    });
                }
            }
            return results;
        }

        foreach (var node in resultNodes.Take(maxResults))
        {
            var titleNode = node.SelectSingleNode(".//a[@class='result__a']");
            var snippetNode = node.SelectSingleNode(".//a[@class='result__snippet']"
                + " | .//div[@class='result__snippet']");

            if (titleNode == null) continue;

            var href = titleNode.GetAttributeValue("href", "");
            if (href.Contains("uddg="))
            {
                try
                {
                    var encoded = System.Web.HttpUtility.ParseQueryString(
                        new Uri("https://duckduckgo.com" + href).Query)["uddg"];
                    if (!string.IsNullOrEmpty(encoded))
                        href = Uri.UnescapeDataString(encoded);
                }
                catch { /* mantém href original */ }
            }

            results.Add(new SearchResult
            {
                Title = HtmlAgilityPack.HtmlEntity.DeEntitize(titleNode.InnerText.Trim()),
                Url = href,
                Snippet = snippetNode != null
                    ? HtmlAgilityPack.HtmlEntity.DeEntitize(snippetNode.InnerText.Trim())
                    : string.Empty
            });
        }

        return results;
    }

    private static string ExtractTextFromHtml(string html)
    {
        var doc = new HtmlAgilityPack.HtmlDocument();
        doc.LoadHtml(html);

        // Remove scripts, styles, nav e outros elementos não-conteúdo
        var nodesToRemove = doc.DocumentNode
            .SelectNodes("//script|//style|//nav|//header|//footer|//iframe|//noscript");
        if (nodesToRemove != null)
            foreach (var node in nodesToRemove.ToList())
                node.Remove();

        // Extrai texto dos elementos de conteúdo principal
        var contentNodes = doc.DocumentNode
            .SelectNodes("//p|//h1|//h2|//h3|//h4|//li|//td|//th|//span[@class]|//div[@class]");

        if (contentNodes == null)
            return doc.DocumentNode.InnerText;

        var lines = new List<string>();
        foreach (var node in contentNodes)
        {
            var text = HtmlAgilityPack.HtmlEntity.DeEntitize(node.InnerText).Trim();
            if (text.Length > 20) // ignora textos muito curtos (menus, botões)
                lines.Add(text);
        }

        return string.Join("\n", lines.Distinct());
    }
}
