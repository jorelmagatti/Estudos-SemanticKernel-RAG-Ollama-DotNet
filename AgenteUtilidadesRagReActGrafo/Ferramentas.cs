namespace AgenteUtilidadesRagReActGrafo;

using Tavily;
public class BuscaWebTool
{
    public string Name => "buscar_na_web";
    private readonly string _apiKey;
    private readonly int _maxResultados;

    public BuscaWebTool(string apiKey, int maxResultados = 4)
    {
        _apiKey = apiKey;
        _maxResultados = maxResultados;
    }

    public async Task<string> BuscarAsync(string query)
    {
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine($"  🔍 Buscando: \"{query}\"");
        Console.ResetColor();

        using var client = new TavilyClient();
        var response = await client.SearchAsync(
            apiKey: _apiKey,
            query: query,
            maxResults: _maxResultados);

        if (response.Results == null || !response.Results.Any())
            return $"Nenhum resultado encontrado para: \"{query}\".";

        var sb = new System.Text.StringBuilder();
        foreach (var r in response.Results)
        {
            sb.AppendLine($"Título  : {r.Title}");
            sb.AppendLine($"Fonte   : {r.Url}");
            sb.AppendLine($"Conteúdo: {r.Content}");
            sb.AppendLine();
        }

        return sb.ToString();
    }
}