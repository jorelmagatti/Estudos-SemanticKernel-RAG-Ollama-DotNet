using Tavily;

namespace EssayWriterBlazor.Plugins;

public class TavilySearchService : IDisposable
{
    private readonly TavilyClient _client;
    private readonly string _key;
    public TavilySearchService()
    {
        _client = new TavilyClient();
        _key = "";
    }

    public async Task<List<string>> SearchMultipleAsync(IEnumerable<string> queries, int maxEach = 2)
    {
        var results = new List<string>();
        foreach (var q in queries)
        {
            try
            {
                var r = await _client.SearchAsync(_key, query: q, maxResults: maxEach);
                results.AddRange((r.Results ?? [])
                    .Select(x => x.Content ?? "")
                    .Where(c => !string.IsNullOrWhiteSpace(c)));
            }
            catch { /* ignora falhas individuais */ }
        }
        return results;
    }

    public void Dispose() => _client.Dispose();
}
