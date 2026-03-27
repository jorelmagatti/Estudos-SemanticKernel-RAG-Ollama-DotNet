using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using MultAgentConsultaRagGrafoSemanticKernel;

Console.OutputEncoding = System.Text.Encoding.UTF8;
ConsoleRenderer.PrintBanner();

// ── Configuração ──────────────────────────────────────────────────────────────
var config = new OllamaConfig
{
    BaseUrl =  "http://localhost:11434",
    ChatModel = "llama3.2"
};

Console.ForegroundColor = ConsoleColor.DarkGray;
Console.WriteLine($"  Ollama : {config.BaseUrl}  |  Modelo: {config.ChatModel}");
Console.ResetColor();

// ── Kernel ────────────────────────────────────────────────────────────────────
// Registra HttpClient com timeout generoso para modelos Ollama locais lentos
var builder = Kernel.CreateBuilder();

builder.Services.AddHttpClient("ollama", c =>
{
    c.BaseAddress = new Uri(config.BaseUrl);
    c.Timeout = TimeSpan.FromMinutes(10);
});

builder.AddOllamaChatCompletion(config.ChatModel, new Uri(config.BaseUrl));

var kernel = builder.Build();

// ── Serviços ──────────────────────────────────────────────────────────────────
var tavily = new TavilySearchService();

// ── Orquestrador multi-agente ─────────────────────────────────────────────────
using var orchestrator = new MultiAgentOrchestrator(kernel, tavily);

// ── Estado da sessão ──────────────────────────────────────────────────────────
var maxRevisions = 2;

Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine($"\n  ✅ Pronto! Revisões: {maxRevisions}");
Console.ResetColor();
ConsoleRenderer.PrintHelp();

// ── Loop interativo ───────────────────────────────────────────────────────────
while (true)
{
    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.Write("\n📝 Tema para a redação: ");
    Console.ResetColor();

    var input = Console.ReadLine()?.Trim();
    if (string.IsNullOrWhiteSpace(input)) continue;

    // ── Comandos especiais ────────────────────────────────────────────────────
    if (input.StartsWith("/revisoes ", StringComparison.OrdinalIgnoreCase))
    {
        if (int.TryParse(input[10..].Trim(), out var n) && n > 0)
        {
            maxRevisions = n;
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"  ✅ Revisões definidas: {maxRevisions}");
            Console.ResetColor();
        }
        continue;
    }

    if (input.Equals("/sair", StringComparison.OrdinalIgnoreCase)) break;

    // ── Executa o pipeline multi-agente ───────────────────────────────────────
    // Equivalente ao:
    //   for s in graph.stream({'task': input, 'max_revisions': 2, ...}, thread):
    //       print(s)
    Console.ForegroundColor = ConsoleColor.DarkGray;
    Console.WriteLine($"\n  Iniciando pipeline com {maxRevisions} revisão(ões)...");
    Console.ResetColor();

    try
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var stream = orchestrator.RunAsync(input, maxRevisions);
        await ConsoleRenderer.RenderPipelineAsync(stream);
        sw.Stop();

        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine($"\n  ⏱  Tempo total: {sw.Elapsed.TotalSeconds:F1}s");
        Console.ResetColor();
    }
    catch (Exception ex)
    {
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine($"\n  ❌ Erro: {ex.Message}");
        Console.WriteLine("  Verifique se o Ollama está rodando: ollama serve");
        Console.ResetColor();
    }
}

Console.ForegroundColor = ConsoleColor.DarkGray;
Console.WriteLine("\n  Encerrando... Até mais! 👋");
Console.ResetColor();