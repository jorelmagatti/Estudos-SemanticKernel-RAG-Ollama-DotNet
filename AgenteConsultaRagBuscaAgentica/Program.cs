using AgenteConsultaRagBuscaAgentica;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Embeddings;

// ════════════════════════════════════════════════════════════════════════════════
//  RAG com Busca Agêntica — Semantic Kernel + Ollama Local
//  Equivalente ao notebook Python (LangGraph + Tavily + BeautifulSoup + Gemini)
// ════════════════════════════════════════════════════════════════════════════════

Console.OutputEncoding = System.Text.Encoding.UTF8;
PrintBanner();

// ── Configuração ──────────────────────────────────────────────────────────────
var config = new OllamaConfig
{
    BaseUrl = "http://localhost:11434",
    ChatModel = "llama3.2",
    EmbeddingModel = "nomic-embed-text"
};

Console.ForegroundColor = ConsoleColor.DarkGray;
Console.WriteLine($"  Ollama URL   : {config.BaseUrl}");
Console.WriteLine($"  Chat Model   : {config.ChatModel}");
Console.WriteLine($"  Embed Model  : {config.EmbeddingModel}");
Console.ResetColor();

// ── Construção do Kernel ──────────────────────────────────────────────────────
var builder = Kernel.CreateBuilder();


// Chat completion via Ollama
builder.AddOllamaChatCompletion(
    modelId: config.ChatModel,
    endpoint: new Uri(config.BaseUrl));

// Text embedding via Ollama
builder.AddOllamaTextEmbeddingGeneration(
    modelId: config.EmbeddingModel,
    endpoint: new Uri(config.BaseUrl));


var kernel = builder.Build();

// ── Instanciação dos serviços ─────────────────────────────────────────────────
var httpClient = new HttpClient { Timeout = TimeSpan.FromSeconds(20) };
var searchPlugin = new WebSearchPlugin(httpClient);

kernel.Plugins.AddFromObject(searchPlugin, "WebSearch");

var embeddingService = kernel.GetRequiredService<ITextEmbeddingGenerationService>();
var ragMemory = new RagMemoryService(embeddingService, chunkSize: 800, chunkOverlap: 150);
var agentService = new AgenticSearchService(kernel, ragMemory, searchPlugin);

// ── Loop interativo ───────────────────────────────────────────────────────────
Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine("\n  ✅ Sistema pronto! Digite sua pergunta ou 'sair' para encerrar.");
Console.ResetColor();

while (true)
{
    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.Write("\n❓ Pergunta: ");
    Console.ResetColor();

    var input = Console.ReadLine()?.Trim();

    if (string.IsNullOrWhiteSpace(input)) continue;
    if (input.Equals("sair", StringComparison.OrdinalIgnoreCase) ||
        input.Equals("exit", StringComparison.OrdinalIgnoreCase)) break;

    try
    {
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        var state = await agentService.RunAsync(input);

        stopwatch.Stop();

        // ── Exibir resultado final ─────────────────────────────────────────
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine("\n╔══════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║                     RESPOSTA FINAL                          ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════════╝");
        Console.ResetColor();

        Console.ForegroundColor = ConsoleColor.White;
        Console.WriteLine(state.FinalAnswer ?? "Não foi possível gerar uma resposta.");
        Console.ResetColor();

        // ── Metadados da execução ─────────────────────────────────────────
        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine($"\n  ⏱  Tempo total: {stopwatch.Elapsed.TotalSeconds:F1}s");
        Console.WriteLine($"  🔄 Iterações  : {state.Iterations}");
        Console.WriteLine($"  📄 Chunks RAG : {state.RetrievedChunks.Count}");
        Console.WriteLine($"  🌐 URLs buscadas: {state.SearchResults.Count}");

        if (state.ThoughtProcess.Count > 0)
        {
            Console.WriteLine("\n  📋 Processo de raciocínio:");
            foreach (var thought in state.ThoughtProcess)
                Console.WriteLine($"     • {thought}");
        }

        // Fontes
        var sources = state.RetrievedChunks.Select(c => c.SourceUrl).Distinct().ToList();
        if (sources.Count > 0)
        {
            Console.WriteLine("\n  🔗 Fontes utilizadas:");
            for (int i = 0; i < sources.Count; i++)
                Console.WriteLine($"     [{i + 1}] {sources[i]}");
        }

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

// ─────────────────────────────────────────────────────────────────────────────

static void PrintBanner()
{
    Console.ForegroundColor = ConsoleColor.DarkYellow;
    Console.WriteLine("""
    
    ╔═══════════════════════════════════════════════════════════════╗
    ║      RAG com Busca Agêntica — Semantic Kernel + Ollama        ║
    ║                                                               ║
    ║  Estratégia: ReAct Loop (Reason → Act → Observe)             ║
    ║  Equivalente ao: LangGraph + Tavily + Gemini (Python)        ║
    ╚═══════════════════════════════════════════════════════════════╝
    """);
    Console.ResetColor();
}