using AgentConsultaRagPersistencia;
using Microsoft.SemanticKernel;

Console.OutputEncoding = System.Text.Encoding.UTF8;
ConsoleStreamRenderer.PrintBanner();

// ── Configuração ──────────────────────────────────────────────────────────────
var config = new OllamaConfig
{
    BaseUrl = "http://localhost:11434",
    ChatModel = "llama3.2",
    EmbedModel = "nomic-embed-text"
};

var dbPath = Environment.GetEnvironmentVariable("DB_PATH") ?? "conversations.db";

Console.ForegroundColor = ConsoleColor.DarkGray;
Console.WriteLine($"  Ollama : {config.BaseUrl}  |  Modelo: {config.ChatModel}");
Console.WriteLine($"  SQLite : {Path.GetFullPath(dbPath)}");
Console.ResetColor();

// ── Kernel ────────────────────────────────────────────────────────────────────
var kernel = Kernel.CreateBuilder()
    .AddOllamaChatCompletion(config.ChatModel, new Uri(config.BaseUrl))
    .Build();

// ── Plugins ───────────────────────────────────────────────────────────────────
var http = new HttpClient { Timeout = TimeSpan.FromSeconds(20) };
kernel.Plugins.AddFromObject(new WebSearchPlugin(http), "WebSearch");

// ── Persistência SQLite ───────────────────────────────────────────────────────
// Equivalente ao:
//   conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
//   memory = SqliteSaver(conn)
using var repo = new ConversationRepository(dbPath);

// ── System prompt (equivalente ao prompt_system do notebook) ──────────────────
var systemPrompt = """
    Você é um assistente de pesquisa inteligente com acesso a ferramentas de busca na web.
    Use a ferramenta search_web para buscar informações atuais quando necessário.
    Você pode fazer múltiplas buscas em sequência para responder com precisão.
    Quando comparar informações entre turnos anteriores, use o histórico da conversa.
    Responda sempre em português brasileiro.
    """;

// ── Serviço do agente ─────────────────────────────────────────────────────────
var agent = new AgentService(kernel, repo, systemPrompt);

// ── Estado do console ─────────────────────────────────────────────────────────
// thread_id equivalente ao {"configurable": {"thread_id": "1"}} do LangGraph
var currentThread = "1";

Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine($"\n  ✅ Pronto! Thread atual: '{currentThread}'");
Console.ResetColor();

ConsoleStreamRenderer.PrintHelp();

// ── Loop interativo ───────────────────────────────────────────────────────────
while (true)
{
    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.Write($"\n[thread:{currentThread}] ❓ ");
    Console.ResetColor();

    var input = Console.ReadLine()?.Trim();
    if (string.IsNullOrWhiteSpace(input)) continue;

    // ── Comandos especiais ────────────────────────────────────────────────────
    if (input.StartsWith("/thread ", StringComparison.OrdinalIgnoreCase))
    {
        // Equivalente a trocar o thread_id:
        //   thread = {"configurable": {"thread_id": "2"}}
        currentThread = input[8..].Trim();
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"  ↪ Thread trocado para '{currentThread}'");
        Console.ResetColor();
        continue;
    }

    if (input.Equals("/historico", StringComparison.OrdinalIgnoreCase))
    {
        ConsoleStreamRenderer.PrintHistory(agent.GetHistory(currentThread), currentThread);
        continue;
    }

    if (input.Equals("/threads", StringComparison.OrdinalIgnoreCase))
    {
        ConsoleStreamRenderer.PrintThreadList(agent.ListThreads());
        continue;
    }

    if (input.Equals("/limpar", StringComparison.OrdinalIgnoreCase))
    {
        agent.ClearThread(currentThread);
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine($"  🗑  Histórico do thread '{currentThread}' apagado.");
        Console.ResetColor();
        continue;
    }

    if (input.Equals("/sair", StringComparison.OrdinalIgnoreCase) ||
        input.Equals("/exit", StringComparison.OrdinalIgnoreCase))
        break;

    // ── Executa o agente com streaming ────────────────────────────────────────
    // Equivalente ao:
    //   for event in abot.graph.stream({messages}, thread):
    //       for k, v in event.items():
    //           print(f"{k}: {v['messages']}")
    try
    {
        var stream = agent.RunStreamingAsync(currentThread, input);
        await ConsoleStreamRenderer.RenderAsync(stream, currentThread);
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