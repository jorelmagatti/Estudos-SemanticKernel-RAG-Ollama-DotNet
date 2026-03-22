using AgenteConsultaRagHitl;
using Microsoft.SemanticKernel;

Console.OutputEncoding = System.Text.Encoding.UTF8;
ConsoleRenderer.PrintBanner();

var config = new OllamaConfig
{
    BaseUrl =  "http://localhost:11434",
    ChatModel = "llama3.2"
};
var dbPath = "hitl_checkpoints.db";

Console.ForegroundColor = ConsoleColor.DarkGray;
Console.WriteLine($"  Ollama : {config.BaseUrl}  |  Modelo: {config.ChatModel}");
Console.WriteLine($"  SQLite : {Path.GetFullPath(dbPath)}");
Console.ResetColor();

var kernel = Kernel.CreateBuilder()
    .AddOllamaChatCompletion(config.ChatModel, new Uri(config.BaseUrl))
    .Build();

var search = new WebSearchPlugin();
using var repo = new CheckpointRepository(dbPath);

var today = DateTime.Now.ToString("dd/MM/yyyy");
var systemPrompt = $@"Você é um assistente de pesquisa inteligente. A data atual é {today}.
 
    Você tem acesso a uma ferramenta de busca na web.
 
    QUANDO precisar buscar informações atuais ou verificar fatos, responda APENAS com:
    {{""search"" ""sua query de busca aqui""}}
 
    QUANDO já tiver os resultados da busca ou souber a resposta, responda normalmente em português.
 
    Exemplos de quando usar a busca:
    - Perguntas sobre eventos atuais, notícias, clima
    - Perguntas sobre fatos que podem ter mudado (presidentes, preços, resultados)
    - Perguntas sobre distâncias, rotas ou dados geográficos atuais
 
    Ao buscar sobre hoje, inclua '{today}' na query.
    Nunca invente informações. Se não souber, busque primeiro.
    """;

using var agent = new HitlAgentService(kernel, repo, search, systemPrompt);

var currentThread = Guid.NewGuid().ToString()[..8];

Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine($"\n  ✅ Pronto! Thread atual: '{currentThread}'");
Console.ResetColor();
ConsoleRenderer.PrintHelp();

while (true)
{
    var hasPending = agent.HasPendingInterrupt(currentThread);

    // ── Se há HITL pendente, mostra o menu diretamente sem pedir input ────────
    // O input anterior que o usuário digitou ("aprovar", qualquer coisa) é ignorado.
    // O menu HITL já faz o ReadLine() internamente.
    if (hasPending)
    {
        // Lê o checkpoint para mostrar a ação real pendente
        var history = agent.GetHistory(currentThread);
        var pendingInfo = "Ação de busca pendente";

        // Tenta exibir detalhes do checkpoint salvo
        var lastTool = history.LastOrDefault(m => m.Role == MessageRole.Tool);
        var lastAsst = history.LastOrDefault(m => m.Role == MessageRole.Assistant);

        Console.ForegroundColor = ConsoleColor.Red;
        Console.Write($"\n[thread:{currentThread}] ⚠️  PAUSADO > pressione ENTER para ver o menu ");
        Console.ResetColor();
        Console.ReadLine(); // aguarda Enter antes de mostrar o menu

        var decision = ConsoleRenderer.PromptHitlDecision(pendingInfo);

        switch (decision)
        {
            case HitlDecision.Approve:
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine("\n  ▶ Retomando execução...");
                Console.ResetColor();
                try
                {
                    var resumeStream = agent.ResumeAsync(currentThread);
                    await ConsoleRenderer.RenderAsync(resumeStream, currentThread);
                }
                catch (Exception ex)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"\n  ❌ Erro ao retomar: {ex.Message}");
                    Console.ResetColor();
                }
                break;

            case HitlDecision.Inject:
                var injected = ConsoleRenderer.PromptInjectedResponse();
                if (!string.IsNullOrWhiteSpace(injected))
                    agent.InjectResponse(currentThread, injected);
                break;

            case HitlDecision.Cancel:
                agent.ClearThread(currentThread);
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine("\n  ✖ Ação cancelada. Thread reiniciado.");
                Console.ResetColor();
                break;
        }
        continue;
    }

    // ── Leitura normal do input ───────────────────────────────────────────────
    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.Write($"\n[thread:{currentThread}] ❓ ");
    Console.ResetColor();

    var input = Console.ReadLine()?.Trim();
    if (string.IsNullOrWhiteSpace(input)) continue;

    if (input.StartsWith("/thread ", StringComparison.OrdinalIgnoreCase))
    {
        currentThread = input[8..].Trim();
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"  ↪ Thread: '{currentThread}'");
        Console.ResetColor();
        continue;
    }
    if (input.Equals("/historico", StringComparison.OrdinalIgnoreCase))
    {
        ConsoleRenderer.PrintHistory(agent.GetHistory(currentThread), currentThread);
        continue;
    }
    if (input.Equals("/threads", StringComparison.OrdinalIgnoreCase))
    {
        var threads = agent.ListThreads();
        Console.ForegroundColor = ConsoleColor.DarkYellow;
        Console.WriteLine($"\n  Threads: {string.Join(", ", threads.DefaultIfEmpty("(nenhum)"))}");
        Console.ResetColor();
        continue;
    }
    if (input.Equals("/limpar", StringComparison.OrdinalIgnoreCase))
    {
        agent.ClearThread(currentThread);
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine($"  🗑  Thread '{currentThread}' apagado.");
        Console.ResetColor();
        continue;
    }
    if (input.Equals("/sair", StringComparison.OrdinalIgnoreCase)) break;

    // ── Execução normal ───────────────────────────────────────────────────────
    try
    {
        var stream = agent.RunAsync(currentThread, input);
        var interrupted = await ConsoleRenderer.RenderAsync(stream, currentThread);

        if (interrupted)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("""
 
              💡 O agente pausou. Na próxima iteração o menu HITL será exibido.
            """);
            Console.ResetColor();
        }
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