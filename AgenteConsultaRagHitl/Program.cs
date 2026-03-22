using AgenteConsultaRagHitl;
using Microsoft.SemanticKernel;

Console.OutputEncoding = System.Text.Encoding.UTF8;
ConsoleRenderer.PrintBanner();

// ── Configuração ──────────────────────────────────────────────────────────────
var config = new OllamaConfig
{
    BaseUrl = "http://localhost:11434",
    ChatModel =  "llama3.2"
};

var dbPath = Environment.GetEnvironmentVariable("DB_PATH") ?? "hitl_checkpoints.db";

Console.ForegroundColor = ConsoleColor.DarkGray;
Console.WriteLine($"  Ollama : {config.BaseUrl}  |  Modelo: {config.ChatModel}");
Console.WriteLine($"  SQLite : {Path.GetFullPath(dbPath)}");
Console.ResetColor();

// ── Kernel ────────────────────────────────────────────────────────────────────
var kernel = Kernel.CreateBuilder()
    .AddOllamaChatCompletion(config.ChatModel, new Uri(config.BaseUrl))
    .Build();

// ── Serviços ──────────────────────────────────────────────────────────────────
var http = new HttpClient { Timeout = TimeSpan.FromSeconds(20) };
var search = new WebSearchPlugin(http);

using var repo = new CheckpointRepository(dbPath);

// System prompt — equivalente ao prompt do notebook Python com a data atual
var today = DateTime.Now.ToString("dd/MM/yyyy");
var systemPrompt = $"""
    Você é um assistente de pesquisa inteligente e altamente atualizado.
    Sua principal prioridade é encontrar informações RECENTES e em TEMPO REAL.
    A data atual é {today}.
    Ao buscar sobre o tempo ou eventos que se referem a "hoje" ou "agora",
    inclua a data atual '{today}' na sua consulta de busca.
    Use a ferramenta search_web quando precisar de dados atuais.
    Você pode fazer múltiplas buscas em sequência.
    Responda sempre em português brasileiro.
    """;

using var agent = new HitlAgentService(kernel, repo, search, systemPrompt);

// ── Estado do console ─────────────────────────────────────────────────────────
var currentThread = Guid.NewGuid().ToString()[..8]; // UUID curto como no notebook

Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine($"\n  ✅ Pronto! Thread atual: '{currentThread}'");
Console.ResetColor();
ConsoleRenderer.PrintHelp();

// ── Loop interativo principal ─────────────────────────────────────────────────
while (true)
{
    // Indica se há interrupção HITL pendente neste thread
    var hasPending = agent.HasPendingInterrupt(currentThread);

    Console.ForegroundColor = hasPending ? ConsoleColor.Red : ConsoleColor.Cyan;
    Console.Write(hasPending
        ? $"\n[thread:{currentThread}] ⚠️  PAUSADO — aprovação pendente > "
        : $"\n[thread:{currentThread}] ❓ ");
    Console.ResetColor();

    var input = Console.ReadLine()?.Trim();
    if (string.IsNullOrWhiteSpace(input)) continue;

    // ── Comandos especiais ────────────────────────────────────────────────────
    if (input.StartsWith("/thread ", StringComparison.OrdinalIgnoreCase))
    {
        currentThread = input[8..].Trim();
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"  ↪ Thread trocado para '{currentThread}'");
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

    // ── Se há interrupção HITL pendente, o input inicial é ignorado e ─────────
    //    exibimos o menu de decisão humana
    if (hasPending)
    {
        var checkpoint = agent.GetHistory(currentThread).LastOrDefault();
        var pendingInfo = "Ação pendente de aprovação";

        // ── Menu HITL ─────────────────────────────────────────────────────────
        // Equivalente às células 10 e 12 do notebook:
        //   user_input = input("Você deseja executar esta ação? (sim/não)")
        var decision = ConsoleRenderer.PromptHitlDecision(pendingInfo);

        switch (decision)
        {
            // ── Aprovação: retoma o grafo ─────────────────────────────────────
            // Equivalente a:
            //   for event in abot.graph.stream(None, thread_config): ...
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

            // ── Injeção: equivalente ao graph.update_state() do notebook ──────
            case HitlDecision.Inject:
                var injected = ConsoleRenderer.PromptInjectedResponse();
                if (!string.IsNullOrWhiteSpace(injected))
                    agent.InjectResponse(currentThread, injected);
                break;

            // ── Cancelamento ──────────────────────────────────────────────────
            case HitlDecision.Cancel:
                agent.ClearThread(currentThread);
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine("\n  ✖ Ação cancelada. Thread reiniciado.");
                Console.ResetColor();
                break;
        }
        continue;
    }

    // ── Execução normal: envia mensagem ao agente ─────────────────────────────
    try
    {
        var stream = agent.RunAsync(currentThread, input);
        var interrupted = await ConsoleRenderer.RenderAsync(stream, currentThread);

        if (interrupted)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("""
 
              💡 O agente está pausado aguardando sua decisão.
                 Na próxima entrada, escolha aprovar, cancelar ou injetar resposta.
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
