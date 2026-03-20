
namespace AgenteConsultaRagBuscaAgentica;

using System.Text.Json;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;


/// <summary>
/// Serviço de Busca Agêntica — coração do projeto.
/// Implementa o loop ReAct (Reason → Act → Observe) equivalente ao grafo LangGraph do notebook Python:
///
///   [query] → [planejar buscas] → [buscar web] → [fazer scraping] → [indexar RAG]
///          → [recuperar chunks] → [responder] → [avaliar suficiência] → [iterar ou finalizar]
///
/// O agente decide autonomamente quantas buscas fazer e quais URLs acessar.
/// </summary>
public class AgenticSearchService
{
    private readonly Kernel _kernel;
    private readonly RagMemoryService _ragMemory;
    private readonly IChatCompletionService _chatService;
    private readonly WebSearchPlugin _searchPlugin;
    private const int MaxIterations = 5;

    public AgenticSearchService(
        Kernel kernel,
        RagMemoryService ragMemory,
        WebSearchPlugin searchPlugin)
    {
        _kernel = kernel;
        _ragMemory = ragMemory;
        _searchPlugin = searchPlugin;
        _chatService = kernel.GetRequiredService<IChatCompletionService>();
    }

    /// <summary>
    /// Ponto de entrada principal. Recebe a pergunta do usuário e executa o ciclo agêntico completo.
    /// </summary>
    public async Task<AgentState> RunAsync(string userQuery)
    {
        var state = new AgentState { OriginalQuery = userQuery };

        PrintHeader("INICIANDO BUSCA AGÊNTICA");
        Console.ForegroundColor = ConsoleColor.White;
        Console.WriteLine($"  Pergunta: {userQuery}");
        Console.ResetColor();

        // ── Fase 1: Planejar queries de busca ─────────────────────────────────
        await PlanSearchQueriesAsync(state);

        // ── Fase 2: Loop ReAct (buscar → recuperar → avaliar) ─────────────────
        while (!state.IsComplete && state.Iterations < MaxIterations)
        {
            state.Iterations++;
            PrintHeader($"ITERAÇÃO {state.Iterations}/{MaxIterations}");

            // Act: Buscar na web
            await ExecuteSearchesAsync(state);

            // Act: Fazer scraping das URLs encontradas
            await ScrapeAndIndexAsync(state);

            // Observe: Recuperar chunks relevantes
            await RetrieveRelevantChunksAsync(state);

            // Reason: Avaliar se temos contexto suficiente para responder
            var hasSufficientContext = await EvaluateContextSufficiencyAsync(state);

            if (hasSufficientContext || state.Iterations >= MaxIterations)
                state.IsComplete = true;
            else
                await RefineSearchQueriesAsync(state);
            
        }

        // ── Fase 3: Gerar resposta final com RAG ──────────────────────────────
        await GenerateFinalAnswerAsync(state);

        return state;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Fase 1: Planejamento
    // ─────────────────────────────────────────────────────────────────────────

    private async Task PlanSearchQueriesAsync(AgentState state)
    {
        PrintStep("PLANEJANDO QUERIES DE BUSCA");

        var prompt = $"""
            Você é um agente de pesquisa inteligente. Dado o seguinte pedido do usuário,
            gere de 2 a 3 queries de busca web otimizadas para encontrar as informações necessárias.
 
            Pedido: {state.OriginalQuery}
 
            Responda APENAS com as queries, uma por linha, sem numeração ou explicações.
            """;

        var result = await _chatService.GetChatMessageContentAsync(
            new ChatHistory { new ChatMessageContent(AuthorRole.User, prompt) },
            kernel: _kernel);

        var queries = result.Content!
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(q => q.Trim().TrimStart('-', '*', '•').Trim())
            .Where(q => !string.IsNullOrWhiteSpace(q))
            .Take(3)
            .ToList();

        state.SearchQueries = queries;
        state.ThoughtProcess.Add($"[Planejamento] Queries geradas: {string.Join(" | ", queries)}");

        Console.ForegroundColor = ConsoleColor.Magenta;
        foreach (var q in queries)
            Console.WriteLine($"  → {q}");
        Console.ResetColor();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Fase 2: Loop ReAct
    // ─────────────────────────────────────────────────────────────────────────

    private async Task ExecuteSearchesAsync(AgentState state)
    {
        PrintStep("BUSCANDO NA WEB");

        foreach (var query in state.SearchQueries)
        {
            var json = await _searchPlugin.SearchWebAsync(query, maxResults: 4);
            var results = JsonSerializer.Deserialize<List<SearchResult>>(json)
                          ?? new List<SearchResult>();

            // Evita duplicatas por URL
            foreach (var r in results)
                if (!string.IsNullOrEmpty(r.Url) && !state.SearchResults.Any(x => x.Url == r.Url))
                    state.SearchResults.Add(r);
        }

        state.ThoughtProcess.Add($"[Busca] {state.SearchResults.Count} URL(s) únicas encontradas");
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"  Total acumulado: {state.SearchResults.Count} URL(s) únicas");
        Console.ResetColor();
    }

    private async Task ScrapeAndIndexAsync(AgentState state)
    {
        PrintStep("FAZENDO SCRAPING E INDEXANDO NO RAG");

        // Seleciona as URLs mais relevantes (evita re-indexar)
        var urlsJaIndexadas = _ragMemory.ChunkCount > 0
            ? state.SearchResults
                .Where(r => _ragMemory.ChunkCount > 0) // simplificado
                .Select(r => r.Url)
                .ToHashSet()
            : new HashSet<string>();

        // Pega as primeiras 3 URLs não nulas para scraping
        var urlsParaScraping = state.SearchResults
            .Where(r => !string.IsNullOrEmpty(r.Url) && r.Url.StartsWith("http"))
            .Take(3)
            .ToList();

        foreach (var result in urlsParaScraping)
        {
            var content = await _searchPlugin.FetchPageContentAsync(result.Url, maxChars: 5000);
            await _ragMemory.IngestAsync(content, result.Url);
        }
    }

    private async Task RetrieveRelevantChunksAsync(AgentState state)
    {
        PrintStep("RECUPERANDO CHUNKS RELEVANTES (RAG)");

        var chunks = await _ragMemory.RetrieveAsync(state.OriginalQuery, topK: 6);
        state.RetrievedChunks = chunks;

        state.ThoughtProcess.Add($"[RAG] {chunks.Count} chunk(s) recuperados");
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"  {chunks.Count} chunk(s) mais relevantes recuperados");
        Console.ResetColor();
    }

    private async Task<bool> EvaluateContextSufficiencyAsync(AgentState state)
    {
        PrintStep("AVALIANDO SUFICIÊNCIA DO CONTEXTO");

        if (state.RetrievedChunks.Count == 0)
        {
            Console.WriteLine("  ❌ Sem contexto. Nova iteração necessária.");
            return false;
        }

        var context = BuildContextString(state.RetrievedChunks);
        var prompt = $"""
            Você é um avaliador de qualidade de contexto para RAG.
            
            Pergunta original: {state.OriginalQuery}
            
            Contexto recuperado:
            {context}
            
            O contexto acima é suficiente para responder à pergunta com precisão?
            Responda APENAS: SIM ou NAO
            """;

        var result = await _chatService.GetChatMessageContentAsync(
            new ChatHistory { new ChatMessageContent(AuthorRole.User, prompt) },
            kernel: _kernel);

        var isSufficient = result.Content!.ToUpper().Contains("SIM");

        Console.ForegroundColor = isSufficient ? ConsoleColor.Green : ConsoleColor.Yellow;
        Console.WriteLine($"  Contexto suficiente? {(isSufficient ? "✅ SIM" : "⚠️  NÃO — refinando busca")}");
        Console.ResetColor();

        return isSufficient;
    }

    private async Task RefineSearchQueriesAsync(AgentState state)
    {
        PrintStep("REFINANDO QUERIES DE BUSCA");

        var context = BuildContextString(state.RetrievedChunks);
        var prompt = $"""
            Você é um agente de pesquisa. As buscas anteriores não trouxeram contexto suficiente.
            
            Pergunta original: {state.OriginalQuery}
            Contexto parcial obtido: {context[..Math.Min(500, context.Length)]}
            
            Gere 2 novas queries de busca mais específicas para complementar as informações faltantes.
            Responda APENAS com as queries, uma por linha.
            """;

        var result = await _chatService.GetChatMessageContentAsync(
            new ChatHistory { new ChatMessageContent(AuthorRole.User, prompt) },
            kernel: _kernel);

        var newQueries = result.Content!
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(q => q.Trim().TrimStart('-', '*', '•').Trim())
            .Where(q => !string.IsNullOrWhiteSpace(q))
            .Take(2)
            .ToList();

        state.SearchQueries = newQueries;
        state.ThoughtProcess.Add($"[Refinamento] Novas queries: {string.Join(" | ", newQueries)}");

        Console.ForegroundColor = ConsoleColor.Magenta;
        foreach (var q in newQueries)
            Console.WriteLine($"  → {q}");
        Console.ResetColor();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Fase 3: Geração da resposta final
    // ─────────────────────────────────────────────────────────────────────────

    private async Task GenerateFinalAnswerAsync(AgentState state)
    {
        PrintHeader("GERANDO RESPOSTA FINAL COM RAG");

        var context = state.RetrievedChunks.Count > 0
            ? BuildContextString(state.RetrievedChunks)
            : "Nenhum contexto relevante foi encontrado nas buscas realizadas.";

        var sources = state.RetrievedChunks
            .Select(c => c.SourceUrl)
            .Distinct()
            .Take(5)
            .ToList();

        var systemPrompt = """
            Você é um assistente especialista em RAG (Retrieval-Augmented Generation).
            Responda à pergunta do usuário BASEANDO-SE EXCLUSIVAMENTE no contexto fornecido.
            Se o contexto não contiver informação suficiente, diga claramente.
            Seja objetivo, estruturado e cite as fontes quando relevante.
            Responda sempre em português brasileiro.
            """;

        var userPrompt = $"""
            PERGUNTA: {state.OriginalQuery}
            
            CONTEXTO RECUPERADO:
            {context}
            
            FONTES CONSULTADAS:
            {string.Join("\n", sources.Select((s, i) => $"[{i + 1}] {s}"))}
            
            Por favor, responda à pergunta com base no contexto acima.
            """;

        var history = new ChatHistory();
        history.AddSystemMessage(systemPrompt);
        history.AddUserMessage(userPrompt);

        var response = await _chatService.GetChatMessageContentAsync(history, kernel: _kernel);
        state.FinalAnswer = response.Content;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers de UI
    // ─────────────────────────────────────────────────────────────────────────

    private static string BuildContextString(List<DocumentChunk> chunks)
    {
        return string.Join("\n\n---\n\n",
            chunks.Select((c, i) =>
                $"[Fonte {i + 1}: {c.SourceUrl[..Math.Min(60, c.SourceUrl.Length)]}]\n{c.Content}"));
    }

    private static void PrintHeader(string title)
    {
        Console.ForegroundColor = ConsoleColor.DarkYellow;
        Console.WriteLine($"\n{'═'.ToString().PadRight(1)}{'═'.ToString().PadLeft(60, '═')}");
        Console.WriteLine($"  {title}");
        Console.WriteLine($"{'═'.ToString().PadRight(60, '═')}");
        Console.ResetColor();
    }

    private static void PrintStep(string step)
    {
        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine($"\n  ▶ {step}");
        Console.ResetColor();
    }
}