using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.Ollama;
using System.Text;
using System.Text.Json;

namespace AgenteConsultaRagHitl;

public class HitlAgentService : IDisposable
{
    private readonly Kernel _kernel;
    private readonly IChatCompletionService _chat;
    private readonly CheckpointRepository _repo;
    private readonly WebSearchPlugin _search;
    private readonly string _systemPrompt;
    private readonly CompiledGraph<AgentState> _graph;

    public bool DiagnosticMode { get; set; } = true;

    public HitlAgentService(
        Kernel kernel,
        CheckpointRepository repo,
        WebSearchPlugin search,
        string systemPrompt = "")
    {
        _kernel = kernel;
        _chat = kernel.GetRequiredService<IChatCompletionService>();
        _repo = repo;
        _search = search;
        _systemPrompt = systemPrompt;

        if (!_kernel.Plugins.Contains("WebSearch"))
            _kernel.Plugins.AddFromObject(search, "WebSearch");

        _graph = new StateGraph<AgentState>()
            .AddStreamingNode("llm", LlmNodeAsync)
            .AddNode("action", ActionNodeAsync)
            .AddConditionalEdge(
                from: "llm",
                router: ExistsAction,
                routeMap: new Dictionary<string, string>
                {
                    ["tool"] = "action",
                    ["end"] = GraphConstants.END
                })
            .AddEdge("action", "llm")
            .SetEntryPoint("llm")
            .Compile(interruptBefore: ["action"]);
    }

    // ════════════════════════════════════════════════════════════════════════
    //  API pública
    // ════════════════════════════════════════════════════════════════════════

    public async IAsyncEnumerable<GraphEvent> RunAsync(string threadId, string userMessage)
    {
        _repo.SaveMessage(threadId, MessageRole.User, userMessage);
        var state = BuildState(threadId);

        await foreach (var nodeEvent in _graph.StreamAsync(state))
            await foreach (var evt in ProcessNodeEvent(nodeEvent, threadId))
                yield return evt;
    }

    /// <summary>
    /// Resume após aprovação HITL.
    /// Reescrito para processar diretamente sem delegar para ProcessNodeEvent,
    /// evitando o problema de yield break interrompendo a cadeia action→llm.
    /// </summary>
    public async IAsyncEnumerable<GraphEvent> ResumeAsync(string threadId)
    {
        var checkpoint = _repo.GetInterruptCheckpoint(threadId);
        if (checkpoint == null)
        {
            yield return new GraphEvent { Type = GraphEventType.GraphFinished };
            yield break;
        }

        _repo.ClearCheckpoint(threadId);

        var state = BuildState(threadId);
        state.PendingToolCalls.Add(new ToolCallRequest
        {
            Name = checkpoint.Value.ToolName,
            Query = checkpoint.Value.ToolQuery,
            Id = checkpoint.Value.ToolId
        });

        // ── Processa diretamente todos os nós após a retomada ─────────────────
        // Não usa ProcessNodeEvent para evitar yield break cortando a cadeia
        await foreach (var nodeEvent in _graph.ResumeAsync(state, "action"))
        {
            if (nodeEvent.NodeName == "action")
            {
                // Nó action: executa a ferramenta (já foi feito pelo grafo)
                // Só emite o evento de resultado
                yield return new GraphEvent
                {
                    Type = GraphEventType.ToolResult,
                    Content = "Busca concluída.",
                    NodeName = "action"
                };
                // Continua para o próximo NodeEvent (llm) sem interromper
                continue;
            }

            if (nodeEvent.NodeName == "llm" && nodeEvent.StreamTokens != null)
            {
                // Nó llm: streaming da resposta final
                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.WriteLine("\n▶ [LLM]");
                Console.ResetColor();

                // Log do histórico enviado ao LLM para diagnóstico
                Console.ForegroundColor = ConsoleColor.DarkGray;
                Console.WriteLine($"  [DIAG-RESUME] Mensagens no histórico: {nodeEvent.State.Messages.Count}");
                foreach (var msg in nodeEvent.State.Messages)
                {
                    var preview = (msg.Content ?? "").Length > 80
                        ? (msg.Content ?? "")[..80] + "..." : (msg.Content ?? "");
                    Console.WriteLine($"  [DIAG-RESUME]   [{msg.Role}] {preview}");
                }
                Console.ResetColor();

                var fullText = new StringBuilder();

                await foreach (var token in nodeEvent.StreamTokens)
                {
                    fullText.Append(token);
                    yield return new GraphEvent
                    {
                        Type = GraphEventType.LlmToken,
                        Content = token,
                        NodeName = "llm"
                    };
                }

                var text = fullText.ToString();
                var hasTc = nodeEvent.State.PendingToolCalls.Count > 0;

                if (!hasTc && !string.IsNullOrWhiteSpace(text))
                {
                    // Persiste a resposta final
                    _repo.SaveMessage(threadId, MessageRole.Assistant, text);

                    yield return new GraphEvent
                    {
                        Type = GraphEventType.GraphFinished,
                        NodeName = "llm"
                    };
                }
                else if (hasTc)
                {
                    // O LLM quer fazer outra busca — nova interrupção HITL
                    var tc = nodeEvent.State.PendingToolCalls[0];
                    _repo.SaveInterruptCheckpoint(threadId, "action",
                        tc.Name, tc.Query, tc.Id);

                    yield return new GraphEvent
                    {
                        Type = GraphEventType.HumanInterruptRequired,
                        Content = $"Ferramenta: {tc.Name} | Query: \"{tc.Query}\"",
                        ToolCall = tc,
                        NodeName = "action"
                    };
                }
            }
        }
    }

    public void InjectResponse(string threadId, string content)
    {
        _repo.ClearCheckpoint(threadId);
        _repo.SaveMessage(threadId, MessageRole.Assistant, $"[Injetado]\n{content}");
        Console.ForegroundColor = ConsoleColor.Magenta;
        Console.WriteLine($"\n  ✏️  Resposta injetada: {content}");
        Console.ResetColor();
    }

    public List<ChatMessage> GetHistory(string tid) => _repo.GetMessages(tid);
    public List<string> ListThreads() => _repo.ListThreads();
    public void ClearThread(string tid) => _repo.ClearThread(tid);
    public bool HasPendingInterrupt(string tid) => _repo.GetInterruptCheckpoint(tid) != null;
    public void Dispose() => _search.Dispose();

    // ════════════════════════════════════════════════════════════════════════
    //  Nós do grafo
    // ════════════════════════════════════════════════════════════════════════

    private async IAsyncEnumerable<(AgentState, string?)> LlmNodeAsync(AgentState state)
    {
        if (state.Messages.Count == 0 || state.Messages[0].Role != AuthorRole.System)
            if (!string.IsNullOrWhiteSpace(_systemPrompt))
                state.Messages.Insert(0,
                    new ChatMessageContent(AuthorRole.System, _systemPrompt));

#pragma warning disable SKEXP0070
        var settings = new OllamaPromptExecutionSettings { Temperature = 0.3f };
#pragma warning restore SKEXP0070

        var fullText = new StringBuilder();
        var functionCallAccum = new StringBuilder();

        await foreach (var chunk in _chat.GetStreamingChatMessageContentsAsync(
            state.Messages, executionSettings: settings, kernel: _kernel))
        {
            if (chunk.Metadata != null)
                foreach (var kv in chunk.Metadata)
                    if (kv.Key.Contains("tool", StringComparison.OrdinalIgnoreCase) ||
                        kv.Key.Contains("function", StringComparison.OrdinalIgnoreCase))
                        functionCallAccum.Append(kv.Value?.ToString());

            if (string.IsNullOrEmpty(chunk.Content)) continue;
            fullText.Append(chunk.Content);
            yield return (state, chunk.Content);
        }

        var responseText = fullText.ToString().Trim();

        if (DiagnosticMode)
        {
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine($"\n  [DIAG] Texto bruto ({responseText.Length} chars):");
            Console.WriteLine($"  [DIAG] >>>{responseText[..Math.Min(300, responseText.Length)]}<<<");
            Console.ResetColor();
        }

        var searchQuery = DetectSearchIntent(responseText, functionCallAccum.ToString());

        if (DiagnosticMode)
        {
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine($"  [DIAG] Query detectada: {searchQuery ?? "(nenhuma)"}");
            Console.ResetColor();
        }

        state.PendingToolCalls = searchQuery != null
            ? new List<ToolCallRequest>
              { new() { Id = Guid.NewGuid().ToString(), Name = "search_web", Query = searchQuery } }
            : new List<ToolCallRequest>();

        if (state.PendingToolCalls.Count == 0)
        {
            _repo.SaveMessage(state.ThreadId, MessageRole.Assistant, responseText);
            state.Messages.AddAssistantMessage(responseText);
        }
        else
        {
            state.Messages.AddAssistantMessage(responseText);
        }

        yield return (state, null);
    }

    private static string ExistsAction(AgentState state) =>
        state.PendingToolCalls.Count > 0 ? "tool" : "end";

    private async Task<AgentState> ActionNodeAsync(AgentState state)
    {
        foreach (var tc in state.PendingToolCalls)
        {
            Console.ForegroundColor = ConsoleColor.DarkYellow;
            Console.WriteLine($"\n  ⚙️  Buscando no Tavily: \"{tc.Query}\"");
            Console.ResetColor();

            var result = await _search.SearchDirectAsync(tc.Query);

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"  ✅ Resultado recebido ({result.Length} chars)");
            Console.ResetColor();

            _repo.SaveMessage(state.ThreadId, MessageRole.Tool, result, tc.Name);
            state.Messages.AddUserMessage($"[Resultado da busca: {tc.Query}]\n{result}");
        }

        state.PendingToolCalls.Clear();
        return state;
    }

    // ════════════════════════════════════════════════════════════════════════
    //  ProcessNodeEvent — usado apenas por RunAsync (primeira chamada)
    // ════════════════════════════════════════════════════════════════════════

    private async IAsyncEnumerable<GraphEvent> ProcessNodeEvent(
        NodeEvent<AgentState> nodeEvent, string threadId)
    {
        // Interrupção HITL antes do nó action
        if (nodeEvent.IsInterrupted)
        {
            var pending = nodeEvent.State.PendingToolCalls;
            if (pending.Count > 0)
            {
                var tc = pending[0];
                _repo.SaveInterruptCheckpoint(threadId, nodeEvent.NodeName,
                    tc.Name, tc.Query, tc.Id);
                yield return new GraphEvent
                {
                    Type = GraphEventType.HumanInterruptRequired,
                    NodeName = nodeEvent.NodeName,
                    Content = $"Ferramenta: {tc.Name} | Query: \"{tc.Query}\"",
                    ToolCall = tc
                };
            }
            yield break;
        }

        if (nodeEvent.NodeName == "llm" && nodeEvent.StreamTokens != null)
        {
            var fullText = new StringBuilder();
            await foreach (var token in nodeEvent.StreamTokens)
            {
                fullText.Append(token);
                yield return new GraphEvent
                {
                    Type = GraphEventType.LlmToken,
                    Content = token,
                    NodeName = "llm"
                };
            }

            var hasTc = nodeEvent.State.PendingToolCalls.Count > 0;
            if (hasTc)
            {
                var tc = nodeEvent.State.PendingToolCalls[0];
                yield return new GraphEvent
                {
                    Type = GraphEventType.LlmToolDecision,
                    Content = $"Quer buscar: \"{tc.Query}\"",
                    ToolCall = tc,
                    NodeName = "llm"
                };
            }
            else
            {
                yield return new GraphEvent
                {
                    Type = GraphEventType.LlmDirectResponse,
                    Content = fullText.ToString(),
                    NodeName = "llm"
                };
                yield return new GraphEvent
                {
                    Type = GraphEventType.GraphFinished,
                    NodeName = "llm"
                };
            }
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  Detecção de intenção de busca
    // ════════════════════════════════════════════════════════════════════════

    private string? DetectSearchIntent(string text, string functionCallMeta)
    {
        if (!string.IsNullOrEmpty(functionCallMeta))
        {
            var q = ExtractFromJson(functionCallMeta, "query", "q", "search");
            if (q != null) { LogStrategy("metadados SK"); return q; }
        }

        var m = Regex(@"\{\s*""search""\s*:\s*""([^""]+)""", text);
        if (m != null) { LogStrategy(@"{""search"":...}"); return m; }

        var m2b = Regex(@"\{\s*""search""\s+""([^""]+)""", text);
        if (m2b != null) { LogStrategy(@"{""search"" ...} sem dois pontos"); return m2b; }

        var m2c = Regex(@"\{\s*search\s*:?\s*""([^""]+)""", text,
            System.Text.RegularExpressions.RegexOptions.IgnoreCase);
        if (m2c != null) { LogStrategy("search sem aspas na chave"); return m2c; }

        var m2 = Regex(@"""(?:query|tool_input|input)""\s*:\s*""([^""]+)""", text);
        if (m2 != null) { LogStrategy(@"{""query"":...}"); return m2; }

        var m3 = Regex(@"Action Input:\s*(.+?)(?:\n|$)", text);
        if (m3 != null) { LogStrategy("ReAct Action Input"); return m3.Trim(); }

        var m4 = Regex(@"<search>\s*(.+?)\s*</search>", text);
        if (m4 != null) { LogStrategy("XML <search>"); return m4; }

        var m5 = Regex(@"\[SEARCH:\s*(.+?)\]", text,
            System.Text.RegularExpressions.RegexOptions.IgnoreCase);
        if (m5 != null) { LogStrategy("[SEARCH: ...]"); return m5; }

        var m6 = Regex(
            @"(?:vou buscar|vou pesquisar|deixa eu buscar|let me search|searching for|buscar por|pesquisar por)[:\s]+[""']?([^""'\n]{5,80})[""']?",
            text, System.Text.RegularExpressions.RegexOptions.IgnoreCase);
        if (m6 != null) { LogStrategy("keyword intent"); return m6.Trim(' ', '.', '"', '\''); }

        return null;
    }

    private static string? ExtractFromJson(string json, params string[] keys)
    {
        try
        {
            var doc = JsonDocument.Parse(json);
            foreach (var key in keys)
                if (doc.RootElement.TryGetProperty(key, out var val))
                    return val.GetString();
        }
        catch { }
        return null;
    }

    private static string? Regex(string pattern, string text,
        System.Text.RegularExpressions.RegexOptions opts =
        System.Text.RegularExpressions.RegexOptions.None)
    {
        var m = System.Text.RegularExpressions.Regex.Match(text, pattern, opts);
        return m.Success ? m.Groups[1].Value : null;
    }

    private void LogStrategy(string strategy)
    {
        if (!DiagnosticMode) return;
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"  [DIAG] ✅ Estratégia: {strategy}");
        Console.ResetColor();
    }

    private AgentState BuildState(string threadId)
    {
        var history = new ChatHistory();
        foreach (var msg in _repo.GetMessages(threadId))
        {
            switch (msg.Role)
            {
                case MessageRole.User: history.AddUserMessage(msg.Content); break;
                case MessageRole.Assistant: history.AddAssistantMessage(msg.Content); break;
                case MessageRole.Tool:
                    history.AddUserMessage($"[Resultado da busca]\n{msg.Content}"); break;
            }
        }
        return new AgentState { ThreadId = threadId, Messages = history };
    }
}