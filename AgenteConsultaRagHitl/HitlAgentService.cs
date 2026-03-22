using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.Ollama;
using System.Text;
using System.Text.RegularExpressions;

namespace AgenteConsultaRagHitl;

/// <summary>
/// Agente HITL construído sobre o StateGraph.
///
/// Estrutura do grafo — espelho fiel do notebook Python:
///
///  Python:                              C#:
///  ─────────────────────────────────    ──────────────────────────────
///  graph.add_node("llm", call_gemini)   AddStreamingNode("llm", LlmNode)
///  graph.add_node("action", take_action) AddNode("action", ActionNode)
///  graph.add_conditional_edges(         AddConditionalEdge(
///      "llm", exists_action,                "llm", ExistsAction,
///      {True:"action", False:END})           {"tool":"action","end":END})
///  graph.add_edge("action","llm")       AddEdge("action","llm")
///  graph.set_entry_point("llm")         SetEntryPoint("llm")
///  graph.compile(                       Compile(
///      checkpointer=memory,                 interruptBefore:["action"])
///      interrupt_before=["action"])
/// </summary>
public class HitlAgentService : IDisposable
{
    private readonly Kernel _kernel;
    private readonly IChatCompletionService _chat;
    private readonly CheckpointRepository _repo;
    private readonly WebSearchPlugin _search;
    private readonly string _systemPrompt;
    private readonly CompiledGraph<AgentState> _graph;

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

        // ── Monta o grafo — equivalente ao __init__ da classe Agent do Python ──
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
            .Compile(interruptBefore: ["action"]); // ← interrupt_before=["action"]
    }

    // ════════════════════════════════════════════════════════════════════════
    //  API pública
    // ════════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Processa mensagem do usuário com streaming e suporte a HITL.
    ///
    /// Equivalente às células 10 e 12 do notebook:
    ///   for event in abot.graph.stream({messages}, thread_config): ...
    /// </summary>
    public async IAsyncEnumerable<GraphEvent> RunAsync(string threadId, string userMessage)
    {
        // Persiste mensagem do usuário
        _repo.SaveMessage(threadId, MessageRole.User, userMessage);

        var state = BuildState(threadId);

        await foreach (var nodeEvent in _graph.StreamAsync(state))
        {
            await foreach (var evt in ProcessNodeEventAsync(nodeEvent, threadId))
                yield return evt;
        }
    }

    /// <summary>
    /// Retoma o grafo após aprovação humana.
    /// Equivalente a: abot.graph.stream(None, thread_config)
    ///                                   ^^^^ None = retomar do checkpoint
    /// </summary>
    public async IAsyncEnumerable<GraphEvent> ResumeAsync(string threadId)
    {
        var checkpoint = _repo.GetInterruptCheckpoint(threadId);
        if (checkpoint == null)
        {
            yield return new GraphEvent
            {
                Type = GraphEventType.GraphFinished,
                Content = "Nenhum checkpoint de interrupção encontrado."
            };
            yield break;
        }

        _repo.ClearCheckpoint(threadId);

        var state = BuildState(threadId);
        // Restaura o tool call pendente do checkpoint
        state.PendingToolCalls.Add(new ToolCallRequest
        {
            Name = checkpoint.Value.ToolName,
            Query = checkpoint.Value.ToolQuery,
            Id = checkpoint.Value.ToolId
        });

        await foreach (var nodeEvent in _graph.ResumeAsync(state, "action"))
        {
            await foreach (var evt in ProcessNodeEventAsync(nodeEvent, threadId))
                yield return evt;
        }
    }

    /// <summary>
    /// Injeta uma resposta manualmente no estado — equivalente ao:
    ///   graph.update_state(thread_config, modified_state_values)
    /// da célula 14 do notebook Python.
    /// </summary>
    public void InjectResponse(string threadId, string injectedContent)
    {
        _repo.ClearCheckpoint(threadId);
        _repo.SaveMessage(threadId, MessageRole.Assistant,
            $"[Resposta injetada manualmente]\n{injectedContent}");

        Console.ForegroundColor = ConsoleColor.Magenta;
        Console.WriteLine("\n  ✏️  [HITL] Resposta injetada no estado do grafo.");
        Console.WriteLine($"  Conteúdo: {injectedContent}");
        Console.ResetColor();
    }

    public List<ChatMessage> GetHistory(string threadId) => _repo.GetMessages(threadId);
    public List<string> ListThreads() => _repo.ListThreads();
    public void ClearThread(string threadId) => _repo.ClearThread(threadId);

    public bool HasPendingInterrupt(string threadId) =>
        _repo.GetInterruptCheckpoint(threadId) != null;

    public void Dispose() => _search.Dispose();

    // ════════════════════════════════════════════════════════════════════════
    //  Nós do grafo
    // ════════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Nó "llm" — equivalente ao call_gemini() do notebook Python.
    /// Gera resposta com streaming e detecta tool calls no texto.
    /// </summary>
    private async IAsyncEnumerable<(AgentState, string?)> LlmNodeAsync(AgentState state)
    {
        var history = state.Messages;

        // Injeta system prompt se ainda não estiver no histórico
        if (history.Count == 0 || history[0].Role != AuthorRole.System)
        {
            if (!string.IsNullOrWhiteSpace(_systemPrompt))
                history.Insert(0, new Microsoft.SemanticKernel.ChatMessageContent(
                    AuthorRole.System, _systemPrompt));
        }

        var fullResponse = new StringBuilder();

#pragma warning disable SKEXP0070
        var settings = new OllamaPromptExecutionSettings { Temperature = 0.7f };
#pragma warning restore SKEXP0070

        await foreach (var chunk in _chat.GetStreamingChatMessageContentsAsync(
            history, executionSettings: settings, kernel: _kernel))
        {
            if (!string.IsNullOrEmpty(chunk.Content))
            {
                fullResponse.Append(chunk.Content);
                yield return (state, chunk.Content); // streaming token a token
            }
        }

        var responseText = fullResponse.ToString();

        // Detecta tool calls no texto gerado
        var toolCalls = ExtractToolCalls(responseText);

        state.PendingToolCalls = toolCalls;
        state.Messages.AddAssistantMessage(responseText);

        yield return (state, null); // estado final sem token
    }

    /// <summary>
    /// Router — equivalente ao exists_action() do notebook Python.
    /// Decide se vai para "action" (executa ferramenta) ou END (resposta direta).
    /// </summary>
    private static string ExistsAction(AgentState state) =>
        state.PendingToolCalls.Count > 0 ? "tool" : "end";

    /// <summary>
    /// Nó "action" — equivalente ao take_action() do notebook Python.
    /// Executa as ferramentas solicitadas pelo LLM.
    /// </summary>
    private async Task<AgentState> ActionNodeAsync(AgentState state)
    {
        foreach (var tc in state.PendingToolCalls)
        {
            var result = await _search.SearchDirectAsync(tc.Query);
            _repo.SaveMessage(state.ThreadId, MessageRole.Tool, result, tc.Name);
            state.Messages.AddUserMessage($"[Resultado da ferramenta {tc.Name}]\n{result}");
        }

        state.PendingToolCalls.Clear();
        return state;
    }

    // ════════════════════════════════════════════════════════════════════════
    //  Processamento de eventos do grafo
    // ════════════════════════════════════════════════════════════════════════

    private async IAsyncEnumerable<GraphEvent> ProcessNodeEventAsync(
        NodeEvent<AgentState> nodeEvent,
        string threadId)
    {
        // ── Interrupção HITL ──────────────────────────────────────────────────
        // Equivalente ao interrupt_before=["action"] detectado pelo grafo
        if (nodeEvent.IsInterrupted)
        {
            var pending = nodeEvent.State.PendingToolCalls;

            // Persiste checkpoint para permitir retomada
            if (pending.Count > 0)
            {
                var tc = pending[0];
                _repo.SaveInterruptCheckpoint(
                    threadId, nodeEvent.NodeName,
                    tc.Name, tc.Query, tc.Id);
            }

            yield return new GraphEvent
            {
                Type = GraphEventType.HumanInterruptRequired,
                NodeName = nodeEvent.NodeName,
                Content = pending.Count > 0
                    ? $"Ferramenta: {pending[0].Name} | Query: \"{pending[0].Query}\""
                    : "Ação pendente requer aprovação"
            };
            yield break;
        }

        // ── Nó LLM: streaming de tokens ──────────────────────────────────────
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

            var text = fullText.ToString();
            var hasTc = nodeEvent.State.PendingToolCalls.Count > 0;

            // Persiste resposta do assistente (se ainda não foi pelo nó)
            if (!string.IsNullOrWhiteSpace(text) && !hasTc)
            {
                _repo.SaveMessage(threadId, MessageRole.Assistant, text);
                yield return new GraphEvent
                {
                    Type = GraphEventType.LlmDirectResponse,
                    Content = text,
                    NodeName = "llm"
                };
            }
            else if (hasTc)
            {
                var tc = nodeEvent.State.PendingToolCalls[0];
                yield return new GraphEvent
                {
                    Type = GraphEventType.LlmToolDecision,
                    Content = $"Quer chamar: {tc.Name}(\"{tc.Query}\")",
                    ToolCall = tc,
                    NodeName = "llm"
                };
            }
        }

        // ── Nó Action: ferramenta executada ───────────────────────────────────
        if (nodeEvent.NodeName == "action")
        {
            yield return new GraphEvent
            {
                Type = GraphEventType.ToolResult,
                Content = "Ferramenta executada com sucesso.",
                NodeName = "action"
            };
        }

        // ── Fim do grafo ──────────────────────────────────────────────────────
        if (nodeEvent.State.FinalResponse != string.Empty || nodeEvent.NodeName == "llm")
        {
            yield return new GraphEvent
            {
                Type = GraphEventType.GraphFinished,
                NodeName = nodeEvent.NodeName
            };
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  Helpers
    // ════════════════════════════════════════════════════════════════════════

    private AgentState BuildState(string threadId)
    {
        var history = new Microsoft.SemanticKernel.ChatCompletion.ChatHistory();

        foreach (var msg in _repo.GetMessages(threadId))
        {
            switch (msg.Role)
            {
                case MessageRole.User:
                    history.AddUserMessage(msg.Content); break;
                case MessageRole.Assistant:
                    history.AddAssistantMessage(msg.Content); break;
                case MessageRole.Tool:
                    history.AddUserMessage(
                        $"[Resultado da ferramenta {msg.ToolName}]\n{msg.Content}"); break;
            }
        }

        return new AgentState { ThreadId = threadId, Messages = history };
    }

    /// <summary>
    /// Detecta tool calls no texto do LLM.
    /// Modelos Ollama retornam chamadas de função como JSON ou como texto estruturado.
    /// </summary>
    private static List<ToolCallRequest> ExtractToolCalls(string text)
    {
        var calls = new List<ToolCallRequest>();

        // Padrão: {"tool":"search_web","query":"..."}
        var match = Regex.Match(text,
            @"\{[""']?tool[""']?\s*:\s*[""']?(\w+)[""']?\s*,\s*[""']?(?:query|args)[""']?\s*:\s*[""']([^""']+)[""']",
            RegexOptions.IgnoreCase);

        if (match.Success)
        {
            calls.Add(new ToolCallRequest
            {
                Name = match.Groups[1].Value,
                Query = match.Groups[2].Value
            });
        }

        // Padrão alternativo: [search_web: "query"]  ou  search_web("query")
        if (calls.Count == 0)
        {
            var m2 = Regex.Match(text,
                @"search_web\s*[\(\[:\s]+[""']?([^""'\)\]]+)[""']?[\)\]]?",
                RegexOptions.IgnoreCase);
            if (m2.Success)
                calls.Add(new ToolCallRequest
                {
                    Name = "search_web",
                    Query = m2.Groups[1].Value.Trim()
                });
        }

        return calls;
    }
}
