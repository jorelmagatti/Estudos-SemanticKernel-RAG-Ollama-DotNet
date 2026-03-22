using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.Ollama;
using System.Text;
using System.Text.Json;

namespace AgentConsultaRagPersistencia;


/// <summary>
/// Agente conversacional com persistência SQLite e streaming.
///
/// Equivalência com o notebook Python:
///
///   Python (LangGraph)                   │  C# (este serviço)
///   ─────────────────────────────────────┼──────────────────────────────────
///   AgentState { messages }              │  ChatHistory reconstruída do SQLite
///   SqliteSaver(conn)                    │  ConversationRepository (SQLite)
///   graph.stream({messages}, thread)     │  RunStreamingAsync() — IAsyncEnumerable
///   nó "llm"  → call_gemini()           │  StreamLlmNodeAsync()
///   nó "action" → take_action()         │  ExecuteToolNodeAsync()
///   exists_action() → edge condicional  │  loop: verifica tool_calls da resposta
///   thread_id isolamento de contexto    │  threadId → histórico separado no SQLite
/// </summary>
public class AgentService
{
    private readonly Kernel _kernel;
    private readonly IChatCompletionService _chat;
    private readonly ConversationRepository _repo;
    private readonly string _systemPrompt;

    public AgentService(
        Kernel kernel,
        ConversationRepository repo,
        string systemPrompt = "")
    {
        _kernel = kernel;
        _repo = repo;
        _systemPrompt = systemPrompt;
        _chat = kernel.GetRequiredService<IChatCompletionService>();
    }

    /// <summary>
    /// Processa a mensagem do usuário e transmite eventos do agente em tempo real.
    ///
    /// Equivalente a:
    ///   for event in abot.graph.stream({messages}, thread):
    ///       for k, v in event.items():
    ///           print(f"{k}: {v['messages']}")
    ///
    /// Cada AgentStreamEvent representa um evento de um "nó" do grafo:
    ///   Node = "llm"   → token gerado pelo modelo
    ///   Node = "tool"  → ferramenta sendo chamada / resultado
    ///   Node = "final" → resposta completa finalizada
    /// </summary>
    public async IAsyncEnumerable<AgentStreamEvent> RunStreamingAsync(
        string threadId,
        string userMessage)
    {
        // ── 1. Persiste mensagem do usuário ───────────────────────────────────
        _repo.SaveMessage(threadId, MessageRole.User, userMessage);

        // ── 2. Reconstrói ChatHistory a partir do SQLite ──────────────────────
        //    Equivalente ao LangGraph recarregando o checkpoint do thread_id
        var history = BuildChatHistory(threadId);

        // ── 3. Loop ReAct: LLM → (tool?) → LLM → ... → resposta final ────────
        //    Equivalente ao ciclo llm → exists_action → action → llm do grafo
        var maxIterations = 6;
        var iteration = 0;

        while (iteration++ < maxIterations)
        {
            // ── Nó "llm": chama o modelo com streaming ────────────────────────
            yield return new AgentStreamEvent { Node = "llm", Content = "", IsNodeEnd = false };

            var fullResponse = new StringBuilder();
            var toolCallsJson = new StringBuilder();
            var hasToolCalls = false;

            // Streaming token a token — equivalente ao yield dos eventos do LangGraph
#pragma warning disable SKEXP0070
            var streamSettings = new OllamaPromptExecutionSettings
            {
                Temperature = 0.7f,
            };
#pragma warning restore SKEXP0070

            await foreach (var chunk in _chat.GetStreamingChatMessageContentsAsync(
                history,
                executionSettings: streamSettings,
                kernel: _kernel))
            {
                if (!string.IsNullOrEmpty(chunk.Content))
                {
                    fullResponse.Append(chunk.Content);
                    yield return new AgentStreamEvent
                    {
                        Node = "llm",
                        Content = chunk.Content,
                        IsNodeEnd = false
                    };
                }

                // Captura tool_calls se o modelo as incluiu nos metadados
                if (chunk.Metadata?.ContainsKey("tool_calls") == true)
                {
                    var tc = chunk.Metadata["tool_calls"]?.ToString();
                    if (!string.IsNullOrEmpty(tc))
                    {
                        toolCallsJson.Append(tc);
                        hasToolCalls = true;
                    }
                }
            }

            yield return new AgentStreamEvent
            {
                Node = "llm",
                Content = "\n",
                IsNodeEnd = true
            };

            var assistantContent = fullResponse.ToString();

            // ── Detecção de tool calls via texto (função calling textual) ─────
            //    Ollama local retorna function calls como JSON no conteúdo
            var toolCalls = ExtractToolCalls(assistantContent);
            if (toolCalls.Count > 0) hasToolCalls = true;

            // ── Persiste resposta do assistente ───────────────────────────────
            _repo.SaveMessage(threadId, MessageRole.Assistant, assistantContent);
            history.AddAssistantMessage(assistantContent);

            // ── exists_action: há tool calls? ─────────────────────────────────
            if (!hasToolCalls || toolCalls.Count == 0)
                break; // resposta final — sai do loop

            // ── Nó "tool": executa as ferramentas chamadas ────────────────────
            //    Equivalente ao nó "action" → take_action() do LangGraph
            foreach (var tc in toolCalls)
            {
                yield return new AgentStreamEvent
                {
                    Node = "tool",
                    Content = $"⚙️  Chamando ferramenta: {tc.Name}({tc.ArgsDisplay})\n",
                    IsNodeEnd = false
                };

                var toolResult = await InvokeToolAsync(tc);

                yield return new AgentStreamEvent
                {
                    Node = "tool",
                    Content = $"📋 Resultado: {toolResult[..Math.Min(300, toolResult.Length)]}\n",
                    IsNodeEnd = true
                };

                // Persiste resultado da ferramenta
                _repo.SaveMessage(threadId, MessageRole.Tool, toolResult, tc.Name);

                // Adiciona resultado ao histórico para o próximo turno do LLM
                history.AddUserMessage($"[Resultado da ferramenta {tc.Name}]\n{toolResult}");
            }
            // Volta ao topo do loop — LLM processa os resultados das ferramentas
        }

        // ── 4. Evento final ───────────────────────────────────────────────────
        yield return new AgentStreamEvent
        {
            Node = "final",
            Content = string.Empty,
            IsNodeEnd = true
        };
    }

    /// <summary>
    /// Retorna o histórico formatado de um thread (para exibição ao usuário).
    /// </summary>
    public List<ChatMessage> GetHistory(string threadId) =>
        _repo.GetHistory(threadId);

    public List<string> ListThreads() => _repo.ListThreads();

    public void ClearThread(string threadId) => _repo.ClearThread(threadId);

    // ── helpers privados ──────────────────────────────────────────────────────

    /// <summary>
    /// Reconstrói o ChatHistory do Semantic Kernel a partir do SQLite.
    /// Equivalente ao LangGraph recarregando o checkpoint persistido.
    /// </summary>
    private ChatHistory BuildChatHistory(string threadId)
    {
        var history = new ChatHistory();

        if (!string.IsNullOrWhiteSpace(_systemPrompt))
            history.AddSystemMessage(_systemPrompt);

        foreach (var msg in _repo.GetHistory(threadId))
        {
            switch (msg.Role)
            {
                case MessageRole.User:
                    history.AddUserMessage(msg.Content);
                    break;
                case MessageRole.Assistant:
                    history.AddAssistantMessage(msg.Content);
                    break;
                case MessageRole.Tool:
                    history.AddUserMessage($"[Resultado da ferramenta {msg.ToolName}]\n{msg.Content}");
                    break;
            }
        }

        return history;
    }

    /// <summary>
    /// Extrai tool calls do texto gerado pelo Ollama.
    /// Modelos locais frequentemente retornam chamadas de função como JSON no texto.
    /// </summary>
    private static List<ToolCall> ExtractToolCalls(string text)
    {
        var calls = new List<ToolCall>();

        // Padrão 1: JSON explícito {"tool": "nome", "args": {...}}
        try
        {
            var jsonStart = text.IndexOf("{\"tool\":", StringComparison.OrdinalIgnoreCase);
            if (jsonStart >= 0)
            {
                var jsonEnd = text.IndexOf('}', jsonStart + 10);
                if (jsonEnd >= 0)
                {
                    var json = text[jsonStart..(jsonEnd + 1)];
                    var doc = JsonDocument.Parse(json);
                    var name = doc.RootElement.GetProperty("tool").GetString() ?? "";
                    var args = doc.RootElement.TryGetProperty("args", out var a)
                        ? a.ToString() : "{}";
                    if (!string.IsNullOrEmpty(name))
                        calls.Add(new ToolCall { Name = name, Args = args });
                }
            }
        }
        catch { /* ignora JSON malformado */ }

        // Padrão 2: Semantic Kernel function_calls via metadados (já tratado no stream)
        return calls;
    }

    /// <summary>Invoca uma KernelFunction pelo nome com os argumentos fornecidos.</summary>
    private async Task<string> InvokeToolAsync(ToolCall tc)
    {
        try
        {
            var args = new KernelArguments();

            // Tenta deserializar os argumentos como dicionário
            if (!string.IsNullOrWhiteSpace(tc.Args) && tc.Args != "{}")
            {
                try
                {
                    var dict = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(tc.Args);
                    if (dict != null)
                        foreach (var kv in dict)
                            args[kv.Key] = kv.Value.ToString();
                }
                catch { args["query"] = tc.Args; }
            }

            var result = await _kernel.InvokeAsync(
                pluginName: "WebSearch",
                functionName: tc.Name,
                arguments: args);

            return result.ToString();
        }
        catch (Exception ex)
        {
            return $"Erro ao executar {tc.Name}: {ex.Message}";
        }
    }

    private record ToolCall
    {
        public string Name { get; init; } = string.Empty;
        public string Args { get; init; } = "{}";
        public string ArgsDisplay => Args.Length > 80 ? Args[..80] + "..." : Args;
    }
}
