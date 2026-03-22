namespace AgenteConsultaRagHitl;

// ════════════════════════════════════════════════════════════════════════════
//  Models — equivalentes ao TypedDict / dataclasses do notebook Python
// ════════════════════════════════════════════════════════════════════════════

public enum MessageRole { User, Assistant, Tool }

/// <summary>
/// Mensagem persistida no SQLite — equivalente às entradas do SqliteSaver.
/// </summary>
public class ChatMessage
{
    public long Id { get; set; }
    public string ThreadId { get; set; } = string.Empty;
    public MessageRole Role { get; set; }
    public string Content { get; set; } = string.Empty;
    public string? ToolName { get; set; }
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Resultado de busca web.
/// </summary>
public class WebSearchResult
{
    public string Title { get; set; } = string.Empty;
    public string Url { get; set; } = string.Empty;
    public string Snippet { get; set; } = string.Empty;
}

/// <summary>
/// Representa uma chamada de ferramenta que o LLM quer executar.
/// Equivalente ao tool_calls do AIMessage do LangChain.
/// </summary>
public class ToolCallRequest
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string Name { get; set; } = string.Empty;
    public string Query { get; set; } = string.Empty;
}

// ── Eventos de streaming ──────────────────────────────────────────────────────

/// <summary>
/// Tipo de evento emitido pelo grafo durante o streaming.
/// Equivalente às chaves do dicionário retornado por graph.stream() no Python.
/// </summary>
public enum GraphEventType
{
    /// <summary>Token de texto gerado pelo LLM (nó "llm")</summary>
    LlmToken,

    /// <summary>LLM decidiu chamar ferramenta(s) — grafo pausará</summary>
    LlmToolDecision,

    /// <summary>LLM respondeu diretamente sem chamar ferramentas</summary>
    LlmDirectResponse,

    /// <summary>Ferramenta sendo executada (nó "action")</summary>
    ToolExecuting,

    /// <summary>Resultado da ferramenta retornado</summary>
    ToolResult,

    /// <summary>
    /// Grafo pausado aguardando decisão humana — equivalente ao
    /// interrupt_before=["action"] do LangGraph.
    /// </summary>
    HumanInterruptRequired,

    /// <summary>Execução finalizada</summary>
    GraphFinished,
}

public class GraphEvent
{
    public GraphEventType Type { get; init; }
    public string Content { get; init; } = string.Empty;
    public ToolCallRequest? ToolCall { get; init; }
    public string NodeName { get; init; } = string.Empty;
}