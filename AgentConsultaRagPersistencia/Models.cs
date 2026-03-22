namespace AgentConsultaRagPersistencia;

/// <summary>
/// Papéis possíveis de uma mensagem no histórico de conversa.
/// Equivalente aos HumanMessage / AIMessage / ToolMessage do LangChain.
/// </summary>
public enum MessageRole
{
    User,
    Assistant,
    Tool
}

/// <summary>
/// Mensagem armazenada no SQLite — equivalente às entradas do SqliteSaver do LangGraph.
/// </summary>
public class ChatMessage
{
    public long Id { get; set; }
    public string ThreadId { get; set; } = string.Empty;
    public MessageRole Role { get; set; }
    public string Content { get; set; } = string.Empty;
    public string? ToolName { get; set; }          // preenchido quando Role == Tool
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Resultado bruto de uma busca web.
/// </summary>
public class WebSearchResult
{
    public string Title { get; set; } = string.Empty;
    public string Url { get; set; } = string.Empty;
    public string Snippet { get; set; } = string.Empty;
}

/// <summary>
/// Evento emitido durante o streaming do agente — equivalente aos eventos
/// do abot.graph.stream() do LangGraph que o notebook itera em tempo real.
/// </summary>
public class AgentStreamEvent
{
    /// <summary>Nome do nó que emitiu o evento: "llm", "tool", "final".</summary>
    public string Node { get; set; } = string.Empty;

    /// <summary>Conteúdo do token ou mensagem sendo transmitido.</summary>
    public string Content { get; set; } = string.Empty;

    /// <summary>Indica que é o último token do nó atual.</summary>
    public bool IsNodeEnd { get; set; } = false;
}