namespace AgentEmailRagMultAgentesSemanticKernel;

// ════════════════════════════════════════════════════════════════════════════
//  Models — equivalentes aos TypedDict / BaseModel / dataclass do notebook
// ════════════════════════════════════════════════════════════════════════════

/// <summary>
/// Perfil do usuário dono da caixa de entrada.
/// Equivalente ao dict profile = {"name": ..., "full_name": ..., ...}
/// </summary>
public record UserProfile
{
    public string Name { get; init; } = string.Empty;
    public string FullName { get; init; } = string.Empty;
    public string UserProfileBackground { get; init; } = string.Empty;
}

/// <summary>
/// Regras de triagem configuradas pelo usuário.
/// Equivalente ao prompt_instructions["triage_rules"] do notebook.
/// </summary>
public record TriageRules
{
    public string Ignore { get; init; } = string.Empty;
    public string Notify { get; init; } = string.Empty;
    public string Respond { get; init; } = string.Empty;
}

/// <summary>
/// Representa um e-mail de entrada.
/// Equivalente ao dict email_input do notebook.
/// </summary>
public record EmailInput
{
    public string Author { get; init; } = string.Empty;
    public string To { get; init; } = string.Empty;
    public string Subject { get; init; } = string.Empty;
    public string EmailThread { get; init; } = string.Empty;
}

/// <summary>
/// Classificação do e-mail pelo Router.
/// Equivalente ao Literal["ignore", "respond", "notify"] do notebook.
/// </summary>
public enum EmailClassification
{
    Ignore,
    Notify,
    Respond
}

/// <summary>
/// Resultado da triagem com raciocínio.
/// Equivalente ao class Router(BaseModel) do notebook.
/// </summary>
public record TriageResult
{
    public string Reasoning { get; init; } = string.Empty;
    public EmailClassification Classification { get; init; } = EmailClassification.Ignore;
}

/// <summary>
/// Estado compartilhado do grafo.
/// Equivalente ao class State(TypedDict) do notebook:
///   email_input: dict
///   messages: Annotated[list, add_messages]
/// </summary>
public class AgentState
{
    public EmailInput EmailInput { get; set; } = new();
    public List<ChatMessage> Messages { get; set; } = new();
    public TriageResult? Triage { get; set; }
    public string FinalReply { get; set; } = string.Empty;
}

/// <summary>Mensagem de chat no histórico do agente.</summary>
public record ChatMessage
{
    public string Role { get; init; } = "user";  // user | assistant | tool
    public string Content { get; init; } = string.Empty;
    public string? ToolName { get; init; }
}

/// <summary>
/// Evento emitido pelo grafo durante o streaming.
/// </summary>
public class GraphEvent
{
    public string NodeName { get; init; } = string.Empty;
    public AgentState State { get; init; } = new();
    public bool IsEnd { get; init; } = false;
    public string Log { get; init; } = string.Empty;
}
