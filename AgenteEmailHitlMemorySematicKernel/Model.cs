namespace AgenteEmailHitlMemorySematicKernel;

// ════════════════════════════════════════════════════════════════════════════
//  Models — equivalentes aos TypedDict / BaseModel do notebook Python
// ════════════════════════════════════════════════════════════════════════════

public record UserProfile
{
    public string Name { get; init; } = string.Empty;
    public string FullName { get; init; } = string.Empty;
    public string UserProfileBackground { get; init; } = string.Empty;
}

public record TriageRules
{
    public string Ignore { get; init; } = string.Empty;
    public string Notify { get; init; } = string.Empty;
    public string Respond { get; init; } = string.Empty;
}

public record EmailInput
{
    public string From { get; init; } = string.Empty;  // equivalente ao "from" do Python
    public string To { get; init; } = string.Empty;
    public string Subject { get; init; } = string.Empty;
    public string Body { get; init; } = string.Empty;
}

public enum EmailClassification { Ignore, Notify, Respond }

public record TriageResult
{
    public string Reasoning { get; init; } = string.Empty;
    public EmailClassification Classification { get; init; } = EmailClassification.Ignore;
}

/// <summary>
/// Memória semântica — equivalente ao InMemoryStore do LangGraph/LangMem.
/// Cada entrada tem um ID único, conteúdo textual e namespace do usuário.
/// </summary>
public record MemoryEntry
{
    public string Id { get; init; } = Guid.NewGuid().ToString();
    public string UserId { get; init; } = string.Empty;
    public string Namespace { get; init; } = "collection";
    public string Content { get; init; } = string.Empty;
    public DateTime CreatedAt { get; init; } = DateTime.UtcNow;
}

/// <summary>
/// Estado do grafo — equivalente ao class State(TypedDict) do notebook:
///   email_input: dict
///   messages: Annotated[list, add_messages]
/// </summary>
public class AgentState
{
    public EmailInput EmailInput { get; set; } = new();
    public List<ChatMessage> Messages { get; set; } = new();
    public TriageResult? Triage { get; set; }
    public string FinalReply { get; set; } = string.Empty;
    public string UserId { get; set; } = "default";
    public string NextNode { get; set; } = string.Empty;
}

public record ChatMessage
{
    public string Role { get; init; } = "user";
    public string Content { get; init; } = string.Empty;
    public string? ToolName { get; init; }
}
