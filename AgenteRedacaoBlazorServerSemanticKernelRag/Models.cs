namespace EssayWriterBlazor.Models;

// ════════════════════════════════════════════════════════════════════════════
//  AgentState — equivalente ao TypedDict AgentState do new_backend.py
// ════════════════════════════════════════════════════════════════════════════

public record AgentState
{
    public string       Task           { get; init; } = string.Empty;
    public string       Plan           { get; init; } = string.Empty;
    public string       Draft          { get; init; } = string.Empty;
    public string       Critique       { get; init; } = string.Empty;
    public List<string> Content        { get; init; } = new();
    public int          RevisionNumber { get; init; } = 0;
    public int          MaxRevisions   { get; init; } = 2;
}

/// <summary>
/// Representa um passo do pipeline — equivalente a cada item do graph.stream() do Python.
/// Cada step é exibido em tempo real no Blazor via streaming.
/// </summary>
public class PipelineStep
{
    public StepType Type    { get; init; }
    public string   Content { get; init; } = string.Empty;
    public int      Revision { get; init; } = 0;
}

public enum StepType
{
    Plan,           // 📝 plan_node
    ResearchPlan,   // 🔍 research_plan_node
    Draft,          // ✍️  generation_node
    Critique,       // 🧐 reflection_node
    ResearchCritique, // 🔍 research_critique_node
    Complete        // ✅ fim
}
