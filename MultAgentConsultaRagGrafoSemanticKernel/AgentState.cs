namespace MultAgentConsultaRagGrafoSemanticKernel;

// ════════════════════════════════════════════════════════════════════════════
//  AgentState — equivalente ao TypedDict do notebook Python:
//
//  Python:                          C#:
//  class AgentState(TypedDict):     public class AgentState
//      task: str                        Task            (tarefa do usuário)
//      plan: str                        Plan            (esboço do planner)
//      draft: str                       Draft           (rascunho do writer)
//      critique: str                    Critique        (crítica do reflect)
//      content: List[str]               Content         (resultados do Tavily)
//      revision_number: int             RevisionNumber  (iteração atual)
//      max_revisions: int               MaxRevisions    (limite de revisões)
// ════════════════════════════════════════════════════════════════════════════

public record AgentState
{
    /// <summary>Tarefa/tema fornecido pelo usuário.</summary>
    public string Task { get; set; } = string.Empty;

    /// <summary>Esboço gerado pelo agente Planner.</summary>
    public string Plan { get; set; } = string.Empty;

    /// <summary>Rascunho atual gerado pelo agente Writer.</summary>
    public string Draft { get; set; } = string.Empty;

    /// <summary>Crítica gerada pelo agente Reflect (revisor).</summary>
    public string Critique { get; set; } = string.Empty;

    /// <summary>
    /// Conteúdo de pesquisa coletado pelo Tavily.
    /// Equivalente ao content: List[str] do Python — acumulado entre revisões.
    /// </summary>
    public List<string> Content { get; set; } = new();

    /// <summary>Número da revisão atual (começa em 1).</summary>
    public int RevisionNumber { get; set; } = 1;

    /// <summary>Número máximo de revisões antes de finalizar.</summary>
    public int MaxRevisions { get; set; } = 2;

    /// <summary>Nome do nó atual (para logging).</summary>
    public string CurrentNode { get; set; } = string.Empty;
}

/// <summary>
/// Resultado estruturado para queries de pesquisa.
/// Equivalente ao Queries(BaseModel) do Python com structured output.
/// </summary>
public class SearchQueries
{
    public List<string> Queries { get; set; } = new();
}

/// <summary>Evento emitido por cada nó durante o streaming do grafo.</summary>
public class GraphEvent
{
    public string NodeName { get; init; } = string.Empty;
    public string Content { get; init; } = string.Empty;
    public AgentState State { get; init; } = new();
    public bool IsComplete { get; init; } = false;
}
