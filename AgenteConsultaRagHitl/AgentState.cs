using Microsoft.SemanticKernel.ChatCompletion;

namespace AgenteConsultaRagHitl;

// ════════════════════════════════════════════════════════════════════════════
//  AgentState — equivalente ao TypedDict do LangGraph:
//
//  Python:                              C#:
//  ─────────────────────────────────    ────────────────────────────────────
//  class AgentState(TypedDict):         public class AgentState
//      messages: Annotated[               ChatHistory Messages   (acumulado)
//          list[AnyMessage],              List<ToolCallRequest>  (pendentes)
//          reduce_messages]               bool IsInterrupted
//                                         string ThreadId
// ════════════════════════════════════════════════════════════════════════════

public class AgentState
{
    /// <summary>
    /// Histórico completo da conversa — acumulado a cada nó.
    /// Equivalente ao Annotated[list[AnyMessage], operator.add].
    /// </summary>
    public ChatHistory Messages { get; set; } = new();

    /// <summary>Thread de conversa (equivalente ao thread_id do LangGraph).</summary>
    public string ThreadId { get; set; } = string.Empty;

    /// <summary>
    /// Tool calls que o LLM quer executar e que aguardam aprovação humana.
    /// Equivalente ao tool_calls do AIMessage quando o grafo pausa em
    /// interrupt_before=["action"].
    /// </summary>
    public List<ToolCallRequest> PendingToolCalls { get; set; } = new();

    /// <summary>
    /// Indica se o grafo está pausado aguardando intervenção humana.
    /// Equivalente ao estado após interrupt_before.
    /// </summary>
    public bool IsInterrupted { get; set; } = false;

    /// <summary>
    /// Permite ao humano injetar uma resposta diretamente no estado,
    /// sobrepondo o que o agente responderia.
    /// Equivalente ao graph.update_state() do notebook Python.
    /// </summary>
    public string? InjectedResponse { get; set; } = null;

    /// <summary>Resposta final gerada (pelo LLM ou injetada pelo humano).</summary>
    public string FinalResponse { get; set; } = string.Empty;
}
