using Microsoft.SemanticKernel.ChatCompletion;


namespace AgenteUtilidadesRagReActGrafo;

/// <summary>
/// Estado do agente — equivalente ao AgentState TypedDict do LangGraph:
///
///   class AgentState(TypedDict):
///       messages: Annotated[list[AnyMessage], operator.add]
///
/// O LangGraph usa operator.add para acumular mensagens a cada nó.
/// Aqui fazemos o mesmo: Messages é uma lista que só cresce (append-only).
/// </summary>
public class AgentState
{
    /// <summary>
    /// Histórico completo de mensagens da conversa.
    /// Equivalente a state['messages'] — acumulado a cada iteração do grafo.
    /// </summary>
    public ChatHistory Messages { get; } = new();

    /// <summary>
    /// Indica se o último nó "llm" solicitou chamada de ferramenta.
    /// Equivalente à aresta condicional: exists_action(state) → True/False
    /// </summary>
    public bool ToolCallPending { get; set; } = false;

    /// <summary>
    /// Nome da ferramenta a ser chamada (preenchido pelo nó llm).
    /// Equivalente a: tool_calls[-1]['name']
    /// </summary>
    public string ToolName { get; set; } = string.Empty;

    /// <summary>
    /// Argumento da ferramenta (a query de busca).
    /// Equivalente a: tool_calls[-1]['args']
    /// </summary>
    public string ToolArgument { get; set; } = string.Empty;
}
