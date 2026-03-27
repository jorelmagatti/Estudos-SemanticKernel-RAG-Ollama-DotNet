using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.Ollama;

namespace MultAgentConsultaRagGrafoSemanticKernel;

// ════════════════════════════════════════════════════════════════════════════
//  MultiAgentOrchestrator — monta e executa o grafo de 5 agentes
//
//  Equivalência com o notebook Python:
//
//  Python (LangGraph)                   C# (StateGraph<AgentState>)
//  ──────────────────────────────────   ──────────────────────────────────
//  plan_node()                      →   PlannerNodeAsync()
//  research_plan_node()             →   ResearchPlanNodeAsync()
//  generation_node()                →   GenerationNodeAsync()
//  reflection_node()                →   ReflectionNodeAsync()
//  research_critique_node()         →   ResearchCritiqueNodeAsync()
//  should_continue()                →   ShouldContinue()
//
//  Grafo:
//  planner → research_plan → generate ──should_continue──→ reflect → research_critique ──┐
//                                 ↑                                                       │
//                                 └───────────────────────────────────────────────────────┘
//                                 (loop até max_revisions → END)
// ════════════════════════════════════════════════════════════════════════════

public class MultiAgentOrchestrator : IDisposable
{
    private readonly Kernel _kernel;
    private readonly IChatCompletionService _chat;
    private readonly TavilySearchService _tavily;
    private readonly CompiledGraph<AgentState> _graph;

    public MultiAgentOrchestrator(Kernel kernel, TavilySearchService tavily)
    {
        _kernel = kernel;
        _chat = kernel.GetRequiredService<IChatCompletionService>();
        _tavily = tavily;

        // ── Monta o grafo — equivalente às células 18-23 do notebook ──────────
        _graph = new StateGraph<AgentState>()
            .AddNode("planner", PlannerNodeAsync)
            .AddNode("research_plan", ResearchPlanNodeAsync)
            .AddNode("generate", GenerationNodeAsync)
            .AddNode("reflect", ReflectionNodeAsync)
            .AddNode("research_critique", ResearchCritiqueNodeAsync)
            .SetEntryPoint("planner")
            .AddEdge("planner", "research_plan")
            .AddEdge("research_plan", "generate")
            .AddConditionalEdge(
                from: "generate",
                router: ShouldContinue,
                routeMap: new Dictionary<string, string>
                {
                    [GraphEnd.Node] = GraphEnd.Node,
                    ["reflect"] = "reflect"
                })
            .AddEdge("reflect", "research_critique")
            .AddEdge("research_critique", "generate")
            .Compile();
    }

    // ════════════════════════════════════════════════════════════════════════
    //  API pública
    // ════════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Executa o pipeline multi-agente completo emitindo eventos por nó.
    /// Equivalente ao: for s in graph.stream({task, max_revisions, ...}, thread)
    /// </summary>
    public IAsyncEnumerable<NodeEvent<AgentState>> RunAsync(
        string task,
        int maxRevisions = 2) =>
        _graph.StreamAsync(new AgentState
        {
            Task = task,
            MaxRevisions = maxRevisions,
            RevisionNumber = 1,
            Content = new List<string>()
        });

    public void Dispose() => _tavily.Dispose();

    // ════════════════════════════════════════════════════════════════════════
    //  Nós do grafo (agentes)
    // ════════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Agente Planner — cria o esboço de alto nível.
    /// Equivalente ao plan_node() do notebook Python.
    /// </summary>
    private async Task<AgentState> PlannerNodeAsync(AgentState state)
    {
        var history = new ChatHistory();
        history.AddSystemMessage(AgentPrompts.Planner);
        history.AddUserMessage(state.Task);

        var response = await _chat.GetChatMessageContentAsync(history, kernel: _kernel);

        return state with { Plan = response.Content ?? string.Empty, CurrentNode = "planner" };
    }

    /// <summary>
    /// Agente ResearchPlan — gera queries e busca conteúdo para o plano inicial.
    /// Equivalente ao research_plan_node() do notebook Python.
    ///
    /// No Python usa model.with_structured_output(Queries) para forçar JSON.
    /// Aqui instruímos o modelo a retornar uma query por linha e parseamos.
    /// </summary>
    private async Task<AgentState> ResearchPlanNodeAsync(AgentState state)
    {
        var queries = await GenerateSearchQueriesAsync(
            AgentPrompts.ResearchPlan,
            $"Tarefa: {state.Task}\n\nPlano: {state.Plan}");

        var newContent = await _tavily.SearchMultipleAsync(queries, maxResultsEach: 2);

        var updatedContent = new List<string>(state.Content);
        updatedContent.AddRange(newContent);

        return state with { Content = updatedContent, CurrentNode = "research_plan" };
    }

    /// <summary>
    /// Agente Writer — gera/revisa a redação.
    /// Equivalente ao generation_node() do notebook Python.
    /// </summary>
    private async Task<AgentState> GenerationNodeAsync(AgentState state)
    {
        var content = string.Join("\n\n", state.Content);
        var writerPrompt = AgentPrompts.Writer.Replace("{content}", content);

        var history = new ChatHistory();
        history.AddSystemMessage(writerPrompt);
        history.AddUserMessage(
            $"{state.Task}\n\nAqui está meu plano:\n\n{state.Plan}");

        // Se há uma crítica anterior, adiciona ao histórico para revisão
        if (!string.IsNullOrWhiteSpace(state.Critique))
        {
            history.AddAssistantMessage(state.Draft);
            history.AddUserMessage(
                $"Aqui está a crítica recebida:\n\n{state.Critique}\n\n" +
                "Por favor, revise a redação incorporando as sugestões acima.");
        }

        var response = await _chat.GetChatMessageContentAsync(history, kernel: _kernel);

        return state with
        {
            Draft = response.Content ?? string.Empty,
            RevisionNumber = state.RevisionNumber + 1,
            CurrentNode = "generate"
        };
    }

    /// <summary>
    /// Agente Reflect — critica a redação e sugere melhorias.
    /// Equivalente ao reflection_node() do notebook Python.
    /// </summary>
    private async Task<AgentState> ReflectionNodeAsync(AgentState state)
    {
        var history = new ChatHistory();
        history.AddSystemMessage(AgentPrompts.Reflect);
        history.AddUserMessage(state.Draft);

        var response = await _chat.GetChatMessageContentAsync(history, kernel: _kernel);

        return state with { Critique = response.Content ?? string.Empty, CurrentNode = "reflect" };
    }

    /// <summary>
    /// Agente ResearchCritique — busca conteúdo para endereçar as críticas.
    /// Equivalente ao research_critique_node() do notebook Python.
    /// </summary>
    private async Task<AgentState> ResearchCritiqueNodeAsync(AgentState state)
    {
        var queries = await GenerateSearchQueriesAsync(
            AgentPrompts.ResearchCritique,
            $"Crítica: {state.Critique}");

        var newContent = await _tavily.SearchMultipleAsync(queries, maxResultsEach: 2);

        var updatedContent = new List<string>(state.Content);
        updatedContent.AddRange(newContent);

        return state with { Content = updatedContent, CurrentNode = "research_critique" };
    }

    // ════════════════════════════════════════════════════════════════════════
    //  Router — equivalente ao should_continue() do notebook Python
    // ════════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Decide se continua revisando ou finaliza.
    /// Equivalente ao should_continue() do notebook Python:
    ///   if state["revision_number"] > state["max_revisions"]: return END
    ///   return "reflect"
    /// </summary>
    private static string ShouldContinue(AgentState state)
    {
        if (state.RevisionNumber > state.MaxRevisions)
            return GraphEnd.Node;
        return "reflect";
    }

    // ════════════════════════════════════════════════════════════════════════
    //  Helpers
    // ════════════════════════════════════════════════════════════════════════

    /// <summary>
    /// Gera queries de pesquisa usando o LLM.
    /// Equivalente ao model.with_structured_output(Queries).invoke([...]) do Python.
    /// Como o Ollama não suporta structured output nativo, instruímos o modelo
    /// a retornar uma query por linha e parseamos o texto.
    /// </summary>
    private async Task<List<string>> GenerateSearchQueriesAsync(
        string systemPrompt,
        string userMessage)
    {
        var history = new ChatHistory();
        history.AddSystemMessage(systemPrompt);
        history.AddUserMessage(userMessage);

#pragma warning disable SKEXP0070
        var settings = new OllamaPromptExecutionSettings { Temperature = 0.3f };
#pragma warning restore SKEXP0070

        var response = await _chat.GetChatMessageContentAsync(
            history, executionSettings: settings, kernel: _kernel);

        // Parse: uma query por linha, ignora linhas vazias e numeração
        var queries = (response.Content ?? string.Empty)
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(l => l.Trim().TrimStart('-', '*', '1', '2', '3', '.', ' ').Trim())
            .Where(l => l.Length > 5)
            .Take(3)
            .ToList();

        return queries;
    }
}
