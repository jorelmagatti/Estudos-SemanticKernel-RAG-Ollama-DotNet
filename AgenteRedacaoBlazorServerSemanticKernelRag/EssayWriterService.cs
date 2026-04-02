using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.Ollama;
using EssayWriterBlazor.Agents;
using EssayWriterBlazor.Graph;
using EssayWriterBlazor.Models;
using EssayWriterBlazor.Plugins;

namespace EssayWriterBlazor.Services;

/// <summary>
/// Orquestrador do pipeline de redação — equivalente ao new_backend.py.
///
/// Grafo (idêntico ao Python):
///   planner → research_plan → generate ──should_continue──→ reflect
///                                ↑                              │
///                                └──── research_critique ───────┘
///                                (loop até max_revisions → END)
/// </summary>
public class EssayWriterService : IDisposable
{
    private readonly IChatCompletionService     _chat;
    private readonly Kernel                     _kernel;
    private readonly TavilySearchService        _tavily;
    private readonly CompiledGraph<AgentState>  _graph;
    private OllamaPromptExecutionSettings _ollamaConfig;
    public EssayWriterService(Kernel kernel, TavilySearchService tavily)
    {
        _kernel = kernel;
        _chat   = kernel.GetRequiredService<IChatCompletionService>();
        _tavily = tavily;
        _ollamaConfig = new() { Temperature = 0.2f };
        // Monta o grafo — espelho do new_backend.py
        _graph = new StateGraph<AgentState>()
            .AddNode("planner",           PlannerNodeAsync)
            .AddNode("research_plan",     ResearchPlanNodeAsync)
            .AddNode("generate",          GenerationNodeAsync)
            .AddNode("reflect",           ReflectionNodeAsync)
            .AddNode("research_critique", ResearchCritiqueNodeAsync)
            .SetEntryPoint("planner")
            .AddEdge("planner",           "research_plan")
            .AddEdge("research_plan",     "generate")
            .AddConditionalEdge(
                "generate",
                ShouldContinue,
                new Dictionary<string, string>
                {
                    [GraphEnd.Node] = GraphEnd.Node,
                    ["reflect"]     = "reflect"
                })
            .AddEdge("reflect",           "research_critique")
            .AddEdge("research_critique", "generate")
            .Compile();
    }

    /// <summary>
    /// Executa o pipeline e emite PipelineSteps em tempo real.
    /// Equivalente ao: for s in graph.stream(initial_state, thread_config)
    /// O Blazor consome via IAsyncEnumerable e atualiza a UI a cada passo.
    /// </summary>
    public async IAsyncEnumerable<PipelineStep> RunAsync(string task, int maxRevisions)
    {
        var initial = new AgentState
        {
            Task           = task,
            MaxRevisions   = maxRevisions,
            RevisionNumber = 0,
            Content        = new List<string>()
        };

        await foreach (var evt in _graph.StreamAsync(initial))
        {
            if (evt.IsEnd)
            {
                yield return new PipelineStep { Type = StepType.Complete };
                yield break;
            }

            // Mapeia nó → PipelineStep — equivalente ao if 'plan' in step_output do app.py
            var step = evt.NodeName switch
            {
                "planner"           => new PipelineStep
                {
                    Type     = StepType.Plan,
                    Content  = evt.State.Plan,
                    Revision = evt.State.RevisionNumber
                },
                "research_plan" or "research_critique" => new PipelineStep
                {
                    Type     = evt.NodeName == "research_plan"
                               ? StepType.ResearchPlan
                               : StepType.ResearchCritique,
                    Content  = $"{evt.State.Content.Count} chunk(s) pesquisados",
                    Revision = evt.State.RevisionNumber
                },
                "generate" => new PipelineStep
                {
                    Type     = StepType.Draft,
                    Content  = evt.State.Draft,
                    Revision = evt.State.RevisionNumber
                },
                "reflect" => new PipelineStep
                {
                    Type     = StepType.Critique,
                    Content  = evt.State.Critique,
                    Revision = evt.State.RevisionNumber
                },
                _ => null
            };

            if (step != null) yield return step;
        }
    }

    public void Dispose() => _tavily.Dispose();

    // ════════════════════════════════════════════════════════════════════════
    //  Nós do grafo — equivalentes às funções *_node do new_backend.py
    // ════════════════════════════════════════════════════════════════════════

    private async Task<AgentState> PlannerNodeAsync(AgentState state)
    {
        var h = new ChatHistory();
        h.AddSystemMessage(AgentPrompts.Plan);
        h.AddUserMessage(state.Task);
        var r = await _chat.GetChatMessageContentAsync(h, kernel: _kernel, executionSettings: _ollamaConfig);
        return state with { Plan = r.Content ?? "" };
    }

    private async Task<AgentState> ResearchPlanNodeAsync(AgentState state)
    {
        var queries = await GenerateQueriesAsync(AgentPrompts.ResearchPlan, state.Task);
        var content = await _tavily.SearchMultipleAsync(queries);
        return state with { Content = state.Content.Concat(content).ToList() };
    }

    private async Task<AgentState> GenerationNodeAsync(AgentState state)
    {
        var content = string.Join("\n\n", state.Content);
        var prompt  = AgentPrompts.Writer.Replace("{content}", content);

        var h = new ChatHistory();
        h.AddSystemMessage(prompt);
        h.AddUserMessage($"{state.Task}\n\nAqui está meu plano:\n\n{state.Plan}");

        if (!string.IsNullOrWhiteSpace(state.Critique))
        {
            h.AddAssistantMessage(state.Draft);
            h.AddUserMessage($"Crítica recebida:\n\n{state.Critique}\n\nPor favor, revise.");
        }

        var r = await _chat.GetChatMessageContentAsync(h, kernel: _kernel, executionSettings: _ollamaConfig);
        return state with
        {
            Draft          = r.Content ?? "",
            RevisionNumber = state.RevisionNumber + 1
        };
    }

    private async Task<AgentState> ReflectionNodeAsync(AgentState state)
    {
        var h = new ChatHistory();
        h.AddSystemMessage(AgentPrompts.Reflection);
        h.AddUserMessage(state.Draft);
        var r = await _chat.GetChatMessageContentAsync(h, kernel: _kernel, executionSettings: _ollamaConfig);
        return state with { Critique = r.Content ?? "" };
    }

    private async Task<AgentState> ResearchCritiqueNodeAsync(AgentState state)
    {
        var queries = await GenerateQueriesAsync(AgentPrompts.ResearchCritique, state.Critique);
        var content = await _tavily.SearchMultipleAsync(queries);
        return state with { Content = state.Content.Concat(content).ToList() };
    }

    // ── Router ───────────────────────────────────────────────────────────────

    private static string ShouldContinue(AgentState state) =>
        state.RevisionNumber > state.MaxRevisions ? GraphEnd.Node : "reflect";

    // ── Helper: gera queries com o LLM ───────────────────────────────────────

    private async Task<List<string>> GenerateQueriesAsync(string systemPrompt, string userMessage)
    {
        var h = new ChatHistory();
        h.AddSystemMessage(systemPrompt);
        h.AddUserMessage(userMessage);

        var r = await _chat.GetChatMessageContentAsync(h, executionSettings: _ollamaConfig, kernel: _kernel);

        return (r.Content ?? "")
            .Split('\n', StringSplitOptions.RemoveEmptyEntries)
            .Select(l => l.Trim().TrimStart('-', '*', '1', '2', '3', '.').Trim())
            .Where(l => l.Length > 5)
            .Take(3)
            .ToList();
    }
}
