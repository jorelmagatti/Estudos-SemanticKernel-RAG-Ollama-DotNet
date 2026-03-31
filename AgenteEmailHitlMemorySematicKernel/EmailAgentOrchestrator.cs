using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Memory;

namespace AgenteEmailHitlMemorySematicKernel;

/// <summary>
/// Monta e executa o grafo completo com memória.
///
/// Equivalente à célula 5 do notebook:
///   builder = StateGraph(State)
///   builder.add_node("triage_router", triage_router)
///   builder.add_node("response_agent", response_agent)
///   builder.compile()
/// </summary>
public class EmailAgentOrchestrator
{
    private readonly CompiledGraph<AgentState> _graph;
    private readonly HumanInTheLoopService _hitl;
    private readonly MemoryStore _memStore;

    public EmailAgentOrchestrator(
        Kernel kernel,
        UserProfile profile,
        TriageRules triageRules,
        string agentInstructions,
        MemoryStore memStore)
    {
        _memStore = memStore;

        // Plugins registrados no kernel
        var emailTools = new EmailTools();
        var memPlugin = new MemoryPlugin(memStore);

        kernel.Plugins.AddFromObject(emailTools, "EmailTools");
        kernel.Plugins.AddFromObject(memPlugin, "MemoryTools");

        // Agentes
        var triageAgent = new TriageAgent(kernel, profile, triageRules);
        var responseAgent = new ResponseAgent(kernel, profile, agentInstructions, memPlugin);

        // Grafo
        _graph = new StateGraph<AgentState>()
            .AddNode("triage_router", triageAgent.RunAsync)
            .AddNode("response_agent", responseAgent.RunAsync)
            .SetEntryPoint("triage_router")
            .Compile();

        _hitl = new HumanInTheLoopService(memStore, emailTools);
    }

    public IAsyncEnumerable<NodeEvent<AgentState>> ProcessEmailAsync(
        EmailInput email, string userId = "default") =>
        _graph.StreamAsync(new AgentState
        {
            EmailInput = email,
            UserId = userId
        });

    /// <summary>
    /// Executa o fluxo HITL após o pipeline principal.
    /// Equivalente à célula 8: human_in_the_loop_schedule(...)
    /// </summary>
    public async Task RunHitlAsync(EmailInput email, string userId) =>
        await _hitl.RunAsync(email, userId);

    /// <summary>
    /// Pré-popula memória com contexto inicial.
    /// Equivalente ao manage_memory_tool.invoke({action: "create", ...}) da célula 8.
    /// </summary>
    public string SeedMemory(string userId, string content) =>
        _memStore.Create(userId, content);

    public List<MemoryEntry> GetAllMemories(string userId) =>
        _memStore.GetAll(userId);
}