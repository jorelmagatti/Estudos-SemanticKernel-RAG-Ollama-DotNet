using Microsoft.SemanticKernel;

namespace AgentEmailRagMultAgentesSemanticKernel;

/// <summary>
/// Orquestrador do grafo de e-mail.
///
/// Equivalente à montagem do grafo no notebook Python:
///   email_agent = StateGraph(State)
///   email_agent.add_node("triage_router", triage_router)
///   email_agent.add_node("response_agent", agent)
///   email_agent.add_edge(START, "triage_router")
///   email_agent.compile()
///
/// Grafo:
///   START → triage_router ──respond──→ response_agent → END
///                         ──ignore──→ END
///                         ──notify──→ END
/// </summary>
public class EmailAgentOrchestrator
{
    private readonly CompiledGraph<AgentState> _graph;

    public EmailAgentOrchestrator(
        Kernel kernel,
        UserProfile profile,
        TriageRules triageRules,
        string agentInstructions)
    {
        // Registra as ferramentas no kernel
        kernel.Plugins.AddFromObject(new EmailTools(), "EmailTools");

        var triageAgent = new TriageAgent(kernel, profile, triageRules);
        var responseAgent = new ResponseAgent(kernel, profile, agentInstructions);

        // Monta o grafo — equivalente às células 27 do notebook
        _graph = new StateGraph<AgentState>()
            .AddNode("triage_router", triageAgent.RunAsync)
            .AddNode("response_agent", responseAgent.RunAsync)
            .SetEntryPoint("triage_router")
            .Compile();
    }

    /// <summary>
    /// Processa um e-mail através do pipeline multi-agente.
    /// Equivalente ao: response = email_agent.invoke({"email_input": email_input})
    /// </summary>
    public IAsyncEnumerable<NodeEvent<AgentState>> ProcessEmailAsync(EmailInput email) =>
        _graph.StreamAsync(new AgentState { EmailInput = email });
}
