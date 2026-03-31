using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.Ollama;
using System.Text;

namespace AgenteEmailHitlMemorySematicKernel;

/// <summary>
/// Agente de resposta com memória semântica.
///
/// Equivalente ao create_react_agent(..., store=store) do notebook Python.
/// Além das ferramentas de e-mail, tem acesso a:
///   - manage_memory: salva contexto de interações
///   - search_memory: busca memórias relevantes antes de responder
///
/// O loop ReAct é executado via FunctionChoiceBehavior.Auto().
/// </summary>
public class ResponseAgent
{
    private readonly IChatCompletionService _chat;
    private readonly Kernel _kernel;
    private readonly UserProfile _profile;
    private readonly string _instructions;
    private readonly MemoryPlugin _memPlugin;

    public ResponseAgent(
        Kernel kernel,
        UserProfile profile,
        string instructions,
        MemoryPlugin memPlugin)
    {
        _kernel = kernel;
        _chat = kernel.GetRequiredService<IChatCompletionService>();
        _profile = profile;
        _instructions = instructions;
        _memPlugin = memPlugin;
    }

    public async Task<(AgentState, string, string)> RunAsync(AgentState state)
    {
        // Atualiza o userId no plugin de memória para o contexto atual
        _memPlugin.SetUserId(state.UserId);

        var systemPrompt = AgentPrompts.AgentSystemMemory
            .Replace("{full_name}", _profile.FullName)
            .Replace("{name}", _profile.Name)
            .Replace("{user_profile_background}", _profile.UserProfileBackground)
            .Replace("{instructions}", _instructions);

        var history = new ChatHistory();
        history.AddSystemMessage(systemPrompt);

        foreach (var msg in state.Messages)
        {
            if (msg.Role == "user") history.AddUserMessage(msg.Content);
            else if (msg.Role == "assistant") history.AddAssistantMessage(msg.Content);
        }

#pragma warning disable SKEXP0070
        var settings = new OllamaPromptExecutionSettings
        {
            Temperature = 0.3f,
            FunctionChoiceBehavior = FunctionChoiceBehavior.Auto()
        };
#pragma warning restore SKEXP0070

        // Loop ReAct — equivalente ao ciclo automático do create_react_agent
        var maxIterations = 8;
        var iteration = 0;
        var finalReply = string.Empty;
        var log = new StringBuilder();

        while (iteration++ < maxIterations)
        {
            var response = await _chat.GetChatMessageContentAsync(
                history, executionSettings: settings, kernel: _kernel);

            history.Add(response);

            var functionCalls = FunctionCallContent.GetFunctionCalls(response).ToList();

            if (functionCalls.Count == 0)
            {
                finalReply = response.Content ?? string.Empty;
                log.AppendLine($"✅ Resposta gerada após {iteration} iteração(ões).");
                break;
            }

            log.AppendLine($"  Iteração {iteration}: {functionCalls.Count} ferramenta(s)");

            var toolResults = new ChatMessageContentItemCollection();
            foreach (var fc in functionCalls)
            {
                log.AppendLine($"    → {fc.FunctionName}");
                var result = await fc.InvokeAsync(_kernel);
                toolResults.Add(result);
            }

            history.Add(new ChatMessageContent(AuthorRole.Tool, toolResults));
        }

        if (!string.IsNullOrWhiteSpace(finalReply))
        {
            state.Messages.Add(new ChatMessage { Role = "assistant", Content = finalReply });
            state.FinalReply = finalReply;
        }

        return (state, GraphEnd.Node, log.ToString());
    }
}
