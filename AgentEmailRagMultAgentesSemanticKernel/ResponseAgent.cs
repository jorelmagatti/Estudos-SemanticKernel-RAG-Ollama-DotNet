using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.Ollama;
using System.Text;

namespace AgentEmailRagMultAgentesSemanticKernel;

/// <summary>
/// Agente de resposta — agente ReAct com ferramentas de e-mail e calendário.
///
/// Equivalente ao create_react_agent() do notebook Python:
///   agent = create_react_agent(
///       model=llm,
///       tools=[write_email, schedule_meeting, check_calendar_availability],
///       prompt=create_prompt,
///   )
///
/// O SK executa o loop ReAct automaticamente via FunctionChoiceBehavior.Auto()
/// quando plugins estão registrados no Kernel.
/// </summary>
public class ResponseAgent
{
    private readonly IChatCompletionService _chat;
    private readonly Kernel _kernel;
    private readonly UserProfile _profile;
    private readonly string _agentInstructions;

    public ResponseAgent(Kernel kernel, UserProfile profile, string agentInstructions)
    {
        _kernel = kernel;
        _chat = kernel.GetRequiredService<IChatCompletionService>();
        _profile = profile;
        _agentInstructions = agentInstructions;
    }

    /// <summary>
    /// Executa o agente de resposta com loop ReAct.
    /// Equivalente ao: response = agent.invoke({"messages": [...]})
    /// </summary>
    public async Task<(AgentState, string, string)> RunAsync(AgentState state)
    {
        // Monta o system prompt — equivalente ao create_prompt(state) do notebook
        var systemPrompt = AgentPrompts.AgentSystem
            .Replace("{full_name}", _profile.FullName)
            .Replace("{name}", _profile.Name)
            .Replace("{user_profile_background}", _profile.UserProfileBackground)
            .Replace("{instructions}", _agentInstructions);

        var history = new ChatHistory();
        history.AddSystemMessage(systemPrompt);

        // Adiciona mensagens do estado ao histórico
        foreach (var msg in state.Messages)
        {
            switch (msg.Role)
            {
                case "user": history.AddUserMessage(msg.Content); break;
                case "assistant": history.AddAssistantMessage(msg.Content); break;
            }
        }

#pragma warning disable SKEXP0070
        // FunctionChoiceBehavior.Auto() habilita o loop ReAct:
        // O SK detecta automaticamente quando o LLM quer chamar uma ferramenta,
        // executa-a e volta o resultado ao LLM — equivalente ao create_react_agent
        var settings = new OllamaPromptExecutionSettings
        {
            Temperature = 0.3f,
            FunctionChoiceBehavior = FunctionChoiceBehavior.Auto()
        };
#pragma warning restore SKEXP0070

        // Loop ReAct manual para ter visibilidade de cada passo
        var maxIterations = 6;
        var iteration = 0;
        var finalReply = string.Empty;
        var log = new StringBuilder();

        while (iteration++ < maxIterations)
        {
            var response = await _chat.GetChatMessageContentAsync(
                history, executionSettings: settings, kernel: _kernel);

            history.Add(response);

            // Verifica se há function calls pendentes
            var functionCalls = FunctionCallContent.GetFunctionCalls(response).ToList();

            if (functionCalls.Count == 0)
            {
                // Sem tool calls → resposta final do agente
                finalReply = response.Content ?? string.Empty;
                log.AppendLine($"✅ Resposta final gerada após {iteration} iteração(ões).");
                break;
            }

            // Executa cada ferramenta e injeta o resultado no histórico
            log.AppendLine($"  Iteração {iteration}: {functionCalls.Count} ferramenta(s) chamada(s)");

            var toolResults = new ChatMessageContentItemCollection();
            foreach (var fc in functionCalls)
            {
                log.AppendLine($"    → {fc.FunctionName}({string.Join(", ", fc.Arguments?.Select(a => $"{a.Key}={a.Value}") ?? [])})");

                var result = await fc.InvokeAsync(_kernel);
                toolResults.Add(result);
            }

            history.Add(new ChatMessageContent(AuthorRole.Tool, toolResults));
        }

        // Persiste a resposta no estado
        if (!string.IsNullOrWhiteSpace(finalReply))
        {
            state.Messages.Add(new ChatMessage
            {
                Role = "assistant",
                Content = finalReply
            });
            state.FinalReply = finalReply;
        }

        return (state, GraphEnd.Node, log.ToString());
    }
}