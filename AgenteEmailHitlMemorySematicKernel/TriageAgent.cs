using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.Ollama;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace AgenteEmailHitlMemorySematicKernel;

/// <summary>
/// Agente de triagem — classifica e-mails e roteia para o agente certo.
/// Equivalente ao triage_router() do notebook Python.
/// </summary>
public class TriageAgent
{
    private readonly IChatCompletionService _chat;
    private readonly Kernel _kernel;
    private readonly UserProfile _profile;
    private readonly TriageRules _rules;

    public TriageAgent(Kernel kernel, UserProfile profile, TriageRules rules)
    {
        _kernel = kernel;
        _chat = kernel.GetRequiredService<IChatCompletionService>();
        _profile = profile;
        _rules = rules;
    }

    public async Task<(AgentState, string, string)> RunAsync(AgentState state)
    {
        var email = state.EmailInput;

        var systemPrompt = AgentPrompts.TriageSystem
            .Replace("{triage_no}", _rules.Ignore)
            .Replace("{triage_notify}", _rules.Notify)
            .Replace("{triage_email}", _rules.Respond);

        var userPrompt = AgentPrompts.TriageUser
            .Replace("{author}", email.From)
            .Replace("{to}", email.To)
            .Replace("{subject}", email.Subject)
            .Replace("{email_thread}", email.Body);

        var history = new ChatHistory();
        history.AddSystemMessage(systemPrompt);
        history.AddUserMessage(userPrompt);

#pragma warning disable SKEXP0070
        var settings = new OllamaPromptExecutionSettings { Temperature = 0.0f };
#pragma warning restore SKEXP0070

        var response = await _chat.GetChatMessageContentAsync(
            history, executionSettings: settings, kernel: _kernel);

        var result = ParseResult(response.Content ?? string.Empty);
        state.Triage = result;

        string nextNode, log;

        switch (result.Classification)
        {
            case EmailClassification.Respond:
                log = "📧 RESPONDER — Este e-mail requer uma resposta";
                nextNode = "response_agent";
                state.Messages.Add(new ChatMessage
                {
                    Role = "user",
                    Content = $"Responda ao e-mail:\nDe: {email.From}\nPara: {email.To}\n" +
                              $"Assunto: {email.Subject}\n\n{email.Body}"
                });
                break;

            case EmailClassification.Notify:
                log = "🔔 NOTIFICAR — Informações importantes (sem resposta necessária)";
                nextNode = GraphEnd.Node;
                break;

            default:
                log = "🚫 IGNORAR — E-mail irrelevante";
                nextNode = GraphEnd.Node;
                break;
        }

        return (state, nextNode, log);
    }

    private static TriageResult ParseResult(string text)
    {
        try
        {
            var jsonMatch = Regex.Match(text, @"\{[\s\S]*\}", RegexOptions.Singleline);
            var json = jsonMatch.Success ? jsonMatch.Value : text;
            var doc = JsonDocument.Parse(json);

            var reasoning = doc.RootElement.TryGetProperty("reasoning", out var r)
                ? r.GetString() ?? "" : "";
            var classStr = doc.RootElement.TryGetProperty("classification", out var c)
                ? c.GetString()?.ToLower() ?? "ignore" : "ignore";

            return new TriageResult
            {
                Reasoning = reasoning,
                Classification = classStr switch
                {
                    "respond" => EmailClassification.Respond,
                    "notify" => EmailClassification.Notify,
                    _ => EmailClassification.Ignore
                }
            };
        }
        catch
        {
            var lower = text.ToLower();
            return new TriageResult
            {
                Reasoning = text,
                Classification = lower.Contains("respond") ? EmailClassification.Respond
                               : lower.Contains("notify") ? EmailClassification.Notify
                               : EmailClassification.Ignore
            };
        }
    }
}
