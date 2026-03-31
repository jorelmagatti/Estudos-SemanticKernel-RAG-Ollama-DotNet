using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.Ollama;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace AgentEmailRagMultAgentesSemanticKernel;

/// <summary>
/// Agente de triagem — classifica o e-mail e decide o próximo nó.
///
/// Equivalente ao triage_router() do notebook Python:
///   result = llm_router.invoke([system, user])
///   if result.classification == "respond": goto = "response_agent"
///   elif result.classification == "ignore": goto = END
///   elif result.classification == "notify": goto = END
///   return Command(goto=goto, update=update)
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

    /// <summary>
    /// Executa a triagem e retorna (novo estado, próximo nó, log).
    /// Assinatura compatível com o StateGraph.AddNode().
    /// </summary>
    public async Task<(AgentState, string, string)> RunAsync(AgentState state)
    {
        var email = state.EmailInput;

        // Monta os prompts — equivalente ao format() do Python
        var systemPrompt = AgentPrompts.TriageSystem
            .Replace("{full_name}", _profile.FullName)
            .Replace("{name}", _profile.Name)
            .Replace("{user_profile_background}", _profile.UserProfileBackground)
            .Replace("{triage_no}", _rules.Ignore)
            .Replace("{triage_notify}", _rules.Notify)
            .Replace("{triage_email}", _rules.Respond);

        var userPrompt = AgentPrompts.TriageUser
            .Replace("{author}", email.Author)
            .Replace("{to}", email.To)
            .Replace("{subject}", email.Subject)
            .Replace("{email_thread}", email.EmailThread);

        var history = new ChatHistory();
        history.AddSystemMessage(systemPrompt);
        history.AddUserMessage(userPrompt);

#pragma warning disable SKEXP0070
        var settings = new OllamaPromptExecutionSettings { Temperature = 0.0f };
#pragma warning restore SKEXP0070

        var response = await _chat.GetChatMessageContentAsync(
            history, executionSettings: settings, kernel: _kernel);

        var result = ParseTriageResult(response.Content ?? string.Empty);

        // Log e decisão de roteamento — equivalente ao Command(goto=...) do LangGraph
        string log;
        string nextNode;

        switch (result.Classification)
        {
            case EmailClassification.Respond:
                log = "📧 Classificação: RESPONDER — Este e-mail requer uma resposta";
                nextNode = "response_agent";

                // Adiciona o e-mail como mensagem para o agente de resposta
                state.Messages.Add(new ChatMessage
                {
                    Role = "user",
                    Content = $"Responda ao seguinte e-mail em nome de {_profile.FullName}:\n\n" +
                              $"De: {email.Author}\n" +
                              $"Para: {email.To}\n" +
                              $"Assunto: {email.Subject}\n\n" +
                              $"{email.EmailThread}"
                });
                break;

            case EmailClassification.Ignore:
                log = "🚫 Classificação: IGNORAR — Este e-mail pode ser ignorado com segurança";
                nextNode = GraphEnd.Node;
                break;

            case EmailClassification.Notify:
                log = "🔔 Classificação: NOTIFICAR — Este e-mail contém informações importantes";
                nextNode = GraphEnd.Node;
                break;

            default:
                throw new InvalidOperationException($"Classificação inválida: {result.Classification}");
        }

        state.Triage = result;
        return (state, nextNode, log);
    }

    // ── Parser do JSON retornado pelo LLM ─────────────────────────────────────

    private static TriageResult ParseTriageResult(string text)
    {
        try
        {
            // Extrai JSON do texto (modelo pode incluir texto antes/depois)
            var jsonMatch = Regex.Match(text, @"\{[\s\S]*\}", RegexOptions.Singleline);
            var json = jsonMatch.Success ? jsonMatch.Value : text;

            var doc = JsonDocument.Parse(json);

            var reasoning = doc.RootElement.TryGetProperty("reasoning", out var r)
                ? r.GetString() ?? string.Empty : string.Empty;

            var classStr = doc.RootElement.TryGetProperty("classification", out var c)
                ? c.GetString()?.ToLowerInvariant() ?? "ignore" : "ignore";

            var classification = classStr switch
            {
                "respond" => EmailClassification.Respond,
                "notify" => EmailClassification.Notify,
                _ => EmailClassification.Ignore
            };

            return new TriageResult { Reasoning = reasoning, Classification = classification };
        }
        catch
        {
            // Fallback: tenta detectar a classificação no texto livre
            var lower = text.ToLowerInvariant();
            var classification =
                lower.Contains("respond") ? EmailClassification.Respond :
                lower.Contains("notify") ? EmailClassification.Notify :
                                            EmailClassification.Ignore;

            return new TriageResult
            {
                Reasoning = text,
                Classification = classification
            };
        }
    }
}
