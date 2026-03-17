namespace AgenteUtilidadesRagReActGrafo;

using System.Text.RegularExpressions;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.Ollama;

/// <summary>
/// Agente baseado em grafo de estados — equivalente ao Agent + StateGraph do LangGraph.
///
/// Grafo Python:
///   graph.add_node("llm",    self.call_gemini)
///   graph.add_node("action", self.take_action)
///   graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
///   graph.add_edge("action", "llm")
///   graph.set_entry_point("llm")
///
/// Grafo C# (equivalente):
///
///         ┌─────────────────────────────┐
///         │                             │
///         ▼                             │
///   [ NÓ: llm ] ──── tool_call? ────► [ NÓ: action ]
///         │
///         └── sem tool_call ──► END
///
/// O loop continua até o LLM responder sem pedir ferramenta.
/// </summary>
public class GrafoAgente
{
    private readonly IChatCompletionService _chatService;
    private readonly BuscaWebTool _ferramenta;
    private readonly string _systemPrompt;
    private readonly int _maxIteracoes;

#pragma warning disable SKEXP0070
    private static readonly OllamaPromptExecutionSettings Settings = new()
    {
        Temperature = 0.0f,
    };
#pragma warning restore SKEXP0070

    // Instrução extra injetada no system prompt para que o LLM
    // declare explicitamente quando quer usar a ferramenta.
    // Equivalente ao model.bind_tools(tools) do LangChain.
    private const string INSTRUCAO_FERRAMENTA = """
 
        Quando precisar buscar informações na web, responda EXATAMENTE neste formato:
        TOOL_CALL: buscar_na_web
        QUERY: <sua consulta de busca aqui>
 
        Após receber o resultado da busca (prefixado com "TOOL_RESULT:"),
        use as informações para formular sua resposta final ao usuário.
        Responda em Português Brasileiro.
        """;

    public GrafoAgente(
        IChatCompletionService chatService,
        BuscaWebTool ferramenta,
        string systemPrompt,
        int maxIteracoes = 10)
    {
        _chatService = chatService;
        _ferramenta = ferramenta;
        _systemPrompt = systemPrompt + INSTRUCAO_FERRAMENTA;
        _maxIteracoes = maxIteracoes;
    }

    /// <summary>
    /// Executa o grafo para uma pergunta do usuário.
    /// Equivalente a: result = abot.graph.invoke({"messages": messages})
    /// </summary>
    public async Task<string> InvocarAsync(string perguntaUsuario)
    {
        // ── Inicializa o estado ───────────────────────────────────────────────
        // Equivalente a: state = AgentState(); state['messages'] = [HumanMessage(...)]
        var state = new AgentState();
        state.Messages.AddSystemMessage(_systemPrompt);
        state.Messages.AddUserMessage(perguntaUsuario);

        for (int i = 0; i < _maxIteracoes; i++)
        {
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine($"\n  [Grafo] Iteração {i + 1} — Nó: llm");
            Console.ResetColor();

            // ── NÓ: llm ──────────────────────────────────────────────────────
            // Equivalente a: self.call_gemini(state)
            //   message = self.model.invoke(messages)
            //   return {'messages': [message]}
            var respostaLlm = await _chatService.GetChatMessageContentAsync(
                state.Messages, Settings);

            var textoLlm = respostaLlm.Content?.Trim() ?? string.Empty;

            Console.ForegroundColor = ConsoleColor.DarkCyan;
            Console.WriteLine($"  [llm] → {textoLlm[..Math.Min(200, textoLlm.Length)]}");
            Console.ResetColor();

            // Adiciona resposta do LLM ao histórico
            state.Messages.AddAssistantMessage(textoLlm);

            // ── ARESTA CONDICIONAL: exists_action ─────────────────────────────
            // Equivalente a:
            //   def exists_action(self, state): return len(result.tool_calls) > 0
            //   {True: "action", False: END}
            var (querFerramenta, query) = DetectarChamadaFerramenta(textoLlm);

            if (!querFerramenta)
            {
                // Sem tool_call → END — retorna a resposta final
                return textoLlm;
            }

            // ── NÓ: action ────────────────────────────────────────────────────
            // Equivalente a: self.take_action(state)
            //   result = self.tools[t['name']].invoke(t['args'])
            //   results.append(ToolMessage(..., content=str(result)))
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine($"\n  [Grafo] Nó: action");
            Console.ResetColor();

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"  Calling tool: {_ferramenta.Name} with args: \"{query}\"");
            Console.ResetColor();

            var toolResult = await _ferramenta.BuscarAsync(query);

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("  Back to the model!");
            Console.ResetColor();

            // Injeta o resultado da ferramenta de volta ao histórico
            // para que o LLM o use na próxima iteração
            state.Messages.AddUserMessage($"TOOL_RESULT:\n{toolResult}");

            // Aresta action → llm: loop continua
        }

        return "Erro: limite máximo de iterações atingido sem resposta final.";
    }

    /// <summary>
    /// Detecta se o LLM emitiu uma chamada de ferramenta no formato acordado.
    ///
    /// Equivalente a:
    ///   result.tool_calls  (LangChain detecta automaticamente via bind_tools)
    ///
    /// Como o Ollama local não tem function-calling nativo confiável,
    /// usamos um formato de texto explícito que o LLM foi instruído a seguir.
    /// </summary>
    private static (bool quer, string query) DetectarChamadaFerramenta(string texto)
    {
        // Detecta: TOOL_CALL: buscar_na_web \n QUERY: <texto>
        var match = Regex.Match(
            texto,
            @"TOOL_CALL:\s*buscar_na_web\s*\nQUERY:\s*(.+)",
            RegexOptions.IgnoreCase);

        if (match.Success)
            return (true, match.Groups[1].Value.Trim());

        // Fallback: detecta apenas QUERY: <texto> caso o LLM omita a primeira linha
        var fallback = Regex.Match(texto, @"QUERY:\s*(.+)", RegexOptions.IgnoreCase);
        if (fallback.Success)
            return (true, fallback.Groups[1].Value.Trim());

        return (false, string.Empty);
    }
}
