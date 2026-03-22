namespace AgenteConsultaRagHitl;

/// <summary>
/// Renderiza eventos do grafo no console em tempo real.
/// Equivalente ao loop `for event in abot.graph.stream(...)` do notebook.
/// </summary>
public static class ConsoleRenderer
{
    /// <summary>
    /// Consome o stream de eventos e exibe no console nó a nó.
    /// Retorna true se o grafo foi interrompido (HITL requerido).
    /// </summary>
    public static async Task<bool> RenderAsync(
        IAsyncEnumerable<GraphEvent> stream,
        string threadId)
    {
        var interrupted = false;
        var currentNode = string.Empty;
        var printedHeader = false;

        await foreach (var evt in stream)
        {
            switch (evt.Type)
            {
                // ── LLM streaming token a token ───────────────────────────────
                case GraphEventType.LlmToken:
                    if (!printedHeader)
                    {
                        PrintNodeHeader("llm", ConsoleColor.Cyan);
                        printedHeader = true;
                        currentNode = "llm";
                    }
                    Console.ForegroundColor = ConsoleColor.White;
                    Console.Write(evt.Content);
                    Console.ResetColor();
                    break;

                // ── LLM decidiu chamar ferramenta ─────────────────────────────
                case GraphEventType.LlmToolDecision:
                    Console.WriteLine();
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine($"\n  ⚡ Agente quer chamar: {evt.Content}");
                    Console.ResetColor();
                    break;

                // ── LLM respondeu diretamente (sem ferramenta) ────────────────
                case GraphEventType.LlmDirectResponse:
                    // conteúdo já foi impresso token a token
                    Console.WriteLine();
                    break;

                // ── HITL: grafo pausado, aguardando aprovação humana ──────────
                // Equivalente ao interrupt_before=["action"] do LangGraph
                case GraphEventType.HumanInterruptRequired:
                    Console.WriteLine();
                    PrintHitlBanner(evt.Content);
                    interrupted = true;
                    break;

                // ── Ferramenta executada ──────────────────────────────────────
                case GraphEventType.ToolExecuting:
                    PrintNodeHeader("action", ConsoleColor.Yellow);
                    Console.ForegroundColor = ConsoleColor.DarkYellow;
                    Console.WriteLine($"  ⚙️  {evt.Content}");
                    Console.ResetColor();
                    currentNode = "action";
                    printedHeader = false;
                    break;

                case GraphEventType.ToolResult:
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine($"  ✅ {evt.Content}");
                    Console.ResetColor();
                    // Novo header para a próxima saída do LLM
                    printedHeader = false;
                    break;

                // ── Fim ───────────────────────────────────────────────────────
                case GraphEventType.GraphFinished:
                    if (!interrupted)
                    {
                        Console.ForegroundColor = ConsoleColor.DarkGray;
                        Console.WriteLine($"\n  ✅ [thread:{threadId}] Resposta persistida no SQLite.");
                        Console.ResetColor();
                    }
                    break;
            }
        }

        return interrupted;
    }

    // ── Prompt HITL ───────────────────────────────────────────────────────────

    /// <summary>
    /// Exibe o menu HITL e retorna a escolha do usuário.
    /// Equivalente ao input("Você deseja executar esta ação? (sim/não)") do notebook.
    /// </summary>
    public static HitlDecision PromptHitlDecision(string pendingAction)
    {
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine("""
 
        ╔══════════════════════════════════════════════════════════════╗
        ║              ⚠️  INTERVENÇÃO HUMANA NECESSÁRIA               ║
        ╚══════════════════════════════════════════════════════════════╝
        """);
        Console.ResetColor();

        Console.ForegroundColor = ConsoleColor.White;
        Console.WriteLine($"  Ação pendente: {pendingAction}");
        Console.ResetColor();

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("""
 
          O que deseja fazer?
            [1] Aprovar e executar a ação
            [2] Cancelar (descarta a ação)
            [3] Injetar resposta manual (update_state)
        """);
        Console.Write("  Sua escolha: ");
        Console.ResetColor();

        var input = Console.ReadLine()?.Trim();
        return input switch
        {
            "1" or "sim" or "s" => HitlDecision.Approve,
            "3" => HitlDecision.Inject,
            _ => HitlDecision.Cancel
        };
    }

    public static string PromptInjectedResponse()
    {
        Console.ForegroundColor = ConsoleColor.Magenta;
        Console.Write("\n  Digite a resposta a ser injetada: ");
        Console.ResetColor();
        return Console.ReadLine()?.Trim() ?? string.Empty;
    }

    // ── Helpers de exibição ───────────────────────────────────────────────────

    public static void PrintBanner()
    {
        Console.ForegroundColor = ConsoleColor.DarkYellow;
        Console.WriteLine("""
 
        ╔══════════════════════════════════════════════════════════════╗
        ║   Human-in-the-Loop (HITL) — Semantic Kernel + Ollama       ║
        ║   RAG + SQLite Checkpoints + StateGraph                      ║
        ║                                                              ║
        ║   Equivalente: LangGraph + interrupt_before + update_state   ║
        ╚══════════════════════════════════════════════════════════════╝
        """);
        Console.ResetColor();
    }

    public static void PrintHelp()
    {
        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine("""
          Comandos:
            /thread <id>   — trocar de conversa
            /historico     — ver histórico do thread atual
            /threads       — listar todos os threads
            /limpar        — apagar thread atual
            /sair          — encerrar
        """);
        Console.ResetColor();
    }

    public static void PrintHistory(List<ChatMessage> messages, string threadId)
    {
        Console.ForegroundColor = ConsoleColor.DarkYellow;
        Console.WriteLine($"\n  ── Histórico do thread '{threadId}' ({messages.Count} msgs) ──");
        Console.ResetColor();

        foreach (var msg in messages)
        {
            Console.ForegroundColor = msg.Role switch
            {
                MessageRole.User => ConsoleColor.Cyan,
                MessageRole.Assistant => ConsoleColor.White,
                MessageRole.Tool => ConsoleColor.DarkYellow,
                _ => ConsoleColor.Gray
            };
            var prefix = msg.Role switch
            {
                MessageRole.User => "👤 Você",
                MessageRole.Assistant => "🤖 Agente",
                MessageRole.Tool => $"⚙  [{msg.ToolName}]",
                _ => msg.Role.ToString()
            };
            var preview = msg.Content.Length > 120
                ? msg.Content[..120] + "..."
                : msg.Content;
            Console.WriteLine($"  {prefix}: {preview}");
            Console.ResetColor();
        }
        Console.WriteLine();
    }

    private static void PrintNodeHeader(string node, ConsoleColor color)
    {
        Console.ForegroundColor = color;
        Console.WriteLine($"\n▶ [{node.ToUpper()}]");
        Console.ResetColor();
    }

    private static void PrintHitlBanner(string content)
    {
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine($"\n  🛑 GRAFO PAUSADO — {content}");
        Console.ResetColor();
    }
}

public enum HitlDecision { Approve, Cancel, Inject }
