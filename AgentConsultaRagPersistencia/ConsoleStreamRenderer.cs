namespace AgentConsultaRagPersistencia;

/// <summary>
/// Responsável por exibir os eventos de streaming no console de forma visual.
/// Equivalente ao loop `for event in abot.graph.stream(...)` do notebook.
/// </summary>
public static class ConsoleStreamRenderer
{
    /// <summary>
    /// Consome o IAsyncEnumerable de eventos e renderiza no console em tempo real,
    /// nó a nó — exatamente como o notebook exibe `llm: [...]` e `action: [...]`.
    /// </summary>
    public static async Task RenderAsync(
        IAsyncEnumerable<AgentStreamEvent> stream,
        string threadId)
    {
        var currentNode = string.Empty;
        var nodeHasContent = false;

        await foreach (var evt in stream)
        {
            // ── Troca de nó: imprime cabeçalho ────────────────────────────────
            if (evt.Node != currentNode && evt.Node != "final")
            {
                if (nodeHasContent) Console.WriteLine();

                currentNode = evt.Node;
                nodeHasContent = false;

                Console.ForegroundColor = evt.Node switch
                {
                    "llm" => ConsoleColor.Cyan,
                    "tool" => ConsoleColor.Yellow,
                    _ => ConsoleColor.Gray
                };

                var label = evt.Node switch
                {
                    "llm" => "▶ [LLM]",
                    "tool" => "⚙  [TOOL]",
                    _ => $"▶ [{evt.Node.ToUpper()}]"
                };

                Console.WriteLine($"\n{label}");
                Console.ResetColor();
            }

            // ── Evento final ──────────────────────────────────────────────────
            if (evt.Node == "final")
            {
                Console.ForegroundColor = ConsoleColor.DarkGray;
                Console.WriteLine($"\n  ✅ Thread '{threadId}' — resposta persistida no SQLite.");
                Console.ResetColor();
                continue;
            }

            // ── Streaming de conteúdo ─────────────────────────────────────────
            if (!string.IsNullOrEmpty(evt.Content))
            {
                Console.ForegroundColor = evt.Node switch
                {
                    "llm" => ConsoleColor.White,
                    "tool" => ConsoleColor.DarkYellow,
                    _ => ConsoleColor.Gray
                };

                Console.Write(evt.Content);
                Console.ResetColor();
                nodeHasContent = true;
            }
        }

        Console.WriteLine();
    }

    // ── Helpers de exibição ───────────────────────────────────────────────────

    public static void PrintBanner()
    {
        Console.ForegroundColor = ConsoleColor.DarkYellow;
        Console.WriteLine("""
 
        ╔══════════════════════════════════════════════════════════════╗
        ║   RAG com Persistência SQLite + Streaming                    ║
        ║   Semantic Kernel + Ollama Local                             ║
        ║                                                              ║
        ║   Equivalente ao: LangGraph + SqliteSaver + Gemini (Python)  ║
        ╚══════════════════════════════════════════════════════════════╝
        """);
        Console.ResetColor();
    }

    public static void PrintHelp()
    {
        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine("""
          Comandos especiais:
            /thread <id>   — trocar de conversa  (ex: /thread 2)
            /historico     — ver histórico do thread atual
            /threads       — listar todos os threads
            /limpar        — apagar histórico do thread atual
            /sair          — encerrar
        """);
        Console.ResetColor();
    }

    public static void PrintHistory(List<ChatMessage> messages, string threadId)
    {
        Console.ForegroundColor = ConsoleColor.DarkYellow;
        Console.WriteLine($"\n  ── Histórico do thread '{threadId}' ({messages.Count} mensagens) ──");
        Console.ResetColor();

        if (messages.Count == 0)
        {
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine("  (vazio)");
            Console.ResetColor();
            return;
        }

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
                MessageRole.Tool => $"⚙  Tool [{msg.ToolName}]",
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

    public static void PrintThreadList(List<string> threads)
    {
        Console.ForegroundColor = ConsoleColor.DarkYellow;
        Console.WriteLine($"\n  ── Threads existentes ({threads.Count}) ──");
        Console.ResetColor();

        if (threads.Count == 0)
        {
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine("  (nenhum thread ainda)");
        }
        else
        {
            Console.ForegroundColor = ConsoleColor.White;
            foreach (var t in threads)
                Console.WriteLine($"  • {t}");
        }

        Console.ResetColor();
        Console.WriteLine();
    }
}
