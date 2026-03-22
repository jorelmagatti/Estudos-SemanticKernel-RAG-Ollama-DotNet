namespace AgenteConsultaRagHitl;

public static class ConsoleRenderer
{
    /// <summary>
    /// Consome o stream e renderiza no console.
    /// Sem spinner — os tokens do LLM já aparecem em tempo real via streaming.
    /// </summary>
    public static async Task<bool> RenderAsync(
        IAsyncEnumerable<GraphEvent> stream,
        string threadId)
    {
        var interrupted = false;
        var printedHeader = false;

        await foreach (var evt in stream)
        {
            switch (evt.Type)
            {
                case GraphEventType.LlmToken:
                    if (!printedHeader)
                    {
                        Console.ForegroundColor = ConsoleColor.Cyan;
                        Console.WriteLine("\n▶ [LLM]");
                        Console.ResetColor();
                        printedHeader = true;
                    }
                    Console.ForegroundColor = ConsoleColor.White;
                    Console.Write(evt.Content);
                    Console.ResetColor();
                    break;

                case GraphEventType.LlmToolDecision:
                    Console.WriteLine();
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine($"\n  ⚡ {evt.Content}");
                    Console.ResetColor();
                    break;

                case GraphEventType.LlmDirectResponse:
                    Console.WriteLine();
                    break;

                case GraphEventType.HumanInterruptRequired:
                    Console.WriteLine();
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"\n  🛑 GRAFO PAUSADO — {evt.Content}");
                    Console.ResetColor();
                    interrupted = true;
                    break;

                case GraphEventType.ToolExecuting:
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine($"\n▶ [ACTION]\n  ⚙️  {evt.Content}");
                    Console.ResetColor();
                    printedHeader = false;
                    break;

                case GraphEventType.ToolResult:
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine($"  ✅ {evt.Content}");
                    Console.ResetColor();
                    // Reseta o header para o próximo nó LLM
                    printedHeader = false;
                    break;

                case GraphEventType.GraphFinished:
                    // Só exibe mensagem final quando vem do nó "llm" (resposta real terminada)
                    // Ignora GraphFinished intermediário do nó "action"
                    if (!interrupted && evt.NodeName == "llm")
                    {
                        Console.ForegroundColor = ConsoleColor.DarkGray;
                        Console.WriteLine($"\n  ✅ [thread:{threadId}] Persistido no SQLite.");
                        Console.ResetColor();
                    }
                    break;
            }
        }

        return interrupted;
    }

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

    public static void PrintBanner()
    {
        Console.ForegroundColor = ConsoleColor.DarkYellow;
        Console.WriteLine("""
 
        ╔══════════════════════════════════════════════════════════════╗
        ║   Human-in-the-Loop (HITL) — Semantic Kernel + Ollama       ║
        ║   RAG + SQLite Checkpoints + StateGraph                      ║
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
        Console.WriteLine($"\n  ── Histórico '{threadId}' ({messages.Count} msgs) ──");
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
                ? msg.Content[..120] + "..." : msg.Content;
            Console.WriteLine($"  {prefix}: {preview}");
            Console.ResetColor();
        }
        Console.WriteLine();
    }
}

public enum HitlDecision { Approve, Cancel, Inject }