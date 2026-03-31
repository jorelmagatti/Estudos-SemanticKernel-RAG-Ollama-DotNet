namespace AgenteEmailHitlMemorySematicKernel;

public static class ConsoleRenderer
{
    public static async Task RenderAsync(IAsyncEnumerable<NodeEvent<AgentState>> stream)
    {
        await foreach (var evt in stream)
        {
            if (evt.IsEnd) { PrintFinal(evt.State); break; }
            PrintNode(evt);
        }
    }

    private static void PrintNode(NodeEvent<AgentState> evt)
    {
        var (label, color) = evt.NodeName switch
        {
            "triage_router" => ("🔍 TRIAGE ROUTER", ConsoleColor.Cyan),
            "response_agent" => ("🤖 RESPONSE AGENT", ConsoleColor.Green),
            _ => (evt.NodeName.ToUpper(), ConsoleColor.White)
        };

        Console.ForegroundColor = color;
        Console.WriteLine($"\n{'═'.ToString().PadRight(55, '═')}");
        Console.WriteLine($"  {label}");
        Console.WriteLine('═'.ToString().PadRight(55, '═'));
        Console.ResetColor();

        if (!string.IsNullOrWhiteSpace(evt.Log))
        {
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine($"  {evt.Log.Trim()}");
            Console.ResetColor();
        }

        if (evt.NodeName == "triage_router" && evt.State.Triage != null)
        {
            Console.ForegroundColor = ConsoleColor.DarkGray;
            var r = evt.State.Triage.Reasoning;
            Console.WriteLine($"\n  Raciocínio: {r[..Math.Min(200, r.Length)]}");
            Console.ResetColor();
        }
    }

    private static void PrintFinal(AgentState state)
    {
        if (string.IsNullOrWhiteSpace(state.FinalReply)) return;

        Console.ForegroundColor = ConsoleColor.Magenta;
        Console.WriteLine($"\n{'═'.ToString().PadRight(55, '═')}");
        Console.WriteLine("  📬 RESPOSTA FINAL");
        Console.WriteLine('═'.ToString().PadRight(55, '═'));
        Console.ResetColor();

        Console.ForegroundColor = ConsoleColor.White;
        Console.WriteLine($"\n{state.FinalReply}");
        Console.ResetColor();
    }

    public static void PrintBanner()
    {
        Console.ForegroundColor = ConsoleColor.DarkYellow;
        Console.WriteLine("""
 
        ╔══════════════════════════════════════════════════════════════╗
        ║   Email Agent + Memória — Semantic Kernel + Ollama          ║
        ║                                                              ║
        ║   Agentes: Triage → Response (ReAct + Memória SQLite)       ║
        ║   Ferramentas: email, calendário, manage_memory,            ║
        ║                search_memory                                 ║
        ║   Equivalente: LangGraph + LangMem + InMemoryStore (Python) ║
        ╚══════════════════════════════════════════════════════════════╝
        """);
        Console.ResetColor();
    }

    public static void PrintMemories(List<MemoryEntry> memories, string userId)
    {
        Console.ForegroundColor = ConsoleColor.DarkMagenta;
        Console.WriteLine($"\n  🧠 Memórias do usuário '{userId}' ({memories.Count}):");
        Console.ResetColor();
        foreach (var m in memories)
        {
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine($"  [{m.CreatedAt:dd/MM HH:mm}] {m.Content[..Math.Min(100, m.Content.Length)]}");
            Console.ResetColor();
        }
    }
}