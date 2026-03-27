namespace MultAgentConsultaRagGrafoSemanticKernel;

/// <summary>
/// Renderiza o progresso do pipeline multi-agente no console.
/// Equivalente ao loop: for s in graph.stream({state}, thread): print(s)
/// </summary>
public static class ConsoleRenderer
{
    private static readonly Dictionary<string, (string Label, ConsoleColor Color)> NodeInfo = new()
    {
        ["planner"] = ("📋 PLANNER", ConsoleColor.Cyan),
        ["research_plan"] = ("🔍 RESEARCH PLAN", ConsoleColor.DarkCyan),
        ["generate"] = ("✍️  WRITER", ConsoleColor.Green),
        ["reflect"] = ("🪞 REFLECT", ConsoleColor.Yellow),
        ["research_critique"] = ("🔍 RESEARCH CRITIQUE", ConsoleColor.DarkYellow),
        ["__END__"] = ("🏁 CONCLUÍDO", ConsoleColor.Magenta),
    };

    public static async Task RenderPipelineAsync(
        IAsyncEnumerable<NodeEvent<AgentState>> stream)
    {
        await foreach (var evt in stream)
        {
            if (evt.IsEnd)
            {
                PrintFinalResult(evt.State);
                break;
            }

            PrintNodeEvent(evt);
        }
    }

    private static void PrintNodeEvent(NodeEvent<AgentState> evt)
    {
        var (label, color) = NodeInfo.TryGetValue(evt.NodeName, out var info)
            ? info
            : (evt.NodeName.ToUpper(), ConsoleColor.White);

        Console.ForegroundColor = color;
        Console.WriteLine($"\n{'═'.ToString().PadRight(60, '═')}");
        Console.WriteLine($"  {label}  (revisão {evt.State.RevisionNumber - 1}/{evt.State.MaxRevisions})");
        Console.WriteLine($"{'═'.ToString().PadRight(60, '═')}");
        Console.ResetColor();

        switch (evt.NodeName)
        {
            case "planner":
                PrintSection("Plano gerado", evt.State.Plan, ConsoleColor.Cyan);
                break;

            case "research_plan":
            case "research_critique":
                Console.ForegroundColor = ConsoleColor.DarkGray;
                Console.WriteLine($"  {evt.State.Content.Count} chunk(s) de conteúdo pesquisado acumulados.");
                Console.ResetColor();
                break;

            case "generate":
                PrintSection("Rascunho", evt.State.Draft, ConsoleColor.Green);
                break;

            case "reflect":
                PrintSection("Crítica", evt.State.Critique, ConsoleColor.Yellow);
                break;
        }
    }

    private static void PrintFinalResult(AgentState state)
    {
        Console.ForegroundColor = ConsoleColor.Magenta;
        Console.WriteLine($"\n{'═'.ToString().PadRight(60, '═')}");
        Console.WriteLine($"  🏁 REDAÇÃO FINAL — {state.RevisionNumber - 1} revisão(ões)");
        Console.WriteLine($"{'═'.ToString().PadRight(60, '═')}");
        Console.ResetColor();

        Console.ForegroundColor = ConsoleColor.White;
        Console.WriteLine($"\n{state.Draft}");
        Console.ResetColor();

        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine($"\n  Fontes pesquisadas: {state.Content.Count} chunks do Tavily");
        Console.ResetColor();
    }

    private static void PrintSection(string title, string content, ConsoleColor color)
    {
        if (string.IsNullOrWhiteSpace(content)) return;
        var preview = content.Length > 600
            ? content[..600] + "\n  [... continua ...]"
            : content;
        Console.ForegroundColor = color;
        Console.WriteLine($"\n  {title}:");
        Console.ForegroundColor = ConsoleColor.White;
        Console.WriteLine(preview);
        Console.ResetColor();
    }

    public static void PrintBanner()
    {
        Console.ForegroundColor = ConsoleColor.DarkYellow;
        Console.WriteLine("""
 
        ╔══════════════════════════════════════════════════════════════╗
        ║   Multi-Agent Writer — Semantic Kernel + Ollama + Tavily    ║
        ║                                                              ║
        ║   Agentes: Planner → ResearchPlan → Writer → Reflect        ║
        ║            → ResearchCritique → Writer (loop)               ║
        ║                                                              ║
        ║   Equivalente: LangGraph + Gemini + Tavily (Python)         ║
        ╚══════════════════════════════════════════════════════════════╝
        """);
        Console.ResetColor();
    }

    public static void PrintHelp()
    {
        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine("""
          Comandos:
            /revisoes <n>  — definir número de revisões (padrão: 2)
            /sair          — encerrar
        """);
        Console.ResetColor();
    }
}
