namespace AgentEmailRagMultAgentesSemanticKernel;

public class ConsoleRenderer
{
    public static async Task RenderAsync(IAsyncEnumerable<NodeEvent<AgentState>> stream)
    {
        await foreach (var evt in stream)
        {
            if (evt.IsEnd)
            {
                PrintFinal(evt.State);
                break;
            }

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
        Console.WriteLine($"\n{'═'.ToString().PadRight(60, '═')}");
        Console.WriteLine($"  {label}");
        Console.WriteLine('═'.ToString().PadRight(60, '═'));
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
            Console.WriteLine($"\n  Raciocínio: {evt.State.Triage.Reasoning[..Math.Min(200, evt.State.Triage.Reasoning.Length)]}");
            Console.ResetColor();
        }
    }

    private static void PrintFinal(AgentState state)
    {
        if (string.IsNullOrWhiteSpace(state.FinalReply)) return;

        Console.ForegroundColor = ConsoleColor.Magenta;
        Console.WriteLine($"\n{'═'.ToString().PadRight(60, '═')}");
        Console.WriteLine("  📬 RESPOSTA FINAL GERADA");
        Console.WriteLine('═'.ToString().PadRight(60, '═'));
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
        ║   Email Agent — Semantic Kernel + Ollama                    ║
        ║                                                              ║
        ║   Agentes: Triage Router → Response Agent (ReAct)           ║
        ║   Ferramentas: write_email, schedule_meeting, calendar       ║
        ║   Equivalente: LangGraph + create_react_agent (Python)      ║
        ╚══════════════════════════════════════════════════════════════╝
        """);
        Console.ResetColor();
    }

    public static void PrintEmailSummary(EmailInput email)
    {
        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine($"\n  📨 De: {email.Author}");
        Console.WriteLine($"  📨 Assunto: {email.Subject}");
        Console.ResetColor();
    }
}
