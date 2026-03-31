using Microsoft.SemanticKernel.Memory;

namespace AgenteEmailHitlMemorySematicKernel;

/// <summary>
/// Serviço Human-in-the-Loop para decisão de agendamento de reunião.
///
/// Equivalente à função human_in_the_loop_schedule() do notebook Python:
///   1. Busca memória: há reunião já agendada com este remetente?
///   2. Se sim: envia e-mail informando que já foi agendado
///   3. Se não: pergunta ao usuário se quer agendar
///      - Sim: agenda reunião + salva na memória + envia e-mail
///      - Não: envia e-mail de acompanhamento + salva na memória
/// </summary>
public class HumanInTheLoopService
{
    private readonly MemoryStore _store;
    private readonly EmailTools _emailTools;

    public HumanInTheLoopService(MemoryStore store, EmailTools emailTools)
    {
        _store = store;
        _emailTools = emailTools;
    }

    /// <summary>
    /// Executa o fluxo HITL de decisão de agendamento.
    /// Equivalente ao human_in_the_loop_schedule() do notebook.
    /// </summary>
    public async Task RunAsync(EmailInput email, string userId)
    {
        PrintHeader("HUMAN-IN-THE-LOOP");

        // Equivalente ao search_memory_tool.invoke({"query": ...})
        var memoryQuery = $"Reunião agendada para {email.From}";
        var searchResults = _store.Search(userId, memoryQuery);

        if (searchResults.Count > 0)
        {
            // Já há reunião agendada — informa o remetente diretamente
            Console.ForegroundColor = ConsoleColor.DarkMagenta;
            Console.WriteLine($"  🧠 Memória encontrada: reunião já agendada para {email.From}");
            Console.ResetColor();

            var emailResult = _emailTools.WriteEmail(
                to: email.From,
                subject: $"Re: {email.Subject}",
                content: "Olá, acabei de agendar uma conversa contigo para discutirmos esse assunto.");

            PrintToolResult("write_email", emailResult);
            return;
        }

        // Sem memória — pergunta ao humano
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.Write(
            $"\n  ❓ Deseja agendar uma reunião para discutir o pedido de {email.From}? (sim/não): ");
        Console.ResetColor();

        var decision = (Console.ReadLine() ?? string.Empty).Trim().ToLower();

        if (decision is "sim" or "s" or "yes" or "y")
        {
            // Agenda reunião
            var attendees = $"{email.To.Split('<')[0].Trim()},{email.From.Split('<')[0].Trim()}";
            var meetingResult = _emailTools.ScheduleMeeting(
                attendees: attendees,
                subject: $"Acompanhamento do pedido de {email.From}",
                durationMinutes: 30,
                preferredDay: "amanhã");

            PrintToolResult("schedule_meeting", meetingResult);

            // Salva na memória — equivalente ao manage_memory_tool.invoke(...)
            var memResult = _store.Create(userId,
                $"Reunião agendada para discutir o pedido de {email.From}");
            PrintToolResult("manage_memory", memResult);

            // Envia e-mail confirmando
            var emailResult = _emailTools.WriteEmail(
                to: email.From,
                subject: $"Re: {email.Subject}",
                content: "Já agendei uma reunião contigo para discutirmos esse assunto.");

            PrintToolResult("write_email", emailResult);
        }
        else
        {
            // Não quer agendar — envia e-mail de acompanhamento
            var emailResult = _emailTools.WriteEmail(
                to: email.From,
                subject: $"Re: {email.Subject}",
                content: "Estou acompanhando seu pedido e entrarei em contato assim que houver novidades.");

            PrintToolResult("write_email", emailResult);

            // Salva na memória
            var memResult = _store.Create(userId,
                $"E-mail de acompanhamento enviado para {email.From}. Reunião não agendada.");
            PrintToolResult("manage_memory", memResult);
        }
    }

    private static void PrintHeader(string title)
    {
        Console.ForegroundColor = ConsoleColor.Magenta;
        Console.WriteLine($"\n{'═'.ToString().PadRight(55, '═')}");
        Console.WriteLine($"  🤝 {title}");
        Console.WriteLine('═'.ToString().PadRight(55, '═'));
        Console.ResetColor();
    }

    private static void PrintToolResult(string tool, string result)
    {
        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine($"\n  ══ Tool: {tool} ══");
        Console.ForegroundColor = ConsoleColor.White;
        Console.WriteLine($"  {result}");
        Console.ResetColor();
    }
}
