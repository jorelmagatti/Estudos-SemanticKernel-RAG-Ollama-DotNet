using Microsoft.SemanticKernel;
using System.ComponentModel;

namespace AgentEmailRagMultAgentesSemanticKernel;

// ════════════════════════════════════════════════════════════════════════════
//  EmailTools — equivalente aos @tool decorators do notebook Python
//
//  Python:                       C# (SK KernelFunction):
//  @tool                         [KernelFunction]
//  def write_email(...)          WriteEmail(...)
//  def schedule_meeting(...)     ScheduleMeeting(...)
//  def check_calendar(...)       CheckCalendarAvailability(...)
// ════════════════════════════════════════════════════════════════════════════

public class EmailTools
{
    /// <summary>
    /// Escreve e envia um e-mail.
    /// Equivalente ao @tool write_email do notebook Python.
    /// </summary>
    [KernelFunction("write_email")]
    [Description("Escreve e envia um e-mail de resposta em nome do usuário.")]
    public string WriteEmail(
        [Description("Endereço de e-mail do destinatário")] string to,
        [Description("Assunto do e-mail")] string subject,
        [Description("Conteúdo/corpo do e-mail")] string content)
    {
        // Placeholder — em produção integraria com Gmail/Outlook API
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"\n  📧 [TOOL: write_email]");
        Console.WriteLine($"     Para: {to}");
        Console.WriteLine($"     Assunto: {subject}");
        Console.WriteLine($"     Conteúdo: {content[..Math.Min(100, content.Length)]}...");
        Console.ResetColor();

        return $"E-mail enviado para {to} com o assunto '{subject}'";
    }

    /// <summary>
    /// Agenda uma reunião no calendário.
    /// Equivalente ao @tool schedule_meeting do notebook Python.
    /// </summary>
    [KernelFunction("schedule_meeting")]
    [Description("Agenda uma reunião no calendário do usuário.")]
    public string ScheduleMeeting(
        [Description("Lista de participantes (e-mails separados por vírgula)")] string attendees,
        [Description("Assunto/título da reunião")] string subject,
        [Description("Duração da reunião em minutos")] int durationMinutes,
        [Description("Dia preferido para a reunião (ex: segunda-feira, 2026-03-25)")] string preferredDay)
    {
        var attendeeList = attendees.Split(',', StringSplitOptions.RemoveEmptyEntries);

        Console.ForegroundColor = ConsoleColor.DarkCyan;
        Console.WriteLine($"\n  📅 [TOOL: schedule_meeting]");
        Console.WriteLine($"     Assunto: {subject}");
        Console.WriteLine($"     Dia: {preferredDay} | Duração: {durationMinutes}min");
        Console.WriteLine($"     Participantes: {attendeeList.Length}");
        Console.ResetColor();

        return $"Reunião '{subject}' agendada para {preferredDay} com {attendeeList.Length} participante(s) por {durationMinutes} minutos.";
    }

    /// <summary>
    /// Verifica a disponibilidade do calendário.
    /// Equivalente ao @tool check_calendar_availability do notebook Python.
    /// </summary>
    [KernelFunction("check_calendar_availability")]
    [Description("Verifica os horários disponíveis no calendário para um determinado dia.")]
    public string CheckCalendarAvailability(
        [Description("O dia para verificar disponibilidade (ex: segunda-feira, 2026-03-25)")] string day)
    {
        Console.ForegroundColor = ConsoleColor.DarkYellow;
        Console.WriteLine($"\n  🗓️  [TOOL: check_calendar_availability]");
        Console.WriteLine($"     Dia: {day}");
        Console.ResetColor();

        // Simula horários disponíveis
        return $"Horários disponíveis em {day}: 9:00 AM, 2:00 PM, 4:00 PM";
    }
}

