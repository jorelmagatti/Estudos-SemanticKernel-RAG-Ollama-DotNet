using Microsoft.SemanticKernel;
using System.ComponentModel;

namespace AgenteEmailHitlMemorySematicKernel;


public class EmailTools
{
    [KernelFunction("write_email")]
    [Description("Envia um e-mail para o destinatário especificado com o assunto e conteúdo fornecidos.")]
    public string WriteEmail(
        [Description("Endereço de e-mail do destinatário")] string to,
        [Description("Assunto do e-mail")] string subject,
        [Description("Conteúdo do e-mail")] string content)
    {
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"\n  📧 [write_email]");
        Console.WriteLine($"     Para   : {to}");
        Console.WriteLine($"     Assunto: {subject}");
        Console.WriteLine($"     Conteúdo: {content[..Math.Min(120, content.Length)]}");
        Console.ResetColor();
        return $"E-mail enviado para {to} com o assunto '{subject}'";
    }

    [KernelFunction("schedule_meeting")]
    [Description("Agenda uma reunião com os participantes especificados.")]
    public string ScheduleMeeting(
        [Description("Participantes separados por vírgula")] string attendees,
        [Description("Assunto da reunião")] string subject,
        [Description("Duração em minutos")] int durationMinutes,
        [Description("Dia preferido")] string preferredDay)
    {
        var count = attendees.Split(',').Length;
        Console.ForegroundColor = ConsoleColor.DarkCyan;
        Console.WriteLine($"\n  📅 [schedule_meeting]");
        Console.WriteLine($"     Assunto: {subject} | Dia: {preferredDay} | {durationMinutes}min | {count} participante(s)");
        Console.ResetColor();
        return $"Reunião '{subject}' agendada para {preferredDay} com {count} participante(s) por {durationMinutes} minutos.";
    }

    [KernelFunction("check_calendar_availability")]
    [Description("Verifica os horários disponíveis para o dia fornecido.")]
    public string CheckCalendarAvailability(
        [Description("Dia para verificar")] string day)
    {
        Console.ForegroundColor = ConsoleColor.DarkYellow;
        Console.WriteLine($"\n  🗓️  [check_calendar_availability] Dia: {day}");
        Console.ResetColor();
        return $"Horários disponíveis em {day}: 9:00, 14:00, 16:00";
    }
}
