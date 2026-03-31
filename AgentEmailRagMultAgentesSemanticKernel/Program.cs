using AgentEmailRagMultAgentesSemanticKernel;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;

Console.OutputEncoding = System.Text.Encoding.UTF8;
ConsoleRenderer.PrintBanner();

// ── Configuração ──────────────────────────────────────────────────────────────
var ollamaUrl = "http://localhost:11434";
var ollamaModel = "llama3.2";

Console.ForegroundColor = ConsoleColor.DarkGray;
Console.WriteLine($"  Ollama : {ollamaUrl}  |  Modelo: {ollamaModel}");
Console.ResetColor();

// ── Kernel com timeout generoso para Ollama local ─────────────────────────────
var builder = Kernel.CreateBuilder();
builder.Services.AddHttpClient("ollama", c =>
{
    c.BaseAddress = new Uri(ollamaUrl);
    c.Timeout = TimeSpan.FromMinutes(10);
});
builder.AddOllamaChatCompletion(ollamaModel, new Uri(ollamaUrl));
var kernel = builder.Build();

// ── Perfil do usuário — equivalente ao dict profile do notebook ───────────────
var profile = new UserProfile
{
    Name = "Sarah",
    FullName = "Sarah Chen",
    UserProfileBackground = "Engenheira de software sênior liderando uma equipe de 5 desenvolvedores"
};

// ── Regras de triagem — equivalente ao prompt_instructions do notebook ─────────
var triageRules = new TriageRules
{
    Ignore = "Newsletters de marketing, e-mails de spam, comunicados gerais da empresa",
    Notify = "Membro da equipe doente, notificações do sistema de build, atualizações de status de projeto",
    Respond = "Perguntas diretas de membros da equipe, solicitações de reunião, relatórios de bugs críticos"
};

var agentInstructions =
    "Use estas ferramentas quando apropriado para ajudar a gerenciar as tarefas de Sarah de forma eficiente.";

// ── Instancia o orquestrador ──────────────────────────────────────────────────
var orchestrator = new EmailAgentOrchestrator(kernel, profile, triageRules, agentInstructions);

// ── E-mails de teste — equivalentes às células 29-32 do notebook ──────────────
var testEmails = new[]
{
    // E-mail 1: Marketing spam → deve ser IGNORADO (célula 29-30)
    new EmailInput
    {
        Author      = "Equipe de Marketing <marketing@amazingdeals.com>",
        To          = "Sarah Chen <sarah.chen@company.com>",
        Subject     = "🔥 OFERTA EXCLUSIVA: Desconto por Tempo Limitado em Ferramentas para Desenvolvedores! 🔥",
        EmailThread = """
            Prezado(a) Desenvolvedor(a),
 
            Não perca esta oportunidade INCRÍVEL!
 
            🚀 POR TEMPO LIMITADO, obtenha 80% DE DESCONTO em nosso Pacote Premium para Desenvolvedores!
 
            💰 Preço Normal: R$ 999/mês
            🎉 SEU PREÇO ESPECIAL: Apenas R$ 199/mês!
 
            🕒 Corra! Esta oferta expira em: APENAS 24 HORAS!
 
            Clique aqui: https://amazingdeals.com/special-offer
 
            Atenciosamente,
            Equipe de Marketing
            """
    },
 
    // E-mail 2: Dúvida técnica da equipe → deve RESPONDER (célula 31-33)
    new EmailInput
    {
        Author      = "Alice Smith <alice.smith@company.com>",
        To          = "Sarah Chen <sarah.chen@company.com>",
        Subject     = "Dúvida rápida sobre a documentação da API",
        EmailThread = """
            Olá Sarah,
 
            Eu estava revisando a documentação da API para o novo serviço de autenticação
            e notei que alguns endpoints parecem estar faltando nas especificações.
            Você poderia me ajudar a esclarecer se isso foi intencional ou se devemos
            atualizar a documentação?
 
            Especificamente, estou procurando por:
            - /auth/refresh
            - /auth/validate
 
            Obrigada!
            Alice
            """
    },
 
    // E-mail 3: Notificação do sistema de build → deve NOTIFICAR
    new EmailInput
    {
        Author      = "CI/CD Sistema <ci@company.com>",
        To          = "Sarah Chen <sarah.chen@company.com>",
        Subject     = "Build #4521 falhou no branch main",
        EmailThread = """
            Build #4521 falhou.
 
            Branch: main
            Commit: a3f8c91 - "Merge PR #142: adiciona autenticação OAuth"
            Erro: Falha nos testes de integração — 3 testes falharam
 
            Veja os logs em: https://ci.company.com/builds/4521
 
            — CI/CD Bot
            """
    }
};

// ── Loop interativo ───────────────────────────────────────────────────────────
Console.ForegroundColor = ConsoleColor.Green;
Console.WriteLine($"\n  ✅ Pronto! {testEmails.Length} e-mails de teste disponíveis.");
Console.ResetColor();

Console.ForegroundColor = ConsoleColor.DarkGray;
Console.WriteLine("""
  Comandos:
    1, 2, 3    — processar e-mail de teste
    /email     — digitar e-mail manualmente
    /sair      — encerrar
""");
Console.ResetColor();

while (true)
{
    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.Write("\n📧 Escolha [1/2/3] ou comando: ");
    Console.ResetColor();

    var input = Console.ReadLine()?.Trim();
    if (string.IsNullOrWhiteSpace(input)) continue;
    if (input.Equals("/sair", StringComparison.OrdinalIgnoreCase)) break;

    EmailInput? email = null;

    if (input == "1") email = testEmails[0];
    else if (input == "2") email = testEmails[1];
    else if (input == "3") email = testEmails[2];
    else if (input.Equals("/email", StringComparison.OrdinalIgnoreCase))
    {
        Console.Write("  De: "); var from = Console.ReadLine() ?? "";
        Console.Write("  Assunto: "); var subject = Console.ReadLine() ?? "";
        Console.Write("  Corpo (ENTER duas vezes para terminar):\n");
        var lines = new System.Text.StringBuilder();
        string? line;
        while (!string.IsNullOrEmpty(line = Console.ReadLine()))
            lines.AppendLine(line);
        email = new EmailInput
        {
            Author = from,
            To = $"Sarah Chen <sarah.chen@company.com>",
            Subject = subject,
            EmailThread = lines.ToString()
        };
    }

    if (email == null) { Console.WriteLine("  Opção inválida."); continue; }

    ConsoleRenderer.PrintEmailSummary(email);

    try
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var stream = orchestrator.ProcessEmailAsync(email);
        await ConsoleRenderer.RenderAsync(stream);
        sw.Stop();

        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine($"\n  ⏱  Tempo: {sw.Elapsed.TotalSeconds:F1}s");
        Console.ResetColor();
    }
    catch (Exception ex)
    {
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine($"\n  ❌ Erro: {ex.Message}");
        Console.ResetColor();
    }
}

Console.ForegroundColor = ConsoleColor.DarkGray;
Console.WriteLine("\n  Encerrando... Até mais! 👋");
Console.ResetColor();