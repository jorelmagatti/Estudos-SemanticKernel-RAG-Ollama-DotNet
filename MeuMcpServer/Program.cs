using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using ModelContextProtocol.Server;

var builder = WebApplication.CreateBuilder(args);

builder.Services
    .AddMcpServer()
    .WithHttpTransport(options =>
    {
        options.Stateless = true;        // ← Esta linha resolve o problema
    })
    .WithToolsFromAssembly();            // ou .WithTools<ExemploTools>()

var app = builder.Build();

// Health check
app.MapGet("/health", () => "MCP Server está rodando! ✅");

app.MapMcp();       // ou app.MapMcp("/mcp"); se preferir

app.Run();