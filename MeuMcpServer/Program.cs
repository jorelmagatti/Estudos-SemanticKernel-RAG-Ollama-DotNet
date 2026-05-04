var builder = WebApplication.CreateBuilder(args);

builder.Services
    .AddMcpServer()
    .WithHttpTransport(options =>
    {
        options.Stateless = true;        
    })
    .WithToolsFromAssembly();            

var app = builder.Build();

app.MapGet("/health", () => "MCP Server está rodando! ✅");

app.MapMcp("/mcp");

app.Run();