using Microsoft.SemanticKernel;
using EssayWriterBlazor.Plugins;
using EssayWriterBlazor.Services;
using EssayWriterBlazor;

var builder = WebApplication.CreateBuilder(args);

// ── Blazor Server ─────────────────────────────────────────────────────────────
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

// ── Configuração do Ollama ────────────────────────────────────────────────────
var ollamaUrl   = "http://localhost:11434";
var ollamaModel = "llama3.2";

// ── Kernel com timeout generoso ───────────────────────────────────────────────
builder.Services.AddHttpClient("ollama", c =>
{
    c.BaseAddress = new Uri(ollamaUrl);
    c.Timeout     = TimeSpan.FromMinutes(10);
});

var kernelBuilder = Kernel.CreateBuilder();
kernelBuilder.Services.AddHttpClient("ollama", c =>
{
    c.BaseAddress = new Uri(ollamaUrl);
    c.Timeout     = TimeSpan.FromMinutes(10);
});
kernelBuilder.AddOllamaChatCompletion(ollamaModel, new Uri(ollamaUrl));
var kernel = kernelBuilder.Build();

builder.Services.AddSingleton(kernel);
builder.Services.AddSingleton<TavilySearchService>();

// EssayWriterService com Scoped para que cada usuário tenha sua instância
builder.Services.AddScoped<EssayWriterService>(sp =>
    new EssayWriterService(
        sp.GetRequiredService<Kernel>(),
        sp.GetRequiredService<TavilySearchService>()));

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();
app.UseAntiforgery();

app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();
