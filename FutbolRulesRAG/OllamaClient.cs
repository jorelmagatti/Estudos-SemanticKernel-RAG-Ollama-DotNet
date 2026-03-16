using System.Net.Http.Json;

namespace RaglangChainTestes;

public class OllamaClient(string baseUrl = "http://localhost:11434")
{
    private readonly HttpClient _http = new() { BaseAddress = new Uri(baseUrl), Timeout = TimeSpan.FromMinutes(5) };

    // ──────────────────────────────────────────────
    // Embeddings
    // ──────────────────────────────────────────────

    private record EmbedRequest(string model, string prompt);
    private record EmbedResponse(float[] embedding);

    public async Task<float[]> GetEmbeddingAsync(string text, string model = "nomic-embed-text")
    {
        var response = await _http.PostAsJsonAsync("/api/embeddings", new EmbedRequest(model, text));
        response.EnsureSuccessStatusCode();

        var result = await response.Content.ReadFromJsonAsync<EmbedResponse>()
            ?? throw new InvalidOperationException("Embedding response was null.");

        return result.embedding;
    }

    // ──────────────────────────────────────────────
    // Chat
    // ──────────────────────────────────────────────

    private record ChatMessage(string role, string content);
    private record ChatRequest(string model, ChatMessage[] messages, bool stream);
    private record ChatResponse(ChatMessage message);

    public async Task<string> ChatAsync(
        string systemPrompt,
        string userMessage,
        string model = "llama3.2")
    {
        var messages = new[]
        {
            new ChatMessage("system", systemPrompt),
            new ChatMessage("user", userMessage)
        };

        var response = await _http.PostAsJsonAsync("/api/chat",
            new ChatRequest(model, messages, stream: false));

        response.EnsureSuccessStatusCode();

        var result = await response.Content.ReadFromJsonAsync<ChatResponse>()
            ?? throw new InvalidOperationException("Chat response was null.");

        return result.message.content;
    }
}
