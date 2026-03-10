using FutbolRulesRAGSemanticKernel;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.InMemory;
using Polly;
using Polly.Extensions.Http;
using Polly.Retry;

// =============================================
// Configurações
// =============================================
const string CAMINHO_PDF = "C:\\Users\\jorel.magatti\\source\\repos\\RaglangChainTestes\\FutbolRulesRAGSemanticKernel\\arquivo\\regras_futebol.pdf";
const string EMBED_MODEL = "nomic-embed-text";   // ollama pull nomic-embed-text
const string LLM_MODEL = "llama3.2";           // ollama pull llama3.2
const string OLLAMA_URL = "http://localhost:11434";
const string COLLECTION = "regras_futebol";
const int CHUNK_SIZE = 500;
const int CHUNK_OVERLAP = 100;
const int TOP_K = 4;

var llmRetryPolicy = new ResiliencePipelineBuilder()
    .AddRetry(new RetryStrategyOptions
    {
        MaxRetryAttempts = 3,
        Delay = TimeSpan.FromSeconds(2),
        BackoffType = DelayBackoffType.Exponential,
        ShouldHandle = new PredicateBuilder()
            .Handle<TaskCanceledException>()
            .Handle<TimeoutException>()
            .Handle<HttpRequestException>(),
        OnRetry = args =>
        {
            Console.WriteLine($"\n⚠️  Tentativa {args.AttemptNumber + 1}/3 falhou. " +
                              $"Aguardando {args.RetryDelay.TotalSeconds}s...");
            return ValueTask.CompletedTask;
        }
    })
    .Build();


var retryPolicy = HttpPolicyExtensions
    .HandleTransientHttpError()          // erros 5xx e IOException
    .Or<TaskCanceledException>()         // timeout do HttpClient
    .Or<TimeoutException>()
    .WaitAndRetryAsync(
        retryCount: 3,
        sleepDurationProvider: attempt => TimeSpan.FromSeconds(Math.Pow(2, attempt)),
        onRetry: (outcome, delay, attempt, _) =>
        {
            Console.WriteLine($"\n⚠️  Tentativa {attempt}/3 falhou ({outcome.Exception?.GetType().Name}). " +
                              $"Aguardando {delay.TotalSeconds}s antes de tentar novamente...");
        });

var builder = Kernel.CreateBuilder();
builder.Services.AddHttpClient("ollama", c =>
{
    c.Timeout = TimeSpan.FromMinutes(10);
})
.ConfigurePrimaryHttpMessageHandler(() => new SocketsHttpHandler
{
    KeepAlivePingPolicy = HttpKeepAlivePingPolicy.WithActiveRequests,
    KeepAlivePingDelay = TimeSpan.FromSeconds(15),
    KeepAlivePingTimeout = TimeSpan.FromSeconds(15),
    PooledConnectionIdleTimeout = TimeSpan.FromMinutes(15),
})
.AddPolicyHandler(retryPolicy);

var httpClient = builder.Services
    .BuildServiceProvider()
    .GetRequiredService<IHttpClientFactory>()
    .CreateClient("ollama");

builder.Services.AddSingleton(httpClient);

builder.Services.ConfigureHttpClientDefaults(b =>
{
    b.ConfigureHttpClient(c => c.Timeout = TimeSpan.FromMinutes(10));
    b.ConfigurePrimaryHttpMessageHandler(() => new SocketsHttpHandler
    {
        KeepAlivePingPolicy = HttpKeepAlivePingPolicy.WithActiveRequests,
        KeepAlivePingDelay = TimeSpan.FromSeconds(15),
        KeepAlivePingTimeout = TimeSpan.FromSeconds(15),
        PooledConnectionIdleTimeout = TimeSpan.FromMinutes(15),
    });
});


builder.AddOllamaEmbeddingGenerator(EMBED_MODEL, new Uri(OLLAMA_URL));
builder.AddOllamaChatCompletion(LLM_MODEL, new Uri(OLLAMA_URL));

var kernel = builder.Build();


var embeddingGenerator = kernel.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();
var chatService = kernel.GetRequiredService<IChatCompletionService>();

// =============================================
// 2. Configura o Vector Store em memória do SK
//    Equivalente ao Chroma do LangChain
// =============================================
var vectorStore = new InMemoryVectorStore();
var collection = vectorStore.GetCollection<string, DocumentRecord>(COLLECTION);
await collection.EnsureCollectionExistsAsync();

// =============================================
// 3. Carrega o PDF  (PyPDFLoader)
// =============================================
Console.WriteLine("📄 Carregando PDF...");
var pages = PdfLoader.Load(CAMINHO_PDF);
Console.WriteLine($"   Páginas carregadas: {pages.Count}");

// =============================================
// 4. Divide em chunks  (RecursiveCharacterTextSplitter)
// =============================================
Console.WriteLine("✂️  Dividindo em chunks...");
var chunks = TextSplitter.Split(pages, CHUNK_SIZE, CHUNK_OVERLAP);
Console.WriteLine($"   Chunks criados: {chunks.Count}");

// =============================================
// 5. Gera embeddings e indexa no vector store
//    O SK permite gerar em batch automaticamente
// =============================================
Console.WriteLine($"\n🔢 Gerando embeddings com '{EMBED_MODEL}'...");

const int BATCH_SIZE = 10;

for (int i = 0; i < chunks.Count; i += BATCH_SIZE)
{
    var batch = chunks.Skip(i).Take(BATCH_SIZE).ToList();

    // Gera embeddings do batch inteiro numa só chamada
    var embeddings = await embeddingGenerator.GenerateAsync(
        batch.Select(c => c.Content).ToList());

    for (int j = 0; j < batch.Count; j++)
    {
        var record = new DocumentRecord
        {
            Id = $"chunk_{i + j}",
            Content = batch[j].Content,
            PageNumber = batch[j].PageNumber,
            Source = batch[j].Source,
            Embedding = embeddings[j].Vector   // .Vector em vez de cast direto
        };

        await collection.UpsertAsync(record);
    }

    Console.Write($"\r   Indexado: {Math.Min(i + BATCH_SIZE, chunks.Count)}/{chunks.Count}");
}

Console.WriteLine($"\r   Indexado: {chunks.Count}/{chunks.Count} ✅");

// =============================================
// 6. Loop de perguntas — RAG Pipeline
// =============================================
Console.WriteLine("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
Console.WriteLine("⚽ RAG - Regras do Futebol | Semantic Kernel + Ollama");
Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
Console.WriteLine("Digite sua pergunta (ou 'sair' para encerrar):\n");

while (true)
{
    Console.Write("❓ Pergunta: ");
    var pergunta = Console.ReadLine()?.Trim();

    if (string.IsNullOrEmpty(pergunta) || pergunta.Equals("sair", StringComparison.OrdinalIgnoreCase))
        break;

    // 6a. Embedding da pergunta com nova API
    var queryEmbedding = await embeddingGenerator.GenerateVectorAsync(pergunta);

    // 6b. Busca vetorial — API abril/2025
    // SearchEmbeddingAsync substituiu VectorizedSearchAsync
    // retorna IAsyncEnumerable<VectorSearchResult<T>> diretamente (sem .Results)
    var contextChunks = new List<DocumentRecord>();
    await foreach (var result in collection.SearchAsync(queryEmbedding, top: TOP_K))
        contextChunks.Add(result.Record);

    // 6c. Monta o contexto
    var contexto = string.Join("\n\n---\n\n",
        contextChunks.Select((c, idx) => $"[Trecho {idx + 1} | Pág. {c.PageNumber}]\n{c.Content}"));

    // 🔍 DIAGNÓSTICO — remova depois que funcionar
    Console.WriteLine($"\n🔍 Chunks recuperados: {contextChunks.Count}");
    foreach (var c in contextChunks)
        Console.WriteLine($"   Pág.{c.PageNumber} | Score | {c.Content[..Math.Min(100, c.Content.Length)]}...");

    // 6d. ChatHistory com system prompt
    var chatHistory = new ChatHistory();

    chatHistory.AddSystemMessage($"""
        Você é um árbitro especialista nas regras oficiais de futebol da FIFA/IFAB.
        Use SOMENTE o contexto abaixo para responder.
        Se a informação não estiver no contexto, responda: "Não sei com base nas regras fornecidas."

        Contexto:
        {contexto}
        """);

    chatHistory.AddUserMessage(pergunta);

    // 6e. Chamada ao LLM
    await llmRetryPolicy.ExecuteAsync(async cancellationToken =>
    {
        Console.WriteLine("\n🤖 Processando...");
        var resposta = await chatService.GetChatMessageContentAsync(chatHistory);
        Console.WriteLine("\n💬 Resposta:");
        Console.WriteLine(resposta.Content);
    });

    // 6f. Trechos utilizados
    Console.WriteLine("\n📚 Trechos utilizados como contexto:");
    for (int i = 0; i < contextChunks.Count; i++)
    {
        var chunk = contextChunks[i];
        Console.WriteLine($"\n--- Trecho {i + 1} ---");
        Console.WriteLine($"Fonte : {chunk.Source}");
        Console.WriteLine($"Página: {chunk.PageNumber}");
        Console.WriteLine("Conteúdo:");
        Console.WriteLine(chunk.Content.Trim());
        Console.WriteLine(new string('-', 80));
    }

    Console.WriteLine();
}

Console.WriteLine("\n👋 Até logo!");
