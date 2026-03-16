
// =============================================
// Configurações
// =============================================
using RaglangChainTestes;

const string CAMINHO_PDF = "C:\\Users\\jorel.magatti\\source\\repos\\RaglangChainTestes\\RaglangChainTestes\\arquivos\\regras_futebol.pdf";
const string EMBED_MODEL = "nomic-embed-text"; // ollama pull nomic-embed-text
const string LLM_MODEL = "llama3.2";         // ollama pull llama3.2
const string OLLAMA_URL = "http://localhost:11434";
const int CHUNK_SIZE = 500;
const int CHUNK_OVERLAP = 100;
const int TOP_K = 3;

// =============================================
// 1. Carrega o PDF  (equivalente ao PyPDFLoader)
// =============================================
Console.WriteLine("📄 Carregando PDF...");
var pages = PdfLoader.Load(CAMINHO_PDF);
Console.WriteLine($"   Páginas carregadas: {pages.Count}");

// =============================================
// 2. Divide em chunks  (RecursiveCharacterTextSplitter)
// =============================================
Console.WriteLine("✂️  Dividindo em chunks...");
var chunks = TextSplitter.Split(pages, CHUNK_SIZE, CHUNK_OVERLAP);
Console.WriteLine($"   Chunks criados: {chunks.Count}");

// =============================================
// 3. Gera embeddings com Ollama  (OllamaEmbeddings)
// =============================================
var ollama = new OllamaClient(OLLAMA_URL);
var vectorStore = new InMemoryVectorStore();

Console.WriteLine($"\n🔢 Gerando embeddings com '{EMBED_MODEL}'...");

var embeddedChunks = new List<(float[] Embedding, TextSplitter.Chunk Chunk)>();

for (int i = 0; i < chunks.Count; i++)
{
    if (i % 10 == 0)
        Console.Write($"\r   Processado: {i}/{chunks.Count}");

    var embedding = await ollama.GetEmbeddingAsync(chunks[i].Content, EMBED_MODEL);
    embeddedChunks.Add((embedding, chunks[i]));
}

Console.WriteLine($"\r   Processado: {chunks.Count}/{chunks.Count} ✅");

// =============================================
// 4. Indexa no vector store
// =============================================
vectorStore.AddChunks(embeddedChunks);
Console.WriteLine("🗄️  Indexado no vector store em memória.");

// =============================================
// 5. Loop de perguntas  (RAG Chain)
// =============================================
Console.WriteLine("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
Console.WriteLine("⚽ RAG - Regras do Futebol | Powered by Ollama");
Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
Console.WriteLine("Digite sua pergunta (ou 'sair' para encerrar):\n");

while (true)
{
    Console.Write("❓ Pergunta: ");
    var pergunta = Console.ReadLine()?.Trim();

    if (string.IsNullOrEmpty(pergunta) || pergunta.Equals("sair", StringComparison.OrdinalIgnoreCase))
        break;

    // 5a. Embedding da pergunta
    var queryEmbedding = await ollama.GetEmbeddingAsync(pergunta, EMBED_MODEL);

    // 5b. Retrieval — busca os k chunks mais relevantes
    var contextChunks = vectorStore.SimilaritySearch(queryEmbedding, TOP_K);

    // 5c. Monta o contexto (equivalente ao create_stuff_documents_chain)
    var contexto = string.Join("\n\n---\n\n",
        contextChunks.Select((c, i) => $"[Trecho {i + 1} | Pág. {c.PageNumber}]\n{c.Content}"));

    // 5d. Prompt (equivalente ao ChatPromptTemplate)
    var systemPrompt = $"""
        Você é um árbitro especialista nas regras oficiais de futebol da FIFA/IFAB.
        Use SOMENTE o contexto abaixo para responder.
        Se a informação não estiver no contexto, responda: "Não sei com base nas regras fornecidas."

        Contexto:
        {contexto}
        """;

    // 5e. Chamada ao LLM via Ollama
    Console.WriteLine("\n🤖 Processando...");
    var resposta = await ollama.ChatAsync(systemPrompt, pergunta, LLM_MODEL);

    Console.WriteLine("\n💬 Resposta:");
    Console.WriteLine(resposta);

    // 5f. Exibe trechos utilizados (equivalente ao resposta["context"])
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