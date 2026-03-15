using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.InMemory;
using Polly;
using Polly.Retry;
using RhAgentRagSemanticKernel;

namespace RagRH;

class Program
{
    // ─── Configurações ────────────────────────────────────────────────────────
    const string EMBED_MODEL = "nomic-embed-text";  // 768 dims
    const string LLM_MODEL = "llama3.2";           // ou llama3.1, gemma3
    const string OLLAMA_URL = "http://localhost:11434";
    const int CHUNK_SIZE = 800;   // equivalente ao Python
    const int CHUNK_OVERLAP = 150;   // equivalente ao Python
    const int TOP_K_INICIAL = 8;     // busca vetorial: recupera mais para o reranking
    const int TOP_K_FINAL = 4;     // após reranking: seleciona os melhores
    // ─────────────────────────────────────────────────────────────────────────

    static async Task Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        Console.WriteLine("╔══════════════════════════════════════════════════╗");
        Console.WriteLine("║      🤖 Agente de RH — RAG + Reranking          ║");
        Console.WriteLine("║         Semantic Kernel + Ollama                 ║");
        Console.WriteLine("╚══════════════════════════════════════════════════╝\n");

        // ══════════════════════════════════════════════════════════════════════
        // CONFIGURAÇÃO DO KERNEL
        // Equivalente ao Python:
        //   embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        //   llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        // ══════════════════════════════════════════════════════════════════════

        var httpClient = new HttpClient(new SocketsHttpHandler
        {
            KeepAlivePingPolicy = HttpKeepAlivePingPolicy.WithActiveRequests,
            KeepAlivePingDelay = TimeSpan.FromSeconds(15),
            KeepAlivePingTimeout = TimeSpan.FromSeconds(15),
        })
        { Timeout = TimeSpan.FromMinutes(10) };

        var builder = Kernel.CreateBuilder();
        builder.Services.AddSingleton(httpClient);

#pragma warning disable SKEXP0070
        builder.AddOllamaEmbeddingGenerator(EMBED_MODEL, new Uri(OLLAMA_URL));
        builder.AddOllamaChatCompletion(LLM_MODEL, new Uri(OLLAMA_URL));
#pragma warning restore SKEXP0070

        var kernel = builder.Build();

        var embeddingGenerator = kernel.Services
            .GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();
        var chatService = kernel.Services
            .GetRequiredService<IChatCompletionService>();

        var reranker = new Reranker(chatService);

        // ══════════════════════════════════════════════════════════════════════
        // ETAPA 3 — LEITURA DOS DOCUMENTOS
        // Equivalente ao Python: carregar_documentos()
        // ══════════════════════════════════════════════════════════════════════

        Console.WriteLine("📂 [Etapa 3] Carregando documentos de RH...");

        var caminhos = new[]
        {
            "C:\\Projetos\\ESTUDOS\\SemanticKernel-RAG-Ollama-DotNet\\RhAgentRagSemanticKernel\\arquivos\\politica_ferias.pdf",
            "C:\\Projetos\\ESTUDOS\\SemanticKernel-RAG-Ollama-DotNet\\RhAgentRagSemanticKernel\\arquivos\\politica_home_office.pdf",
            "C:\\Projetos\\ESTUDOS\\SemanticKernel-RAG-Ollama-DotNet\\RhAgentRagSemanticKernel\\arquivos\\codigo_conduta.pdf"
        };

        var documentos = PdfLoader.Carregar(caminhos);

        if (documentos.Count == 0)
        {
            Console.WriteLine("\n❌ Nenhum PDF encontrado.");
            Console.WriteLine("   Coloque os arquivos PDF na mesma pasta do executável.\n");
            Console.WriteLine("   PDFs esperados:");
            foreach (var c in caminhos) Console.WriteLine($"   - {c}");
            return;
        }

        Console.WriteLine($"  ✅ {documentos.Count} páginas carregadas.\n");

        // ══════════════════════════════════════════════════════════════════════
        // ETAPA 4 — CHUNKING
        // Equivalente ao Python: gerar_chunks(documentos)
        //   splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        // ══════════════════════════════════════════════════════════════════════

        Console.WriteLine("✂️  [Etapa 4] Dividindo em chunks...");
        var splitter = new TextSplitter(CHUNK_SIZE, CHUNK_OVERLAP);
        var chunksRaw = splitter.SplitDocuments(documentos);
        Console.WriteLine($"  ✅ {chunksRaw.Count} chunks gerados.\n");

        // ══════════════════════════════════════════════════════════════════════
        // ETAPA 5 — ENRIQUECIMENTO COM METADADOS
        // Equivalente ao Python: enriquecer_chunks(chunks)
        //   if "férias" in texto: chunk.metadata["categoria"] = "ferias"
        //   elif "home office" in texto: ...
        // ══════════════════════════════════════════════════════════════════════

        Console.WriteLine("🏷️  [Etapa 5] Classificando chunks por categoria...");
        var chunks = MetadataEnricher.Enriquecer(chunksRaw);

        var distribuicao = MetadataEnricher.ContarPorCategoria(chunks);
        Console.WriteLine("  Distribuição:");
        foreach (var (cat, qtd) in distribuicao.OrderByDescending(x => x.Value))
            Console.WriteLine($"    {cat,-20} → {qtd} chunks");
        Console.WriteLine();

        // ══════════════════════════════════════════════════════════════════════
        // ETAPA 6 — VECTOR STORE
        // Equivalente ao Python: criar_vectorstore(_chunks)
        //   vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
        // ══════════════════════════════════════════════════════════════════════

        Console.WriteLine("🔢 [Etapa 6] Gerando embeddings e populando vector store...");

#pragma warning disable SKEXP0020
        var vectorStore = new InMemoryVectorStore();
        var colecao = vectorStore.GetCollection<string, DocumentRecord>("rh_docs");
        await colecao.EnsureCollectionExistsAsync();
#pragma warning restore SKEXP0020

        int idx = 0;
        foreach (var chunk in chunks)
        {
            var embResult = await embeddingGenerator.GenerateAsync(chunk.Content);

            await colecao.UpsertAsync(new DocumentRecord
            {
                Id = Guid.NewGuid().ToString(),
                Content = chunk.Content,
                Documento = chunk.Documento,
                Page = chunk.Page,
                Categoria = chunk.Categoria,
                Embedding = embResult.Vector
            });

            idx++;
            if (idx % 10 == 0 || idx == chunks.Count)
                Console.Write($"\r  ✅ {idx}/{chunks.Count} chunks indexados...");
        }

        Console.WriteLine($"\n  ✅ Vector store pronto.\n");

        // Polly retry para chamadas ao LLM
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
                    Console.WriteLine($"\n  ⚠️  Timeout — tentativa {args.AttemptNumber + 1}/3...");
                    return ValueTask.CompletedTask;
                }
            })
            .Build();

        // Prompt final do agente de RH
        // Equivalente ao prompt_final do Python
        const string SYSTEM_PROMPT = """
            You are a corporate HR agent.
            Answer in Brazilian Portuguese.
            Answer ONLY based on the internal policies provided in the context below.
            If the answer is not in the context, say exactly:
            "Não encontrei essa informação nas políticas internas fornecidas."
            Do NOT invent or assume information beyond what is in the context.
            
            Context:
            {context}
            """;

        // ══════════════════════════════════════════════════════════════════════
        // LOOP PRINCIPAL — PERGUNTAS E RESPOSTAS
        // Equivalente ao Python: responder_pergunta(pergunta, vectorstore)
        //   1. similarity_search(k=8)        → busca vetorial inicial
        //   2. rerank_documentos(...)         → reranking com LLM
        //   3. contexto_final = rerankeados[:4]
        //   4. llm.invoke(prompt_final)       → geração da resposta
        // ══════════════════════════════════════════════════════════════════════

        Console.WriteLine("╔══════════════════════════════════════════════════╗");
        Console.WriteLine("║     💼 Agente de RH pronto!                      ║");
        Console.WriteLine("║  Digite sua pergunta ou 'sair' para encerrar.    ║");
        Console.WriteLine("╚══════════════════════════════════════════════════╝\n");

        Console.WriteLine("💡 Exemplos de perguntas:");
        Console.WriteLine("   - Quais são as regras para concessão de férias?");
        Console.WriteLine("   - Quem pode trabalhar em home office?");
        Console.WriteLine("   - Quais comportamentos são inadequados segundo o código de conduta?\n");

        while (true)
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.Write("❓ Pergunta: ");
            Console.ResetColor();

            var pergunta = Console.ReadLine()?.Trim();
            if (string.IsNullOrEmpty(pergunta)) continue;
            if (pergunta.Equals("sair", StringComparison.OrdinalIgnoreCase)) break;

            // ── ETAPA 7a: Busca vetorial inicial (top-K alto) ─────────────────
            // Equivalente ao Python:
            //   documentos_recuperados = vectorstore.similarity_search(pergunta, k=8)
            Console.WriteLine($"\n🔍 Buscando os {TOP_K_INICIAL} chunks mais similares...");

            var queryEmb = await embeddingGenerator.GenerateAsync(pergunta);
            var queryVec = queryEmb.Vector;

            var recuperados = new List<DocumentRecord>();

#pragma warning disable SKEXP0020
            await foreach (var r in colecao.SearchAsync(queryVec, top: TOP_K_INICIAL))
                recuperados.Add(r.Record);
#pragma warning restore SKEXP0020

            if (recuperados.Count == 0)
            {
                Console.WriteLine("⚠️  Nenhum chunk encontrado.\n");
                continue;
            }

            // ── ETAPA 7b: Reranking semântico com LLM ─────────────────────────
            // Equivalente ao Python:
            //   documentos_rerankeados = rerank_documentos(pergunta, documentos_recuperados, llm)
            //   contexto_final = documentos_rerankeados[:4]
            List<(double Score, DocumentRecord Doc)> rerankeados = [];

            try
            {
                await llmRetryPolicy.ExecuteAsync(async ct =>
                {
                    rerankeados = await reranker.RerankAsync(pergunta, recuperados, TOP_K_FINAL);
                });
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"\n❌ Reranking falhou: {ex.Message}");
                Console.WriteLine("   Usando ordem original da busca vetorial...");
                Console.ResetColor();

                // Fallback: usa os primeiros TOP_K_FINAL sem reranking
                rerankeados = recuperados
                    .Take(TOP_K_FINAL)
                    .Select(d => (0.0, d))
                    .ToList();
            }

            // ── ETAPA 7c: Geração da resposta ─────────────────────────────────
            // Equivalente ao Python:
            //   contexto_texto = "\n\n".join([doc.page_content for doc in contexto_final])
            //   resposta = llm.invoke(prompt_final)
            var contexto = string.Join("\n\n---\n\n",
                rerankeados.Select((r, i) =>
                    $"[Trecho {i + 1} | Doc: {r.Doc.Documento} | " +
                    $"Cat: {r.Doc.Categoria} | Score: {r.Score:F1}]\n{r.Doc.Content}"));

            var chatHistory = new ChatHistory();
            chatHistory.AddSystemMessage(SYSTEM_PROMPT.Replace("{context}", contexto));
            chatHistory.AddUserMessage(pergunta);

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("\n🤖 Resposta:\n");
            Console.ResetColor();

            try
            {
                await llmRetryPolicy.ExecuteAsync(async cancellationToken =>
                {
                    await foreach (var chunk in chatService.GetStreamingChatMessageContentsAsync(
                        chatHistory, cancellationToken: cancellationToken))
                    {
                        Console.Write(chunk.Content);
                    }
                });
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"\n❌ Falhou: {ex.Message}");
                Console.ResetColor();
            }

            // ── Exibe fontes utilizadas ───────────────────────────────────────
            // Equivalente ao Python (Streamlit):
            //   st.subheader("Fontes utilizadas")
            //   for doc in fontes: st.write(doc.metadata, doc.page_content)
            Console.WriteLine("\n\n────────────────────────────────────────────────");
            Console.WriteLine("📎 Fontes utilizadas (após reranking):\n");

            for (int i = 0; i < rerankeados.Count; i++)
            {
                var (score, doc) = rerankeados[i];
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"--- Trecho {i + 1} (Score: {score:F1}/10) ---");
                Console.ResetColor();
                Console.WriteLine($"Documento  : {doc.Documento}");
                Console.WriteLine($"Categoria  : {doc.Categoria}");
                Console.WriteLine($"Página     : {doc.Page}");
                Console.WriteLine($"\nConteúdo:");
                Console.WriteLine(doc.Content);
                Console.WriteLine();
            }

            Console.WriteLine("════════════════════════════════════════════════\n");
        }

        Console.WriteLine("\n👋 Encerrando o Agente de RH. Até logo!");
    }
}