using FarmaceuticAgentRagSemantickernel;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.InMemory;
using Polly;
using Polly.Retry;
class Program
{
    // ─── Configurações ────────────────────────────────────────────────────────
    const string EMBED_MODEL = "nomic-embed-text"; // 768 dimensões
    const string LLM_MODEL = "llama3.2";          // ou llama3.1, gemma3, etc.
    const string OLLAMA_URL = "http://localhost:11434";
    const int CHUNK_SIZE = 600;                // equivalente ao Python: chunk_size=600
    const int CHUNK_OVERLAP = 150;               // equivalente ao Python: chunk_overlap=150
    const int TOP_K = 4;                  // equivalente ao Python: k=4
    // ─────────────────────────────────────────────────────────────────────────

    static async Task Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;
        Console.WriteLine("╔══════════════════════════════════════════════════╗");
        Console.WriteLine("║    🧪 Agente Farmacêutico RAG — Semantic Kernel  ║");
        Console.WriteLine("╚══════════════════════════════════════════════════╝\n");

        // ══════════════════════════════════════════════════════════════════════
        // ETAPA 1 — Configuração do Kernel com Ollama
        // Equivalente ao Python:
        //   embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        //   llm = ChatOpenAI(model="gpt-4o-mini")
        // ══════════════════════════════════════════════════════════════════════

        // HttpClient com timeout longo (modelos locais são mais lentos)
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

        // ══════════════════════════════════════════════════════════════════════
        // ETAPA 2 — Seleção e organização das bulas
        // Equivalente ao Python:
        //   caminhos_bulas = ["dipirona.pdf", "paracetamol.pdf"]
        //   for caminho in caminhos_bulas: loader = PyPDFLoader(caminho); docs = loader.load()
        // ══════════════════════════════════════════════════════════════════════

        Console.WriteLine("📂 [Etapa 2] Carregando bulas PDF...");

        var caminhosBulas = new[]
        {
            "C:\\Users\\jorel.magatti\\source\\repos\\RaglangChainTestes\\FarmaceuticAgentRagSemantickernel\\arquivos\\dipirona.pdf",
            "C:\\Users\\jorel.magatti\\source\\repos\\RaglangChainTestes\\FarmaceuticAgentRagSemantickernel\\arquivos\\paracetamol.pdf"
            // Adicione mais PDFs aqui conforme necessário
        };

        // Verifica quais arquivos existem
        var bulasFiltradas = caminhosBulas
            .Where(c =>
            {
                if (!File.Exists(c))
                {
                    Console.WriteLine($"  ⚠️  Arquivo não encontrado: {c} (ignorado)");
                    return false;
                }
                return true;
            })
            .ToArray();

        if (bulasFiltradas.Length == 0)
        {
            Console.WriteLine("\n❌ Nenhum PDF encontrado.");
            Console.WriteLine("   Coloque os arquivos dipirona.pdf e paracetamol.pdf");
            Console.WriteLine("   na mesma pasta do executável e tente novamente.\n");
            return;
        }

        var documentos = PdfLoader.CarregarBulas(bulasFiltradas);
        Console.WriteLine($"  ✅ Total de páginas carregadas: {documentos.Count}\n");

        // ══════════════════════════════════════════════════════════════════════
        // ETAPA 3 — Extração, Limpeza e Chunking
        // Equivalente ao Python:
        //   text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
        //   chunks = text_splitter.split_documents(documentos)
        // ══════════════════════════════════════════════════════════════════════

        Console.WriteLine("✂️  [Etapa 3] Dividindo em chunks...");

        var splitter = new TextSplitter(CHUNK_SIZE, CHUNK_OVERLAP);
        var chunksRaw = splitter.SplitDocuments(documentos);
        Console.WriteLine($"  ✅ Total de chunks gerados: {chunksRaw.Count}\n");

        // ══════════════════════════════════════════════════════════════════════
        // ETAPA 4 — Enriquecimento com Metadados (Categorização Semântica)
        // Equivalente ao Python:
        //   for chunk in chunks:
        //       texto = chunk.page_content.lower()
        //       if "identificação do medicamento" in texto:
        //           chunk.metadata["categoria"] = "identificacao"
        //       elif "indicação" in texto: ...
        //       else: chunk.metadata["categoria"] = "geral"
        // ══════════════════════════════════════════════════════════════════════

        Console.WriteLine("🏷️  [Etapa 4] Enriquecendo chunks com metadados...");

        var chunks = MetadataEnricher.Enriquecer(chunksRaw);

        // Exibe distribuição por categoria (como debug)
        var distribuicao = MetadataEnricher.ContarPorCategoria(chunks);
        Console.WriteLine("  Distribuição por categoria:");
        foreach (var (cat, qtd) in distribuicao.OrderByDescending(x => x.Value))
            Console.WriteLine($"    {cat,-30} → {qtd} chunks");

        // Exibe 2 chunks aleatórios (equivalente ao random.sample do Python)
        Console.WriteLine("\n  Exemplos de chunks enriquecidos:");
        var rng = new Random();
        foreach (var chunk in chunks.OrderBy(_ => rng.Next()).Take(2))
        {
            Console.WriteLine($"\n  --- Chunk Aleatório ---");
            Console.WriteLine($"  Medicamento : {chunk.Medicamento}");
            Console.WriteLine($"  Categoria   : {chunk.Categoria}");
            Console.WriteLine($"  Documento   : {chunk.Source}");
            Console.WriteLine($"  Página      : {chunk.Page}");
            Console.WriteLine($"  Conteúdo    : {chunk.Content[..Math.Min(200, chunk.Content.Length)]}...");
        }
        Console.WriteLine();

        // ══════════════════════════════════════════════════════════════════════
        // ETAPA 5 — Geração de Embeddings e Banco Vetorial
        // Equivalente ao Python:
        //   embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        //   vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
        // ══════════════════════════════════════════════════════════════════════

        Console.WriteLine("🔢 [Etapa 5] Gerando embeddings e populando vector store...");
        Console.WriteLine("  (isso pode levar alguns minutos na primeira execução)\n");

#pragma warning disable SKEXP0020
        var vectorStore = new InMemoryVectorStore();
        var colecao = vectorStore.GetCollection<string, DocumentRecord>("bulas");
        await colecao.EnsureCollectionExistsAsync();
#pragma warning restore SKEXP0020

        int processados = 0;
        foreach (var chunk in chunks)
        {
            // Gera embedding para o texto do chunk
            var embResult = await embeddingGenerator.GenerateAsync(chunk.Content);
            var vetor = embResult.Vector;

            await colecao.UpsertAsync(new DocumentRecord
            {
                Id = Guid.NewGuid().ToString(),
                Content = chunk.Content,
                Source = chunk.Source,
                Page = chunk.Page,
                Medicamento = chunk.Medicamento,
                Categoria = chunk.Categoria,
                Embedding = vetor
            });

            processados++;
            if (processados % 10 == 0 || processados == chunks.Count)
                Console.Write($"\r  ✅ {processados}/{chunks.Count} chunks indexados...");
        }

        Console.WriteLine($"\n  ✅ Vector store populado com {chunks.Count} chunks.\n");

        // ══════════════════════════════════════════════════════════════════════
        // ETAPA 6 — Retriever (busca semântica)
        // Equivalente ao Python:
        //   retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        // ══════════════════════════════════════════════════════════════════════

        // Função local de busca semântica (equivalente ao retriever.invoke())
        async Task<List<DocumentRecord>> BuscarChunks(string query)
        {
            var queryEmbResult = await embeddingGenerator.GenerateAsync(query);
            var queryVetor = queryEmbResult.Vector;

            var resultados = new List<DocumentRecord>();

#pragma warning disable SKEXP0020
            await foreach (var r in colecao.SearchAsync(queryVetor, top: TOP_K))
                resultados.Add(r.Record);
#pragma warning restore SKEXP0020

            return resultados;
        }

        // ══════════════════════════════════════════════════════════════════════
        // ETAPA 7 — Pipeline RAG completo
        // Equivalente ao Python:
        //   qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever,
        //                                          return_source_documents=True)
        // ══════════════════════════════════════════════════════════════════════

        // Polly: retry com backoff exponencial (proteção contra timeout do Ollama)
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

        // Prompt do sistema (em inglês para melhor aderência dos modelos locais)
        const string SYSTEM_PROMPT = """
            You are an expert pharmaceutical assistant specializing in Brazilian drug package inserts (bulas).
            Answer the user's question in Brazilian Portuguese.
            Use ONLY the context passages below to formulate your answer.
            If the answer is clearly present in the context, explain it in detail.
            If the answer is NOT in the context, say exactly:
            "Não encontrei essa informação nas bulas fornecidas."
            Do NOT invent information. Do NOT say you don't know if the information IS in the context.
            Read the context carefully before answering.
            
            Context passages:
            {context}
            """;

        // ══════════════════════════════════════════════════════════════════════
        // ETAPA 8 — Loop de perguntas e respostas
        // Equivalente ao Python:
        //   resposta = qa_chain.invoke(pergunta)
        //   print(resposta["result"])
        //   for doc in resposta["source_documents"]: print(doc.metadata, doc.page_content)
        // ══════════════════════════════════════════════════════════════════════

        Console.WriteLine("╔══════════════════════════════════════════════════╗");
        Console.WriteLine("║       💊 Agente Farmacêutico pronto!             ║");
        Console.WriteLine("║  Digite sua pergunta ou 'sair' para encerrar.    ║");
        Console.WriteLine("╚══════════════════════════════════════════════════╝\n");

        while (true)
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.Write("❓ Pergunta: ");
            Console.ResetColor();

            var pergunta = Console.ReadLine()?.Trim();

            if (string.IsNullOrEmpty(pergunta)) continue;
            if (pergunta.Equals("sair", StringComparison.OrdinalIgnoreCase)) break;

            // 1. Recupera chunks relevantes (retriever)
            Console.WriteLine("\n🔍 Buscando trechos relevantes...");
            var trechosRecuperados = await BuscarChunks(pergunta);

            if (trechosRecuperados.Count == 0)
            {
                Console.WriteLine("⚠️  Nenhum trecho relevante encontrado.\n");
                continue;
            }

            // 2. Monta contexto (equivalente ao "stuff" chain type do LangChain)
            var contexto = string.Join("\n\n---\n\n",
                trechosRecuperados.Select((doc, i) =>
                    $"[Trecho {i + 1} | Medicamento: {doc.Medicamento} | " +
                    $"Categoria: {doc.Categoria} | Pág. {doc.Page}]\n{doc.Content}"));

            // 3. Monta o ChatHistory com system prompt + contexto + pergunta
            var chatHistory = new Microsoft.SemanticKernel.ChatCompletion.ChatHistory();
            chatHistory.AddSystemMessage(SYSTEM_PROMPT.Replace("{context}", contexto));
            chatHistory.AddUserMessage(pergunta);

            // 4. Chama o LLM com streaming + retry
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
                Console.WriteLine($"\n❌ Falhou após todas as tentativas: {ex.Message}");
                Console.ResetColor();
            }

            // 5. Exibe trechos utilizados como contexto
            // Equivalente ao Python:
            //   for i, doc in enumerate(resposta["source_documents"], start=1):
            //       print(doc.metadata, doc.page_content)
            Console.WriteLine("\n\n────────────────────────────────────────────────");
            Console.WriteLine("📎 Trechos utilizados como contexto:\n");

            for (int i = 0; i < trechosRecuperados.Count; i++)
            {
                var doc = trechosRecuperados[i];
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"--- Trecho {i + 1} ---");
                Console.ResetColor();
                Console.WriteLine($"Medicamento : {doc.Medicamento}");
                Console.WriteLine($"Categoria   : {doc.Categoria}");
                Console.WriteLine($"Documento   : {doc.Source}");
                Console.WriteLine($"Página      : {doc.Page}");
                Console.WriteLine($"\nConteúdo:");
                Console.WriteLine(doc.Content);
                Console.WriteLine();
            }

            Console.WriteLine("════════════════════════════════════════════════\n");
        }

        Console.WriteLine("\n👋 Encerrando o Agente Farmacêutico. Até logo!");
    }
}
