using Microsoft.SemanticKernel.ChatCompletion;
using System.Text.RegularExpressions;

namespace AgentUtilRagReActSemantickernel;

/// <summary>
/// Motor do ciclo ReAct: Pensamento → Ação → PAUSA → Observação → Resposta.
///
/// Equivalente à função Python: run_react_agent(pergunta, max_iterations)
///
/// O padrão ReAct funciona assim:
///   1. O LLM recebe o system prompt explicando o ciclo e as ferramentas
///   2. O usuário envia a pergunta
///   3. O LLM responde com Pensamento + Ação + PAUSA
///   4. O código extrai a Ação, executa a ferramenta, obtém a Observação
///   5. A Observação é enviada de volta ao LLM
///   6. O LLM responde com a Resposta final (ou outra Ação se precisar)
///   7. Repete até encontrar "Resposta:" ou atingir max_iterations
/// </summary>
public class ReActAgent(IChatCompletionService chatService, int maxIteracoes = 5)
{
    // Prompt do sistema — equivalente ao PROMPT_REACT do Python
    // Ensina o LLM o formato exato: Pensamento / Ação / PAUSA / Observação / Resposta
    private const string PROMPT_REACT = """
        Você funciona em um ciclo de Pensamento, Ação, Pausa e Observação.
        Ao final do ciclo, você fornece uma Resposta.
        Use "Pensamento" para descrever seu raciocínio.
        Use "Ação" para executar ferramentas - e então retorne "PAUSA".
        A "Observação" será o resultado da ação executada.
 
        Ações disponíveis:
          - consultar_estoque: retorna a quantidade disponível de um item no inventário (ex: "consultar_estoque: teclado")
          - consultar_preco_produto: retorna o preço unitário de um produto (ex: "consultar_preco_produto: mouse gamer")
          - encontrar_produto_mais_caro: retorna o nome e o preço do produto mais caro no inventário (não requer argumentos)
          - calcular_valor_total_lista: calcula o valor total de uma lista de itens separados por vírgula (ex: "calcular_valor_total_lista: teclado, mouse gamer, monitor")
 
        Exemplo:
        Pergunta: Quantos monitores temos em estoque?
        Pensamento: Devo consultar a ação consultar_estoque para saber a quantidade de monitores.
        Ação: consultar_estoque: monitor
        PAUSA
 
        Observação: Temos 75 monitores em estoque.
        Resposta: Há 75 monitores em estoque.
 
        Exemplo:
        Pergunta: Qual é o produto mais caro?
        Pensamento: Preciso usar a ação encontrar_produto_mais_caro para descobrir qual produto tem o maior preço.
        Ação: encontrar_produto_mais_caro
        PAUSA
 
        Observação: O produto mais caro é o(a) monitor com preço de R$ 999.90.
        Resposta: O produto mais caro é o(a) monitor com preço de R$ 999.90.
 
        Exemplo:
        Pergunta: Quanto custa um teclado e um mouse gamer?
        Pensamento: O usuário quer saber o valor total de vários itens. Devo usar a ação calcular_valor_total_lista.
        Ação: calcular_valor_total_lista: teclado, mouse gamer
        PAUSA
 
        Observação: O valor total dos itens encontrados é R$ 249.50.
        Resposta: O valor total do teclado e do mouse gamer é R$ 249.50.
        """;

    // Regex para extrair: Ação: <nome_ferramenta>: <argumento>
    // Equivalente ao re.search(r"Ação:\s*(\w+)(?::\s*([^\n]*))?" ...) do Python
    private static readonly Regex RegexAcao = new(
        @"Ação:\s*(\w+)(?::\s*([^\n]*))?",
        RegexOptions.IgnoreCase | RegexOptions.Compiled
    );

    // Regex para extrair a resposta final
    // Equivalente ao re.search(r"Resposta:\s*(.*)" ..., re.DOTALL) do Python
    private static readonly Regex RegexResposta = new(
        @"Resposta:\s*(.*)",
        RegexOptions.IgnoreCase | RegexOptions.Singleline | RegexOptions.Compiled
    );

    /// <summary>
    /// Executa o ciclo ReAct completo para uma pergunta.
    /// Equivalente a: run_react_agent(pergunta, max_iterations)
    /// </summary>
    public async Task<string> ExecutarAsync(string pergunta)
    {
        // Monta o ChatHistory com o system prompt
        // Equivalente a: chat = model.start_chat(history=[]) + chat.send_message(PROMPT_REACT)
        var historico = new ChatHistory();
        historico.AddSystemMessage(PROMPT_REACT);
        historico.AddUserMessage(pergunta);

        // Prompt corrente que será enviado ao LLM a cada iteração
        // Na primeira iteração é a pergunta; nas seguintes é a Observação
        // Equivalente a: current_prompt = pergunta → depois "Observação: ..."
        string promptCorrente = pergunta;

        for (int i = 0; i < maxIteracoes; i++)
        {
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine($"\n--- Iteração {i + 1} ---");
            Console.ResetColor();

            // Envia ao LLM e obtém resposta
            var resposta = await chatService.GetChatMessageContentAsync(historico);
            var textoResposta = resposta.Content?.Trim() ?? string.Empty;

            Console.WriteLine($"Modelo pensou/respondeu:\n{textoResposta}\n");

            // ── Verifica se o LLM chegou à resposta final ─────────────────────
            // Equivalente ao: if response_match_final: return ...
            var matchResposta = RegexResposta.Match(textoResposta);
            if (matchResposta.Success)
            {
                return matchResposta.Groups[1].Value.Trim();
            }

            // ── Verifica se o LLM emitiu uma Ação ─────────────────────────────
            // Equivalente ao: match = re.search(r"Ação:\s*(\w+)(?::\s*([^\n]*))?", ...)
            var matchAcao = RegexAcao.Match(textoResposta);
            if (matchAcao.Success)
            {
                var nomeFerramenta = matchAcao.Groups[1].Value.Trim();
                var argumento = matchAcao.Groups[2].Success
                    ? matchAcao.Groups[2].Value.Trim()
                    : string.Empty;

                // Executa a ferramenta
                // Equivalente ao bloco if/elif do Python
                var observacao = Ferramentas.Executar(nomeFerramenta, argumento);

                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"Executou ação: {nomeFerramenta}('{argumento}')");
                Console.WriteLine($"Observação: {observacao}\n");
                Console.ResetColor();

                // Adiciona a resposta do LLM e a observação ao histórico
                // Equivalente a: current_prompt = f"Observação: {observacao_da_acao}"
                historico.AddAssistantMessage(textoResposta);
                historico.AddUserMessage($"Observação: {observacao}");
            }
            else
            {
                // LLM não emitiu nem Ação nem Resposta — erro
                // Equivalente ao: return f"Erro: O agente não conseguiu extrair..."
                return $"Erro: O agente não conseguiu extrair uma Ação ou Resposta final " +
                       $"após {i + 1} iterações. Última resposta do modelo: {textoResposta}";
            }
        }

        return "Erro: Limite máximo de iterações atingido sem uma resposta final do agente.";
    }
}

