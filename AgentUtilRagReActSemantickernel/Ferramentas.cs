namespace AgentUtilRagReActSemantickernel;


/// <summary>
/// Ferramentas disponíveis para o agente ReAct.
///
/// Equivalente às funções Python do notebook:
///   - consultar_estoque(item)
///   - consultar_preco_produto(produto)
///   - ferramenta_encontrar_produto_mais_caro()
///   - ferramenta_calcular_valor_total_lista(lista_itens)
/// </summary>
public static class Ferramentas
{
    // ─── Dados simulados (equivalente aos dicts Python) ───────────────────────

    private static readonly Dictionary<string, int> Estoque = new()
    {
        ["monitor"] = 75,
        ["teclado"] = 120,
        ["mouse gamer"] = 80,
        ["webcam"] = 40,
        ["headset"] = 60,
        ["impressora"] = 15,
    };

    private static readonly Dictionary<string, double> Precos = new()
    {
        ["monitor"] = 999.90,
        ["teclado"] = 150.00,
        ["mouse gamer"] = 99.50,
        ["webcam"] = 120.00,
        ["headset"] = 180.00,
        ["impressora"] = 750.00,
    };

    // ─── Ferramentas ──────────────────────────────────────────────────────────

    /// <summary>
    /// Retorna a quantidade disponível de um item.
    /// Equivalente a: consultar_estoque(item)
    /// </summary>
    public static string ConsultarEstoque(string item)
    {
        item = item.Trim().ToLower();

        return Estoque.TryGetValue(item, out var qtd)
            ? $"Temos {qtd} {item}s em estoque."
            : $"Item '{item}' não encontrado no inventário.";
    }

    /// <summary>
    /// Retorna o preço unitário de um produto.
    /// Equivalente a: consultar_preco_produto(produto)
    /// </summary>
    public static string ConsultarPrecoProduto(string produto)
    {
        produto = produto.Trim().ToLower();

        return Precos.TryGetValue(produto, out var preco)
            ? $"O preço de um(a) {produto} é R$ {preco:F2}."
            : $"Produto '{produto}' não encontrado na lista de preços.";
    }

    /// <summary>
    /// Retorna o produto mais caro do inventário (sem argumentos).
    /// Equivalente a: ferramenta_encontrar_produto_mais_caro()
    /// </summary>
    public static string EncontrarProdutoMaisCaro()
    {
        if (Precos.Count == 0)
            return "Nenhum produto encontrado na lista de preços para comparação.";

        var maisCaro = Precos.MaxBy(kv => kv.Value);
        return $"O produto mais caro é o(a) {maisCaro.Key} com preço de R$ {maisCaro.Value:F2}.";
    }

    /// <summary>
    /// Calcula o valor total de uma lista de itens separados por vírgula.
    /// Equivalente a: ferramenta_calcular_valor_total_lista(lista_itens)
    ///
    /// Exemplo de entrada: "teclado, mouse gamer, monitor"
    /// </summary>
    public static string CalcularValorTotalLista(string listaItens)
    {
        var itens = listaItens
            .Split(',', StringSplitOptions.RemoveEmptyEntries)
            .Select(i => i.Trim().ToLower())
            .ToList();

        double total = 0;
        var naoEncontrados = new List<string>();

        foreach (var item in itens)
        {
            if (Precos.TryGetValue(item, out var preco))
                total += preco;
            else
                naoEncontrados.Add(item);
        }

        var resposta = $"O valor total dos itens encontrados é R$ {total:F2}.";

        if (naoEncontrados.Count > 0)
            resposta += $" Os seguintes itens não foram encontrados e não foram incluídos no cálculo: {string.Join(", ", naoEncontrados)}.";

        return resposta;
    }

    /// <summary>
    /// Despacha a execução para a ferramenta correta com base no nome e argumento.
    /// Equivalente ao bloco if/elif do Python dentro de run_react_agent().
    /// </summary>
    public static string Executar(string nomeFerramenta, string argumento)
    {
        return nomeFerramenta switch
        {
            "consultar_estoque" => ConsultarEstoque(argumento),
            "consultar_preco_produto" => ConsultarPrecoProduto(argumento),
            "encontrar_produto_mais_caro" => EncontrarProdutoMaisCaro(),
            "calcular_valor_total_lista" => CalcularValorTotalLista(argumento),
            _ => $"Erro: Ação '{nomeFerramenta}' desconhecida. Verifique o prompt ou a implementação da ferramenta."
        };
    }
}