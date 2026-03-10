namespace FarmaceuticAgentRagSemantickernel;

public static class MetadataEnricher
{
    // Mapeamento: categoria => palavras-chave que devem aparecer no texto (lowercase)
    private static readonly (string Categoria, string[] Keywords)[] Regras =
    [
        ("identificacao",         ["identificação do medicamento", "composição"]),
        ("indicacao",             ["indicação", "para que este medicamento é indicado", "indicações"]),
        ("como_funciona",         ["como este medicamento funciona", "mecanismo de ação"]),
        ("contraindicacao",       ["contraindicação", "quando não devo usar", "contra-indicações", "contra-indicação"]),
        ("advertencias_precaucoes", ["advertência", "precaução", "o que devo saber antes de usar"]),
        ("interacoes",            ["interação", "interações medicamentosas"]),
        ("posologia_modo_uso",    ["dose", "posologia", "como devo usar", "modo de usar"]),
        ("reacoes_adversas",      ["reações adversas", "quais os males", "reação adversa"]),
        ("armazenamento",         ["onde, como e por quanto tempo posso guardar", "armazenar", "conservar"]),
        ("superdosagem",          ["quantidade maior do que a indicada", "superdosagem", "overdose"]),
    ];

    /// <summary>
    /// Classifica uma lista de chunks e retorna nova lista com categoria preenchida.
    /// </summary>
    public static List<TextChunk> Enriquecer(IEnumerable<TextChunk> chunks)
    {
        return chunks.Select(Classificar).ToList();
    }

    private static TextChunk Classificar(TextChunk chunk)
    {
        var textoLower = chunk.Content.ToLower();

        foreach (var (categoria, keywords) in Regras)
        {
            if (keywords.Any(kw => textoLower.Contains(kw)))
                return chunk.WithCategoria(categoria);
        }

        return chunk.WithCategoria("geral");
    }

    /// <summary>
    /// Retorna estatísticas de distribuição por categoria (útil para debug).
    /// </summary>
    public static Dictionary<string, int> ContarPorCategoria(IEnumerable<TextChunk> chunks)
    {
        return chunks
            .GroupBy(c => c.Categoria)
            .ToDictionary(g => g.Key, g => g.Count());
    }
}