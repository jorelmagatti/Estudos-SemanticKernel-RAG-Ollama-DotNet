namespace RhAgentRagSemanticKernel;

/// <summary>
/// Classifica chunks por categoria semântica de RH.
///
/// Equivalente ao Python:
///   if "férias" in texto:          chunk.metadata["categoria"] = "ferias"
///   elif "home office" in texto:   chunk.metadata["categoria"] = "home_office"
///   elif "conduta" in texto:       chunk.metadata["categoria"] = "conduta"
///   else:                          chunk.metadata["categoria"] = "geral"
/// </summary>
public static class MetadataEnricher
{
    private static readonly (string Categoria, string[] Keywords)[] Regras =
    [
        ("ferias",      ["férias", "ferias", "recesso", "folga", "descanso anual"]),
        ("home_office", ["home office", "remoto", "teletrabalho", "trabalho remoto", "híbrido"]),
        ("conduta",     ["conduta", "ética", "etica", "compliance", "assédio", "assedio",
                         "comportamento", "disciplinar", "infração", "infracao"]),
    ];

    public static List<TextChunk> Enriquecer(IEnumerable<TextChunk> chunks)
        => chunks.Select(Classificar).ToList();

    private static TextChunk Classificar(TextChunk chunk)
    {
        var texto = chunk.Content.ToLower();

        foreach (var (categoria, keywords) in Regras)
            if (keywords.Any(kw => texto.Contains(kw)))
                return chunk.WithCategoria(categoria);

        return chunk.WithCategoria("geral");
    }

    public static Dictionary<string, int> ContarPorCategoria(IEnumerable<TextChunk> chunks)
        => chunks.GroupBy(c => c.Categoria)
                 .ToDictionary(g => g.Key, g => g.Count());
}
