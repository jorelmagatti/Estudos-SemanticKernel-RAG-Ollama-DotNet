using Microsoft.AspNetCore.Mvc;
using ModelContextProtocol.Server;
using System.ComponentModel;       

[McpServerToolType]
public class ExemploTools
{
    [McpServerTool]
    [ActionName("helloworld")]
    [Description("Retorna uma saudação simples. Útil para testar se o MCP está funcionando.")]
    public string HelloWorld(string nome = "mundo")
    {
        return $"Olá, {nome}! ✅ Servidor MCP rodando com .NET 10.";
    }

    [McpServerTool]
    [ActionName("somar")]
    [Description("Soma dois números inteiros.")]
    public int Somar(
        [Description("Primeiro número")] int a,
        [Description("Segundo número")] int b)
    {
        return a + b;
    }
}