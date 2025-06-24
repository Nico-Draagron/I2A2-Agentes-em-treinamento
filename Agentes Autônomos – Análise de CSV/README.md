
# 🌦️ Projeto de Tratamento de Dados Climáticos do INMET

Este projeto tem como objetivo ler, padronizar, limpar e consolidar múltiplos arquivos CSV fornecidos pelo INMET (Instituto Nacional de Meteorologia), referentes a estações meteorológicas. O pipeline transforma os dados brutos em um único arquivo unificado, pronto para análise e modelagem.

---

## 📁 Estrutura do Projeto

```
projeto_clima/
├── dados/                         # Contém os arquivos CSV originais do INMET
├── scripts/
│   └── tratar_dados_inmet.py     # Script principal de tratamento de dados
├── resultados/
│   └── dados_climaticos_tratados_todas_estacoes.csv  # Saída final
└── README.md                     # Este arquivo
```

---

## 🔧 O que o script faz (`tratar_dados_inmet.py`)

1. **Importa bibliotecas fundamentais**: `pandas`, `numpy`, `glob`, `os`.
2. **Busca todos os arquivos .csv** da pasta de dados usando `glob`.
3. **Lê os dados ignorando o cabeçalho extra** (linhas descritivas iniciais).
4. **Renomeia colunas** para nomes curtos e padronizados.
5. **Converte dados numéricos**: 
   - Troca vírgulas por pontos
   - Converte strings em números
   - Substitui `-999` por `NaN` (exceto para radiação, que vira 0).
6. **Converte a coluna de data** e cria a coluna `dia_semana`.
7. **Adiciona uma coluna com o nome da estação**, extraída do nome do arquivo.
8. **Concatena todos os arquivos** em um único `DataFrame`.
9. **Salva a saída final** no arquivo `dados_climaticos_tratados_todas_estacoes.csv`.

---

## 📊 Variáveis tratadas

As variáveis finais no conjunto de dados são:

- `data`: data da observação
- `hora`: hora UTC da medição
- `dia_semana`: dia da semana (Monday, Tuesday, etc.)
- `precipitacao`: chuva horária (mm)
- `temp_max`, `temp_min`, `temp_inst`: temperaturas máximas, mínimas e instantâneas (°C)
- `pressao`: pressão atmosférica horária (mB)
- `radiacao`: radiação global (Kj/m²)
- `umidade`: umidade relativa (%)
- `vento_vel`: velocidade do vento (m/s)
- `vento_raj`: rajada máxima de vento (m/s)
- `vento_dir`: direção do vento (graus)
- `estacao`: nome da estação meteorológica

---

## ▶️ Como executar

1. Instale os pacotes necessários:
   ```bash
   pip install pandas numpy
   ```

2. Edite o caminho da pasta no script:
   ```python
   pasta_dados = r'C:\seu\caminho\para\Dados_estacoes'
   ```

3. Execute o script:
   ```bash
   python scripts/tratar_dados_inmet.py
   ```

4. O resultado será salvo na pasta `resultados/`.

---

## 📌 Observações

- O script foi testado com arquivos do INMET no formato padrão (com 8 linhas de cabeçalho).
- Para novas estações ou anos, basta adicionar os arquivos na pasta `dados/`.
- Dados duplicados (com mesma data, hora e estação) são automaticamente evitados na consolidação final.

---

## 👨‍🔬 Autor

**Marcelo Bortoluzzi Diaz**  
Faculdade Antônio Meneghetti - Curso de Sistemas de Informação  
Projeto: *Clima e Negócios - Otimização Empresarial com Ciência de Dados e IA*
