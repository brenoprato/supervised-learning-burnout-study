# SPEC.md — Especificação do Sistema de Predição de Burnout

---

## 1. Visão Geral do Sistema

O sistema tem como propósito **predizer o risco de burnout (esgotamento profissional) de funcionários** com base em atributos individuais coletados em ambiente corporativo. O problema central é a identificação precoce de colaboradores vulneráveis ao esgotamento, permitindo que equipes de Recursos Humanos realizem intervenções preventivas antes do agravamento do quadro.

**Principais usuários:** equipes de RH e People Analytics de organizações corporativas.

**Problema resolvido:** as abordagens tradicionais de avaliação de saúde ocupacional são manuais, subjetivas e lentas. O sistema automatiza essa identificação por meio de modelos de aprendizado de máquina, tornando o processo mais rápido, consistente e baseado em dados.

O sistema recebe como entrada os dados de um funcionário e retorna um **score contínuo de risco** entre 0 e 1, acompanhado de uma classificação de risco em três faixas: baixo, moderado e alto.

---

## 2. Escopo Funcional

### 2.1 Análise Exploratória dos Dados

- **Atores:** Cientista de dados / desenvolvedor do projeto
- **Descrição:** O sistema deve permitir a análise descritiva completa do conjunto de dados de entrada, incluindo a distribuição da variável alvo, as correlações entre os atributos, a identificação de valores ausentes e a análise segmentada por subgrupos de gênero, tipo de empresa e disponibilidade de trabalho remoto.
- **Pré-condições:** O dataset original deve estar disponível e acessível no repositório do projeto.
- **Pós-condições:** O analista possui compreensão clara da estrutura, qualidade e distribuição dos dados; visualizações são geradas e persistidas.
- **Critérios de aceitação:**
  - A distribuição da variável alvo `Burn Rate` é apresentada graficamente.
  - É gerado um mapa de calor de correlação entre todas as variáveis numéricas.
  - Todos os valores ausentes no dataset são identificados e documentados.
  - Análises por subgrupo (gênero, tipo de empresa, disponibilidade de home office) são produzidas e exibidas.

---

### 2.2 Pré-processamento dos Dados

- **Atores:** Cientista de dados / desenvolvedor do projeto
- **Descrição:** O sistema deve transformar o dataset bruto em um formato adequado para treinamento dos modelos. Isso inclui o tratamento de valores ausentes, a conversão de variáveis categóricas para representações numéricas, a criação de novas variáveis derivadas e a normalização dos atributos numéricos.
- **Pré-condições:** A análise exploratória foi concluída e os problemas de qualidade de dados foram identificados.
- **Pós-condições:** Um dataset limpo e processado é gerado e persistido para uso nas etapas seguintes.
- **Critérios de aceitação:**
  - Todos os valores ausentes no dataset são tratados por imputação estatística (média ou mediana).
  - Todas as variáveis categóricas são convertidas para representação numérica.
  - Uma nova variável representando o tempo de empresa do funcionário é derivada a partir da data de admissão.
  - Os atributos numéricos são normalizados por escalonamento.
  - O dataset resultante não contém valores ausentes e está pronto para treinamento.

---

### 2.3 Treinamento dos Modelos de Predição

- **Atores:** Cientista de dados / desenvolvedor do projeto
- **Descrição:** O sistema deve treinar três modelos de regressão supervisionada com complexidades distintas — um modelo linear simples (baseline), um modelo de conjunto baseado em árvores de decisão e um modelo de boosting sequencial — utilizando os dados pré-processados.
- **Pré-condições:** O dataset pré-processado está disponível; a divisão entre conjuntos de treino e teste foi realizada com semente aleatória fixa.
- **Pós-condições:** Os três modelos treinados são persistidos em disco para uso posterior.
- **Critérios de aceitação:**
  - O modelo de Regressão Linear é treinado e serve como linha de base comparativa.
  - O modelo Random Forest Regressor é treinado e gera a importância relativa de cada atributo.
  - O modelo XGBoost Regressor é treinado com otimização de hiperparâmetros.
  - Os três modelos treinados são salvos no repositório do projeto.
  - A divisão dos dados utiliza proporção de 80% para treino e 20% para teste, com semente fixa para garantir reprodutibilidade.

---

### 2.4 Validação e Avaliação dos Modelos

- **Atores:** Cientista de dados / desenvolvedor do projeto
- **Descrição:** O sistema deve avaliar o desempenho de cada modelo utilizando métricas quantitativas padronizadas e validação cruzada, além de gerar visualizações comparativas entre os valores reais e os valores preditos.
- **Pré-condições:** Os três modelos foram treinados e estão disponíveis; o conjunto de teste está separado.
- **Pós-condições:** Uma tabela comparativa com as métricas dos três modelos é gerada; gráficos de avaliação são produzidos e persistidos.
- **Critérios de aceitação:**
  - As métricas MAE, RMSE e R² são calculadas e exibidas para cada modelo.
  - Validação cruzada com K-Fold é aplicada para reduzir o viés da divisão aleatória.
  - O modelo XGBoost passa por busca de melhores hiperparâmetros antes da avaliação final.
  - Um gráfico comparando valores reais e valores preditos é gerado para cada modelo.
  - Gráficos de importância de atributos são gerados para o Random Forest e o XGBoost.
  - Uma tabela comparativa consolidada com os resultados dos três modelos é produzida.

---

### 2.5 Função de Scoring Individual

- **Atores:** Sistema de RH integrado / usuário final
- **Descrição:** O sistema deve disponibilizar uma função de avaliação que recebe os dados de um único funcionário como entrada e retorna o score de risco de burnout entre 0 e 1, acompanhado da classificação de risco correspondente (baixo, moderado ou alto).
- **Pré-condições:** Ao menos um modelo treinado está disponível; os dados do funcionário estão fornecidos em formato compatível com as variáveis de entrada do modelo.
- **Pós-condições:** Um score numérico e uma classificação de risco são retornados ao solicitante.
- **Critérios de aceitação:**
  - A função aceita os atributos de um funcionário como entrada.
  - A função retorna um valor numérico contínuo entre 0 e 1.
  - A função retorna a classificação de risco correspondente ao score calculado (baixo, moderado ou alto), conforme as faixas definidas nas regras de negócio.
  - A função é documentada no notebook de avaliação e pode ser reutilizada de forma independente.

---

## 3. Requisitos Não Funcionais

### 3.1 Reprodutibilidade
Todos os resultados produzidos pelo sistema devem ser reprodutíveis. Qualquer execução do pipeline com os mesmos dados de entrada deve produzir os mesmos resultados. Isso implica o uso de semente aleatória fixa em todas as etapas que envolvem aleatoriedade.

### 3.2 Rastreabilidade e Documentação
Cada etapa do pipeline deve estar implementada em um script Python dedicado, com responsabilidade única e claramente descrita. Cada script deve conter um bloco de cabeçalho com objetivo, entradas esperadas e saídas produzidas, favorecendo legibilidade e manutenção.

### 3.3 Organização e Persistência
Os dados brutos, os dados processados, os modelos treinados, os gráficos gerados e os relatórios de resultado devem ser armazenados em diretórios distintos e bem definidos, conforme a estrutura de diretórios estabelecida para o projeto.

### 3.4 Transparência dos Resultados
O sistema deve produzir relatórios que permitam a comparação objetiva entre os três modelos. Os resultados devem ser apresentados de forma clara, com métricas numéricas e visualizações gráficas.

### 3.5 Limitações Declaradas
O sistema deve declarar explicitamente suas limitações, incluindo: a possível natureza sintética do dataset, o caráter preditivo (não diagnóstico) do modelo, e o risco de viés associado a variáveis sensíveis como gênero e tipo de empresa.

### 3.6 Isolamento de Dependências
O ambiente de execução do sistema deve ser isolado por meio de ambiente virtual, com todas as dependências de bibliotecas registradas em arquivo de requisitos versionado.

---

## 4. Regras de Negócio

### RN-01 — Escala do Score de Risco
O score de risco de burnout é um valor contínuo no intervalo fechado de 0 a 1, onde valores próximos de 0 indicam menor risco e valores próximos de 1 indicam maior risco.

### RN-02 — Classificação por Faixa de Risco
O score deve ser mapeado para uma das três categorias de risco:
- Faixa de 0,0 a 0,3 (inclusive): risco **baixo** — nenhuma ação imediata necessária.
- Faixa de 0,3 a 0,6 (exclusive nos extremos): risco **moderado** — atenção recomendada.
- Faixa de 0,6 a 1,0 (inclusive): risco **alto** — intervenção necessária.

### RN-03 — Variável Alvo
A variável a ser predita pelo modelo é denominada `Burn Rate`, de natureza contínua, presente no dataset de origem.

### RN-04 — Atributos de Entrada
Os atributos utilizados como entrada para o modelo incluem: identificador do funcionário, data de admissão, gênero, tipo de empresa, disponibilidade de trabalho remoto (home office), cargo (designação), alocação de recursos e pontuação de fadiga mental.

### RN-05 — Derivação de Atributo Temporal
O tempo de permanência do funcionário na empresa deve ser calculado a partir da data de admissão, sendo tratado como um atributo derivado distinto da data original.

### RN-06 — Proporção de Divisão dos Dados
O conjunto de dados disponível para modelagem deve ser dividido na proporção de 80% para treinamento e 20% para teste, com semente aleatória fixa e conhecida para fins de reprodutibilidade.

### RN-07 — Comparação entre Modelos
Os três modelos devem ser avaliados com as mesmas métricas (MAE, RMSE e R²) e sobre o mesmo conjunto de teste, garantindo comparação justa e consistente.

### RN-08 — Caráter Preditivo, não Diagnóstico
O score gerado pelo sistema é um indicador de risco preditivo baseado em dados históricos e características observáveis. Ele não substitui avaliação clínica ou diagnóstico de saúde mental realizado por profissional habilitado.

### RN-09 — Análise de Equidade
O modelo deve permitir a análise de seus resultados segmentados por variáveis sensíveis (como gênero e tipo de empresa) para identificar possíveis disparidades ou vieses nas predições.

---

## 5. Restrições Técnicas e de Projeto

### 5.1 Ambiente de Execução
O sistema deve ser executado em ambiente Python nas versões 3.10.x ou 3.11.x, dentro de ambiente virtual isolado.

### 5.2 Plataforma de Desenvolvimento
Todo o pipeline do projeto deve ser desenvolvido em scripts Python modulares, organizados por etapa funcional dentro do diretório de código-fonte. Cada script corresponde a um componente funcional distinto da arquitetura.

### 5.3 Dataset de Origem
O dataset utilizado é o "Are Your Employees Burning Out?", disponível publicamente no Kaggle, composto por aproximadamente 22.750 registros com a variável alvo `Burn Rate` e os atributos listados na regra RN-04.

### 5.4 Bibliotecas Obrigatórias
O sistema deve utilizar as seguintes bibliotecas externas para suas funcionalidades principais:

- Manipulação e análise de dados tabulares
- Operações numéricas de alta performance
- Modelos de aprendizado de máquina, métricas de avaliação, pré-processamento e validação cruzada
- Modelo de gradient boosting (XGBoost)
- Geração de gráficos e visualizações estatísticas
- Serialização e carregamento de modelos treinados

### 5.5 Persistência dos Modelos
Os modelos treinados devem ser salvos em disco em formato binário reutilizável, permitindo carregamento posterior sem necessidade de re-treinamento.

### 5.6 Versionamento do Projeto
O projeto deve ser versionado com Git. Os dados brutos originais e o ambiente virtual não devem ser incluídos no repositório versionado.

### 5.7 Alinhamento com ODS da ONU
O projeto deve declarar explicitamente sua relação com os Objetivos de Desenvolvimento Sustentável ODS 3 (Saúde e Bem-Estar), ODS 8 (Trabalho Decente e Crescimento Econômico) e, de forma secundária, ODS 10 (Redução das Desigualdades).

---

## 6. Glossário

| Termo | Definição |
|---|---|
| **Burnout** | Síndrome de esgotamento profissional reconhecida pela OMS, caracterizada por exaustão emocional, distanciamento mental do trabalho e redução da eficácia profissional. |
| **Score de risco** | Valor numérico contínuo entre 0 e 1 que representa a probabilidade estimada de um funcionário desenvolver burnout. |
| **Variável alvo** | Atributo que o modelo busca predizer; neste projeto, é o `Burn Rate`. |
| **Atributo (feature)** | Variável de entrada utilizada pelo modelo para realizar a predição. |
| **Feature engineering** | Processo de criação de novos atributos derivados de atributos existentes para melhorar a capacidade preditiva do modelo. |
| **EDA (Análise Exploratória de Dados)** | Etapa de investigação inicial dos dados para compreender distribuições, relações e qualidade antes da modelagem. |
| **Regressão supervisionada** | Técnica de aprendizado de máquina onde o modelo aprende a prever um valor numérico contínuo a partir de exemplos rotulados. |
| **Regressão Linear** | Modelo de predição que assume relação linear entre os atributos de entrada e a variável alvo. Utilizado como baseline. |
| **Random Forest** | Modelo de aprendizado de máquina baseado na combinação de múltiplas árvores de decisão independentes. |
| **XGBoost / Gradient Boosting** | Modelo de aprendizado de máquina que constrói árvores de decisão sequencialmente, cada uma corrigindo os erros da anterior. Representa o estado da arte para dados tabulares. |
| **Baseline** | Modelo de referência mínimo, utilizado para comparação com modelos mais complexos. |
| **MAE (Mean Absolute Error)** | Métrica de avaliação que representa o erro médio absoluto entre os valores reais e os preditos. |
| **RMSE (Root Mean Squared Error)** | Métrica de avaliação que penaliza erros maiores de forma mais intensa; é a raiz quadrada do erro quadrático médio. |
| **R² (Coeficiente de Determinação)** | Métrica que indica o percentual da variância da variável alvo que é explicado pelo modelo. Varia entre 0 e 1. |
| **K-Fold Cross Validation** | Técnica de validação que divide o conjunto de dados em K partes, treinando e avaliando o modelo K vezes para reduzir o viés da divisão aleatória. |
| **Imputação** | Processo de substituição de valores ausentes por estimativas estatísticas (média, mediana, etc.). |
| **Encoding** | Conversão de variáveis categóricas (texto) para representações numéricas compatíveis com os modelos de aprendizado de máquina. |
| **Normalização** | Processo de reescalonamento dos valores numéricos para uma faixa ou distribuição padrão, visando melhorar o desempenho dos modelos. |
| **People Analytics** | Campo de atuação que aplica análise de dados e inteligência artificial à gestão de pessoas em organizações. |
| **WFH (Work From Home)** | Modalidade de trabalho remoto, referenciada no dataset como disponibilidade de home office. |
| **ODS** | Objetivos de Desenvolvimento Sustentável, agenda global da ONU composta por 17 metas para desenvolvimento humano e ambiental até 2030. |
