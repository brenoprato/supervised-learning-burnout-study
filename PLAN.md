# PLAN.md — Plano de Implementação: Sistema de Predição de Burnout

---

## 1. Visão Geral da Estratégia

O projeto é um **pipeline sequencial de ciência de dados**, organizado como um monolito analítico dividido em etapas discretas e ordenadas. Não há servidor, API nem interface web: o produto final é um conjunto de scripts Python modulares documentados, modelos persistidos, relatórios visuais e uma função de scoring reutilizável.

**Princípios que guiarão a implementação:**

- **Linearidade e dependência explícita entre etapas:** cada módulo consome a saída do anterior. Nenhuma etapa deve ser executada sem que a anterior esteja validada e seus artefatos persistidos.
- **Reprodutibilidade como invariante:** toda fonte de aleatoriedade deve ser controlada por semente fixa desde a primeira linha escrita. Isso não é uma preocupação de última hora — é uma restrição de entrada para cada módulo.
- **Complexidade crescente e comparável:** os três modelos são construídos em ordem de complexidade (baseline → ensemble → boosting), com critérios de avaliação idênticos para garantir comparação justa, conforme RN-07.
- **Documentação como entregável de primeira classe:** cada feature concluída gera seu próprio README. A documentação não é escrita no fim — é parte da definição de "pronto" de cada etapa.
- **Separação entre dados brutos e derivados:** o dataset original nunca é modificado. Toda transformação produz um novo artefato em diretório separado.

---

## 2. Estrutura do Projeto (Repositório)

A estrutura abaixo descreve os diretórios conceituais do repositório e seus propósitos. Os dados brutos e o ambiente virtual são excluídos do versionamento.

**Diretório raiz**
Contém o README principal do projeto, o arquivo de dependências versionado e o arquivo de exclusões do Git. É o ponto de entrada para qualquer colaborador.

**Dados — subdiretório de dados brutos**
Armazena o dataset original obtido do Kaggle, de forma imutável e não versionada. Nunca deve ser alterado após o download.

**Dados — subdiretório de dados processados**
Armazena o dataset gerado após o pré-processamento. Este artefato é versionável, pois é derivado de um processo determinístico documentado.

**Código-fonte — src/**
Contém os scripts Python do projeto, um por componente funcional do pipeline. Cada script possui responsabilidade única e é autossuficiente em termos de documentação interna (bloco de cabeçalho com objetivo, entradas e saídas esperadas). Um script de orquestração na raiz do diretório permite executar o pipeline completo em sequência ou cada etapa de forma isolada.

**Modelos**
Armazena os três modelos treinados em formato binário reutilizável. Os arquivos aqui depositados são os artefatos de saída da etapa de treinamento e os artefatos de entrada da etapa de avaliação e da função de scoring.

**Relatórios — subdiretório de figuras**
Armazena todos os gráficos gerados pelo pipeline: distribuições, mapas de calor, comparativos real vs. predito, importância de atributos. Cada figura deve ter nome descritivo e referência ao script que a gerou.

**Relatórios — subdiretório de resultados**
Armazena o documento de comparação final dos modelos com as métricas consolidadas (MAE, RMSE, R²).

**Documentação — subdiretório geral**
Contém a documentação de arquitetura e contexto do projeto, incluindo o alinhamento com os ODS declarados na SPEC (seção 5.7).

**Documentação — subdiretório por feature**
Cada funcionalidade entregue deposita aqui seu README específico. A estrutura interna é um subdiretório por feature, conforme detalhado na seção 5 deste plano.

**Infraestrutura**
Contém instruções e configurações relacionadas ao ambiente de execução: como criar o ambiente virtual, como instalar as dependências, e qualquer configuração de ferramenta necessária para execução local. Sem configurações técnicas codificadas — apenas prosa instrucional e o arquivo de dependências referenciado.

---

## 3. Decomposição em Componentes (Arquitetura Conceitual)

O pipeline é composto por cinco componentes funcionais e dois componentes de suporte. Cada componente funcional corresponde a uma funcionalidade da SPEC.

---

### Componente 1 — Explorador de Dados (EDA)

**Responsabilidade:** Investigar, descrever e visualizar o dataset bruto para revelar sua estrutura, qualidade e distribuição. Produz diagnóstico completo que orienta as decisões do componente seguinte.

**Consome:** Dataset bruto (artefato externo, obtido do Kaggle).

**Produz:** Conjunto de visualizações (gráficos de distribuição, mapa de calor de correlações, análises por subgrupo) e relatório textual de valores ausentes identificados. Esses artefatos são depositados no diretório de figuras.

**Dependências:** Nenhuma dependência em outro componente interno. É o ponto de entrada do pipeline.

**Contrato de saída:** O analista deve ser capaz de responder, após esta etapa, às seguintes perguntas: qual é a distribuição da variável alvo? Quais atributos apresentam maior correlação entre si e com o alvo? Quais colunas contêm valores ausentes e em que proporção? Há diferenças observáveis entre subgrupos de gênero, tipo de empresa e disponibilidade de home office?

---

### Componente 2 — Transformador de Dados (Pré-processamento)

**Responsabilidade:** Converter o dataset bruto em um dataset limpo, completo e padronizado, pronto para ser consumido pelos modelos de aprendizado de máquina.

**Consome:** Dataset bruto; diagnóstico produzido pelo Explorador de Dados.

**Produz:** Dataset processado, persistido no diretório de dados processados. Inclui: ausência de valores nulos, variáveis categóricas convertidas para numérico, atributo derivado de tempo de empresa, e todos os atributos numéricos normalizados por escalonamento.

**Dependências:** Explorador de Dados (deve estar concluído).

**Contrato de saída:** O dataset produzido deve conter exatamente as colunas esperadas pelos modelos, sem valores ausentes, sem variáveis textuais brutas, com o atributo temporal derivado presente e com escala padronizada. A divisão treino/teste (80/20) com semente fixa é realizada ao final desta etapa e os dois subconjuntos são disponibilizados para os componentes seguintes.

---

### Componente 3 — Treinador de Modelos

**Responsabilidade:** Treinar os três modelos de regressão supervisionada em ordem crescente de complexidade: Regressão Linear (baseline), Random Forest Regressor e XGBoost Regressor. O XGBoost passa por otimização de hiperparâmetros antes de ser finalizado.

**Consome:** Conjunto de treino produzido pelo Transformador de Dados.

**Produz:** Três modelos treinados, persistidos em disco no diretório de modelos. Cada modelo é um artefato binário reutilizável, nomeado de forma descritiva.

**Dependências:** Transformador de Dados (dataset processado e dividido deve estar disponível).

**Contrato de saída:** Três arquivos de modelo disponíveis no diretório de modelos, cada um capaz de receber o mesmo conjunto de atributos de entrada e produzir uma predição de score entre 0 e 1. O Random Forest deve expor adicionalmente a importância relativa de cada atributo.

---

### Componente 4 — Avaliador de Modelos

**Responsabilidade:** Medir e comparar o desempenho dos três modelos treinados sobre o conjunto de teste, utilizando as métricas MAE, RMSE e R². Aplica validação cruzada K-Fold para verificação adicional de robustez. Gera todas as visualizações comparativas exigidas na SPEC (seção 2.4).

**Consome:** Os três modelos persistidos; conjunto de teste produzido pelo Transformador de Dados.

**Produz:** Tabela comparativa de métricas para os três modelos; gráficos de valores reais vs. preditos para cada modelo; gráficos de importância de atributos para Random Forest e XGBoost. Todos os artefatos são depositados nos diretórios de relatórios.

**Dependências:** Treinador de Modelos (todos os três modelos devem estar disponíveis).

**Contrato de saída:** Um documento de resultados consolidado com a tabela de métricas e referências aos gráficos gerados. O analista deve ser capaz de determinar, a partir deste artefato, qual modelo teve o melhor desempenho e em quais métricas.

---

### Componente 5 — Função de Scoring Individual

**Responsabilidade:** Expor uma função autossuficiente e documentada que recebe os atributos de um único funcionário, aplica o pré-processamento necessário, carrega o modelo mais adequado e retorna o score numérico de burnout entre 0 e 1 acompanhado da classificação de risco (baixo, moderado ou alto), conforme RN-01 e RN-02.

**Consome:** Dados de um único funcionário (nos mesmos atributos definidos em RN-04); modelos persistidos; transformações aplicadas na etapa de pré-processamento.

**Produz:** Score numérico contínuo entre 0 e 1; classificação de risco textual correspondente.

**Dependências:** Transformador de Dados (para aplicar as mesmas transformações); Treinador de Modelos (para carregar o modelo serializado).

**Contrato de saída:** A função pode ser chamada de forma independente, sem necessidade de executar todo o pipeline. Ela deve estar documentada com exemplos de entrada e saída descritos em prosa no README da feature correspondente.

---

### Componente de Suporte A — Gestão de Ambiente

**Responsabilidade:** Garantir que qualquer colaborador consiga reproduzir o ambiente de execução de forma isolada, instalando exatamente as versões corretas das dependências. Registra todas as bibliotecas obrigatórias conforme seção 5.4 da SPEC.

**Não possui dependências funcionais.** É configurado antes do início do pipeline.

---

### Componente de Suporte B — Versionamento e Controle de Artefatos

**Responsabilidade:** Garantir que o código-fonte, scripts, modelos e relatórios sejam versionados corretamente. Define o que é incluído e o que é excluído do repositório, conforme restrição 5.6 da SPEC (dados brutos e ambiente virtual excluídos).

---

## 4. Roadmap de Implementação (Fases)

### Fase 0 — Fundação do Projeto

**Objetivo:** Estabelecer o ambiente de trabalho, a estrutura de diretórios e o versionamento antes de qualquer trabalho analítico.

**Atividades:**
- Criação da estrutura completa de diretórios conforme seção 2 deste plano.
- Configuração do ambiente virtual e registro das dependências.
- Inicialização do repositório Git com arquivo de exclusões adequado.
- Download manual do dataset do Kaggle e posicionamento no diretório de dados brutos.
- Criação do README raiz do projeto com descrição geral, instruções de ambiente e referências à SPEC e ao PLAN.

**Componentes envolvidos:** Gestão de Ambiente; Versionamento e Controle de Artefatos.

**Riscos:**
- Incompatibilidade de versões entre bibliotecas — mitigada pelo registro explícito de versões no arquivo de dependências.
- Dataset do Kaggle indisponível ou com estrutura diferente da esperada — verificar as colunas imediatamente após o download.

---

### Fase 1 — Análise Exploratória dos Dados

**Objetivo:** Compreender profundamente a estrutura, qualidade e distribuição do dataset antes de qualquer transformação.

**Funcionalidades incluídas:** SPEC 2.1 — Análise Exploratória dos Dados.

**Componentes envolvidos:** Explorador de Dados.

**Riscos:**
- Maior proporção de valores ausentes do que a esperada pode exigir revisão da estratégia de imputação definida para a Fase 2.
- Distribuição muito assimétrica da variável alvo pode impactar a escolha de métricas de avaliação (confirmada na Fase 3).
- Dataset sintético pode apresentar correlações artificialmente perfeitas — declarar na documentação como limitação (RNF 3.5).

---

### Fase 2 — Pré-processamento dos Dados

**Objetivo:** Transformar o dataset bruto em um artefato limpo, padronizado e pronto para modelagem, garantindo que todas as decisões de transformação sejam rastreáveis e reprodutíveis.

**Funcionalidades incluídas:** SPEC 2.2 — Pré-processamento dos Dados.

**Componentes envolvidos:** Transformador de Dados.

**Riscos:**
- A derivação do atributo de tempo de empresa depende do formato da data de admissão — verificar consistência antes de aplicar a transformação (RN-05).
- A escolha entre estratégias de encoding para variáveis categóricas (nominal vs. ordinal) pode afetar o desempenho dos modelos lineares — documentar a decisão no README da feature.
- Vazamento de dados (data leakage): as transformações de normalização devem ser ajustadas apenas no conjunto de treino e aplicadas ao de teste, nunca o contrário.

---

### Fase 3 — Treinamento dos Modelos

**Objetivo:** Construir, ajustar e persistir os três modelos de regressão em ordem crescente de complexidade, garantindo que todos sejam treinados sobre o mesmo conjunto de dados e com semente fixa.

**Funcionalidades incluídas:** SPEC 2.3 — Treinamento dos Modelos de Predição.

**Componentes envolvidos:** Treinador de Modelos.

**Riscos:**
- A otimização de hiperparâmetros do XGBoost por busca exaustiva pode ser computacionalmente custosa — definir um espaço de busca razoável e documentar o critério de escolha.
- Overfitting no Random Forest ou XGBoost — verificado na Fase 4 pela diferença entre desempenho em treino e teste.
- Modelos salvos em disco devem ser compatíveis com a versão da biblioteca instalada — registrar versão no arquivo de dependências antes de treinar.

---

### Fase 4 — Avaliação e Comparação dos Modelos

**Objetivo:** Medir e comparar objetivamente o desempenho dos três modelos, produzir todas as visualizações exigidas e consolidar os resultados em relatório final.

**Funcionalidades incluídas:** SPEC 2.4 — Validação e Avaliação dos Modelos.

**Componentes envolvidos:** Avaliador de Modelos.

**Riscos:**
- Resultados muito próximos entre modelos podem dificultar a recomendação de um único modelo — neste caso, o critério de desempate deve ser documentado explicitamente (sugestão: preferir o modelo mais simples com desempenho equivalente).
- A análise de equidade por subgrupo (RN-09) pode revelar vieses que precisam ser declarados nas limitações do sistema.

---

### Fase 5 — Função de Scoring e Documentação Final

**Objetivo:** Implementar e validar a função de scoring individual, consolidar toda a documentação do projeto e garantir que o repositório está em estado entregável.

**Funcionalidades incluídas:** SPEC 2.5 — Função de Scoring Individual.

**Componentes envolvidos:** Função de Scoring Individual; todos os READMEs por feature.

**Riscos:**
- A função de scoring precisa replicar exatamente as mesmas transformações de pré-processamento aplicadas durante o treinamento — qualquer divergência produz predições incorretas. Verificar com casos de teste descritos em prosa.
- Documentação incompleta pode comprometer a usabilidade do projeto como trabalho acadêmico — reservar tempo suficiente nesta fase.

---

## 5. Checkpoints e Documentação por Feature

---

### Feature 1 — Análise Exploratória dos Dados

**Checkpoints de implementação:**

1. Dataset bruto carregado com sucesso e dimensões confirmadas (aproximadamente 22.750 registros e 9 colunas).
2. Tipos de dados de cada coluna inspecionados e registrados; colunas numéricas e categóricas identificadas.
3. Contagem de valores ausentes por coluna apurada e documentada; proporção de nulos calculada.
4. Gráfico de distribuição da variável alvo `Burn Rate` gerado e salvo no diretório de figuras.
5. Mapa de calor de correlação entre variáveis numéricas gerado e salvo.
6. Análises segmentadas por gênero, tipo de empresa e disponibilidade de home office geradas e salvas.
7. Todas as visualizações revisadas e consideradas corretas pelo responsável; diagnóstico resumido no log de saída do script de EDA.

**README.md da feature:**
Ao concluir esta feature, será produzido o arquivo `docs/features/eda/README.md` contendo: descrição do objetivo da análise exploratória, lista das visualizações produzidas e o que cada uma revela, principais achados sobre qualidade dos dados (valores ausentes, outliers, distribuições), decisões tomadas a partir da EDA que impactam as etapas seguintes, e como re-executar o script para reproduzir os resultados.

---

### Feature 2 — Pré-processamento dos Dados

**Checkpoints de implementação:**

1. Estratégia de imputação definida para cada coluna com valores ausentes (média ou mediana), com justificativa registrada no README da feature.
2. Imputação aplicada; dataset resultante verificado como livre de valores nulos.
3. Estratégia de encoding definida para cada variável categórica e registrada no README da feature; transformação aplicada e colunas resultantes verificadas.
4. Atributo derivado de tempo de empresa calculado corretamente a partir da data de admissão; coluna original removida ou mantida com justificativa.
5. Normalização aplicada; verificado que o ajuste foi feito exclusivamente sobre o conjunto de treino.
6. Divisão treino/teste (80/20) realizada com semente fixa; tamanhos dos subconjuntos conferidos.
7. Dataset processado persistido no diretório correspondente; integridade do arquivo verificada.

**README.md da feature:**
Ao concluir esta feature, será produzido o arquivo `docs/features/preprocessing/README.md` contendo: descrição de cada transformação aplicada e a justificativa da escolha, detalhamento do atributo derivado de tempo de empresa e como foi calculado, critérios usados para imputação por coluna, estratégias de encoding adotadas, parâmetros de normalização, proporção e tamanho dos conjuntos de treino e teste, e instrução para re-executar o script.

---

### Feature 3 — Treinamento dos Modelos de Predição

**Checkpoints de implementação:**

1. Modelo de Regressão Linear treinado sobre o conjunto de treino; artefato serializado e salvo no diretório de modelos.
2. Modelo Random Forest Regressor treinado; importância de atributos extraída e registrada; artefato serializado e salvo.
3. Espaço de busca de hiperparâmetros do XGBoost definido e documentado no README da feature; busca executada; melhores hiperparâmetros registrados no log de saída do script.
4. Modelo XGBoost Regressor treinado com os melhores hiperparâmetros; artefato serializado e salvo.
5. Os três arquivos de modelo estão presentes no diretório de modelos com nomes descritivos.
6. Semente aleatória fixa confirmada em todas as chamadas de treinamento e divisão de dados.

**README.md da feature:**
Ao concluir esta feature, será produzido o arquivo `docs/features/training/README.md` contendo: descrição dos três modelos treinados e suas diferenças conceituais, justificativa da ordem de complexidade crescente, parâmetros padrão utilizados para Regressão Linear e Random Forest, melhores hiperparâmetros encontrados para o XGBoost e o critério de busca adotado, localização dos artefatos de modelo no repositório, e como recarregar cada modelo sem re-treinar.

---

### Feature 4 — Validação e Avaliação dos Modelos

**Checkpoints de implementação:**

1. Predições dos três modelos geradas sobre o conjunto de teste; resultados armazenados para comparação.
2. MAE, RMSE e R² calculados para cada modelo; valores impressos no log de saída do script e salvos no relatório de resultados.
3. Validação cruzada K-Fold aplicada a cada modelo; resultados médios e desvio padrão registrados.
4. Gráfico de valores reais vs. preditos gerado para cada modelo e salvo no diretório de figuras.
5. Gráficos de importância de atributos gerados para Random Forest e XGBoost e salvos.
6. Análise de equidade por subgrupo (gênero, tipo de empresa) realizada e documentada; vieses identificados declarados.
7. Tabela comparativa final consolidada e salva no diretório de resultados.
8. Conclusão textual sobre qual modelo apresentou melhor desempenho e em quais critérios registrada no relatório de resultados e no README da feature.

**README.md da feature:**
Ao concluir esta feature, será produzido o arquivo `docs/features/evaluation/README.md` contendo: tabela de métricas dos três modelos, interpretação dos resultados por métrica, análise das visualizações produzidas, resultados da validação cruzada, conclusão sobre o modelo recomendado com justificativa, e declaração dos vieses identificados na análise de equidade.

---

### Feature 5 — Função de Scoring Individual

**Checkpoints de implementação:**

1. Função definida com assinatura clara: recebe um conjunto de atributos de um funcionário, retorna score e classificação de risco.
2. Fluxo interno da função documentado em prosa (sem código) no README da feature: quais transformações são aplicadas, qual modelo é carregado, como o score é calculado e como a classificação é derivada das faixas RN-01 e RN-02.
3. Três casos de teste descritos em prosa: um por faixa de risco (baixo, moderado, alto). Para cada caso, os valores de entrada e os resultados esperados são descritos textualmente.
4. Função executada com os três casos de teste e resultados conferidos manualmente contra as faixas de RN-02.
5. Função declarada como reutilizável e independente; verificado que pode ser importada e executada sem re-executar todo o pipeline.

**README.md da feature:**
Ao concluir esta feature, será produzido o arquivo `docs/features/scoring/README.md` contendo: descrição da função de scoring e seu propósito, atributos esperados como entrada (com descrição de cada um), formato e significado da saída (score e classificação), as faixas de risco conforme RN-01 e RN-02, pré-requisitos para uso (modelos treinados disponíveis), exemplos de entrada e saída descritos em prosa para cada faixa de risco, e limitações declaradas conforme RNF 3.5 da SPEC.

---

## 6. Estratégia de Documentação Geral

Os documentos a seguir fazem parte do projeto além dos READMEs por feature.

---

**README raiz do projeto**
Criado na Fase 0. Contém: descrição do sistema e seu propósito, alinhamento com os ODS (conforme restrição 5.7 da SPEC), instruções de configuração do ambiente, estrutura de diretórios explicada em prosa, referência ao SPEC.md e ao PLAN.md, e referências aos READMEs das features. Atualizado ao final da Fase 5 com os resultados obtidos e a conclusão sobre o modelo recomendado.

**Documento de Arquitetura**
Criado na Fase 0 com base neste PLAN.md. Localizado no diretório de documentação geral. Descreve os componentes, suas responsabilidades, dependências e o fluxo do pipeline em linguagem natural. Revisado ao final da Fase 5 para refletir quaisquer decisões tomadas durante a implementação que divergiram do plano original.

**Documento de Limitações do Sistema**
Criado durante a Fase 4 (quando os resultados e vieses são conhecidos). Declara explicitamente: a possível natureza sintética do dataset, o caráter preditivo e não diagnóstico do sistema, os vieses identificados na análise de equidade, e quaisquer outras limitações encontradas. Referenciado no README raiz e no README da feature de avaliação.

**Arquivo de dependências versionado**
Criado na Fase 0 e congelado antes do início dos treinamentos (Fase 3). Garante que qualquer reprodução futura utilize as mesmas versões de todas as bibliotecas.

**Relatório de resultados consolidado**
Produzido na Fase 4 pelo Avaliador de Modelos. Localizado no diretório de resultados. Contém a tabela comparativa de MAE, RMSE e R² dos três modelos, a conclusão sobre o modelo recomendado e referências às figuras geradas.

---

## 7. Tratamento de Requisitos Não Funcionais

**RNF 3.1 — Reprodutibilidade**
Endereçado na Fase 0 pelo registro explícito de versões de dependências, e na Fase 2 pela divisão treino/teste com semente fixa. A semente global deve ser declarada em um módulo de configuração compartilhado por todos os scripts, garantindo consistência em todo o projeto sem repetição manual.

**RNF 3.2 — Rastreabilidade e Documentação**
Endereçado pela estrutura de scripts com responsabilidade única (um por componente funcional) e pela obrigatoriedade de README por feature ao final de cada entrega. Cada script deve ser executável de forma independente, com saída de log descritiva para rastrear o que foi processado.

**RNF 3.3 — Organização e Persistência**
Endereçado integralmente pela estrutura de diretórios definida na seção 2 deste plano. Cada tipo de artefato (dados brutos, dados processados, modelos, figuras, relatórios) possui um diretório dedicado. Nenhum artefato deve ser salvo fora do diretório correspondente à sua categoria.

**RNF 3.4 — Transparência dos Resultados**
Endereçado pela Fase 4, que exige explicitamente a produção de tabela comparativa, gráficos de avaliação e documento de resultados consolidado. O Avaliador de Modelos não está concluído enquanto todos esses artefatos não estiverem produzidos e revisados.

**RNF 3.5 — Limitações Declaradas**
Endereçado pelo Documento de Limitações do Sistema (seção 6 deste plano), pelo README da feature de avaliação e pelo README raiz. As três limitações definidas na SPEC (natureza sintética do dataset, caráter preditivo, risco de viés) devem aparecer em pelo menos dois documentos distintos.

**RNF 3.6 — Isolamento de Dependências**
Endereçado na Fase 0 pela configuração do ambiente virtual e pelo arquivo de dependências. O diretório do ambiente virtual deve estar listado explicitamente no arquivo de exclusões do Git antes do primeiro commit.

---

## 8. Observações e Premissas

**Premissas assumidas neste plano:**

- O dataset "Are Your Employees Burning Out?" está disponível para download no Kaggle e sua estrutura de colunas é a descrita na SPEC (seção 5.3 e RN-04). Qualquer divergência na estrutura real do arquivo exige revisão das Fases 2 e 3.
- O projeto é de natureza acadêmica e de execução local. Não há requisitos de escalabilidade horizontal, tempo de resposta em produção ou segurança de dados em trânsito.
- A função de scoring (Feature 5) utilizará o modelo de melhor desempenho identificado na Fase 4. Se dois modelos apresentarem desempenho equivalente, o mais simples será preferido — essa decisão deve ser registrada no README da feature.
- Os scripts Python são o produto final entregável; não há plano de empacotamento como aplicação web ou API.
- O volume de dados (~22.750 registros) é compatível com execução local em hardware comum. Não são necessárias estratégias de processamento distribuído ou otimização de memória além do que as bibliotecas padrão já oferecem.

**Limitações conhecidas do plano:**

- Este plano não detalha os critérios exatos de seleção de hiperparâmetros para o XGBoost, pois esses dependem de experimentos realizados durante a Fase 3. O espaço de busca deve ser documentado no README da feature de treinamento quando determinado.
- A análise de equidade por subgrupo (RN-09) pode revelar a necessidade de passos adicionais não previstos neste plano (como reamostragem ou análise de fairness mais aprofundada). Nesses casos, a Fase 4 deve ser estendida e o documento de limitações atualizado.
- A possível natureza sintética do dataset (mencionada na SPEC, seção 3.5) pode fazer com que os resultados de desempenho dos modelos sejam artificialmente elevados. Isso deve ser declarado como ressalva na interpretação dos resultados.
