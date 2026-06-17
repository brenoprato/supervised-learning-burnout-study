# Limitações Declaradas do Sistema

Este documento declara explicitamente as limitações conhecidas do sistema de predição de burnout, conforme exigido pelo RNF 3.5 da SPEC.

---

## 1. Dataset de Survey Auto-Reportado com Contexto de Trabalho Remoto

O dataset "Remote Work Health Impact Survey 2025" (Kaggle, coletado em junho de 2025)
representa uma amostra diversificada de trabalhadores globais em contexto pós-pandemia,
cobrindo regimes remoto, híbrido e presencial. Embora mais representativo do cenário
atual do que datasets sintéticos, trata-se de dados de survey auto-reportados, sujeitos
a viés de resposta e seleção amostral. Os resultados devem ser interpretados com essa
limitação em mente.

As correlações entre as features e o alvo são fracas (máximo Pearson ≈ 0,044), resultando
em R² ≈ 0,04 para todos os modelos — desempenho típico para dados reais de saúde
ocupacional auto-reportados. As métricas devem ser lidas como desempenho em um benchmark
de survey, não como expectativa de campo em dados de RH corporativos.

---

## 2. Caráter Preditivo, Não Diagnóstico (RN-08)

O score gerado pelo sistema é um **indicador de risco estatístico** baseado em padrões
históricos observados no dataset de treinamento. Ele **não constitui diagnóstico de
saúde mental** e **não substitui avaliação realizada por profissional habilitado**
(psicólogo, médico do trabalho ou similar).

O sistema deve ser utilizado como ferramenta de triagem e priorização de atenção por
equipes de RH — não como instrumento clínico ou base única para decisões sobre o
funcionário.

---

## 3. Risco de Viés por Variáveis Sensíveis

O modelo utiliza `Gender` e `Work_Arrangement` como atributos de entrada. A análise
de equidade realizada na Feature 4 não identificou diferenças grandes de MAE entre
subgrupos, porém:

- A análise foi superficial (apenas MAE por grupo); métricas de fairness mais rigorosas
  (equalized odds, demographic parity) não foram aplicadas.
- O dataset pode sub-representar determinados grupos, fazendo com que o modelo aprenda
  padrões enviesados que não são detectáveis apenas por comparação de erros médios.
- Em contextos reais, recomenda-se auditoria de equidade mais aprofundada antes de
  qualquer uso em decisões que afetem trabalhadores.

---

## 4. Ausência de Temporalidade

Todos os registros do dataset foram coletados em junho de 2025, sem variabilidade temporal.
O modelo não captura tendências sazonais, mudanças culturais pós-pandemia nem variações
no perfil de burnout ao longo do tempo. Em ambiente corporativo real, padrões de burnout
podem mudar ao longo do tempo (mudanças organizacionais, crises, etc.). O sistema não
possui mecanismo de re-treinamento incremental ou detecção de drift.

---

## 5. Ausência de Atualização do Modelo

O modelo foi treinado uma única vez sobre um snapshot estático do dataset. O sistema
não possui mecanismo de re-treinamento incremental ou detecção de drift. A adoção
em produção deve ser precedida de validação com dados da organização-alvo e de um
plano de re-treinamento periódico.

---

## Referências

- SPEC.md — Seção 3.5 (Limitações Declaradas)
- SPEC.md — RN-08 (Caráter Preditivo, não Diagnóstico)
- SPEC.md — RN-09 (Análise de Equidade)
- Dataset: https://www.kaggle.com/datasets/pratyushpuri/remote-work-health-impact-survey-2025
