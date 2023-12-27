# Senator Investment Data Science Project

## Como Rodar?
Python 3.11 (ou mais recente)

1. Construa seus dados (opcional):
  `python3 -m src.normalize_data.index`
2. Rode o algoritmo de machine learning:
  `python3  -m src.services.machine_learning_service`

## Minerando dados de ações de senadores, pegando o preço
<img src="https://github.com/wh1t3h47/datascience-senate/blob/master/media/data_mining.png?raw=true" alt="" />

## Demonstra precisão do sklearn de 100% usando validação cruzada
<img src="https://github.com/wh1t3h47/datascience-senate/blob/master/media/precision.png?raw=true" alt="" />

Projeto de Data Science
1. Quem são os senadores que mais acertam investimentos: Tem dois tipos de acerto, verdadeiro positivo (comprou ação e valorizou) e verdadeiro negativo (vendeu a ação e baixou);
2. Normalizar os dados;
3. Separar um conjunto de retenção; 
4. Medir entropia e ganho de informação de cada atributo;
5. Definir o nó raiz da árvore que mais vai ter relevância;
6. Criar subgrupos (sim e não);
7. Testar contra conjunto de retenção;
8. Voltar ao passo 4 com subgrupo;
9. Quando bater a condição de parada, plotar o gráfico de sobreajuste;
10. Escolher o melhor número de nós para ávore de decisão;
- Objetivo: Decidir quais são os atributos mais relevantes para dizer quais investimentos mais prevem o mercado, ou seja, quais investimentos são os melhores que mais refletem na realidade?
Crie uma ferramenta que você insira um investimento (do futuro ou conjunto de retenção) e ele mostre qual a probabilidade dele ser um bom investimento baseado nos dados históricos dos atributos da transação
