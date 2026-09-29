Interpretação da Matriz de Confusão Normalizada
A matriz de confusão normalizada apresenta a proporção de acertos e erros do modelo em relação à classe real (True) e à classe predita (Predicted).
Taxa de acertos para as classes principais: O modelo apresenta uma taxa de acerto (Recall) de 92% para a classe fumaça (True: smoke) e de 85% para a classe fogo (True: fire).
Confusão entre classes: Quando há erro na predição da classe fogo real, o modelo a classifica como fumaça em 13% dos casos.
Falsos Positivos (Background): A última coluna (True: background) indica as predições do modelo em imagens contendo apenas o fundo. Nessas instâncias, o modelo prediz fogo em 64% dos casos e fumaça em 36% dos casos, o que indica a geração de falsos positivos em regiões sem a presença dos objetos de interesse.

Interpretação da Curva F1 (F1-Confidence Curve)
A curva F1 ilustra a relação entre a métrica F1 (equilíbrio ou média harmônica entre Precisão e Recall) e o limiar de confiança (confidence threshold) adotado pelo modelo.
Métricas de limiar: A curva correspondente a todas as classes ("all classes") atinge o valor máximo de F1 de 0.72 em um limiar de confiança de 0.287.
Fumaça vs. Fogo: A classe "smoke" atinge um pico de F1 próximo a 0.78, enquanto a classe "fire" apresenta o valor máximo em torno de 0.67. Esses valores indicam que a classe fumaça é detectada com maiores valores de precisão e recall do que a classe fogo neste conjunto de dados.
Resumo
O modelo identifica as classes fumaça e fogo com recall de 92% e 85%, respectivamente. No entanto, observa-se a ocorrência de falsos positivos em áreas de fundo (background). A inclusão de imagens negativas (sem fogo ou fumaça) no conjunto de dados de treinamento pode ser uma alternativa para auxiliar na redução da detecção incorreta do background.

Dataset 
Para mitigar a escassez de datasets para detecção de fogo e fumaça em cenários reais e complexos, pesquisadores da Universidade de Wuhan na China Wang et al. (2022) desenvolveram o FASDD (Flame and Smoke Detection Dataset), composto por mais de 100 mil imagens heterogêneas de fogo e fumaça. O dataset foi distribuído entre mais de 70 pessoas para anotações de fogo e fumaça. Em experimentos com redes neurais, arquiteturas como a YOLO alcançaram cerca de 80% de map-50, demonstrando viabilidade técnica para monitoramento cooperativo entre estações terrestres (torres), drones e plataformas orbitais. 



O Flame and Smoke Detection Dataset (FASDD) é uma coleção pioneira composta por mais de 120.000 imagens heterogêneas que englobam diversos cenários de incêndio, visando impulsionar a evolução de modelos de detecção de fogo. O FASDD atua como um benchmark desafiador para tarefas de object detection e introduz três subconjuntos de dados: FASDD_CV, FASDD_UAV e FASDD_RS. Esses subconjuntos são formados por imagens capturadas por sensores terrestres, aéreos e orbitais (spaceborne).
Experimentos utilizando modelos baseados em Swin Transformer demonstram um desempenho na detecção de fogo alcançando pontuações de mAP de 84,9%, 89,7% e 74,0% nos respectivos conjuntos. Quando treinados sobre o FASDD_RS, FASDD_UAV e FASDD_CV, modelos de deep learning podem ser implantados individualmente em edge devices, incluindo satélites ópticos, unmanned aerial vehicles (UAVs / Drones) e sensores terrestres.

Referências

[Dataset] https://www.scidb.cn/en/detail?dataSetId=ce9c9400b44148e1b0a749f5c3eb0bda
[Dataset - Artigo] https://www.tandfonline.com/doi/full/10.1080/10095020.2024.2347922#abstract
[Matriz de Confusão] https://www.ultralytics.com/glossary/confusion-matrix
[Curva F1] https://www.ultralytics.com/glossary/f1-score
