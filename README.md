# Análise Temática de Trabalhos de Conclusão de Curso para o Curso Tecnologia em Sistemas para Internet no Instituto Federal de Brasília Campus Brasília.

1. **Introdução: Tema e Motivação**
    
    O objetivo geral deste projeto foi aplicar os conhecimentos adquiridos na disciplina para desenvolver um ciclo completo de análise de dados, desde a obtenção e tratamento de dados brutos até a aplicação de modelos de aprendizado de máquina e a comunicação eficaz dos resultados.

    A motivação para a escolha do tema surgiu da necessidade de compreender a paisagem acadêmica do curso Tecnologia em Sistemas para Internet do Instituto Federal de Brasília Campus Brasília. Analisar os Trabalhos de Conclusão de Curso (TCCs) nos permite identificar os principais focos temáticos abordados pelos alunos, as áreas de especialização dos orientadores e a evolução desses temas ao longo do tempo. Este projeto busca, portanto, transformar o volume de dados textuais não estruturados, disponíveis no repositório da instituição (https://bdtcbra.omeka.net/items/browse?collection=9), em insights visuais e interativos que possam ser úteis para alunos, professores e para a coordenação do curso.

2. **Estrutura do Projeto**

    Todo o código-fonte desenvolvido para este projeto, incluindo os scripts de scraping e análise, está disponível publicamente no seguinte repositório do GitHub. O repositório contém o histórico de desenvolvimento e instruções para a execução do pipeline.

    - Link para o Repositório: https://github.com/Ana-Caixeta/projeto_final_introducao_a_ciencia_de_dados

    O projeto está organizado na seguinte estrutura de diretórios para garantir a clareza e reprodutibilidade:

    ```
    projeto_final/
    │
    ├── data/
    │   ├── tccs_sistemas_internet_ifb.json  # Dados brutos coletados pelo scraper
    │   └── tccs_sistemas_internet_ifb.csv   # Dados brutos coletados pelo scraper
    │
    ├── analysis/
    │   ├── analise_orientadores_ranking.html # Visualizações interativas
    │   ├── analise_orientadores_temas.html   # Visualizações interativas
    │   ├── analise_orientadires_tempo.html   # Visualizações interativas
    │   ├── mapa_temas_interativos.html       # Visualizações interativas
    │   ├── nuvem_palavras_temas.png          # Imagem estática das nuvens de palavras
    │   ├── tabela_geral_tccs.html            # Tabela final com os resultados da análise
    │   └── tccs_clusterizados_resultado.csv  # Tabela final com os resultados da análise
    │
    ├── requirements.txt                     # Arquivo txt com as bibliotecas necessárias
    ├── scraper.py                           # Script para a obtenção dos dados (Web Scraping)
    └── analysis.py                          # Script principal para a análise e visualização
    ```

3. **Instalação e Configuração**

    Para executar este projeto localmente, siga os passos abaixo.

    **Pré-requisitos:** Python 3.9 ou superior

    **Git**:

    Clone o repositório:
    
    `git clone https://github.com/seu-usuario/nome-do-seu-repositorio.git`
    
    `cd nome-do-seu-repositorio`
    
    Instale as dependências:

    É recomendado criar um ambiente virtual. As bibliotecas necessárias estão listadas no arquivo requirements.txt.

    `pip install -r requirements.txt`

4. **Como Executar o Pipeline**

    O processo é dividido em duas etapas principais, executadas por scripts separados.

    **Etapa 1: Obtenção de Dados**
    
    Execute o script de web scraping para coletar os dados mais recentes do repositório de TCCs:
    
    `python scraper.py`
    
    O arquivo de saída será salvo em `projeto_final/data/`.

    **Etapa 2: Análise e Geração de Visualizações**
    
    Após a coleta, execute o script de análise Ele irá processar os dados, aplicar o modelo de machine learning e gerar todas as visualizações:
    
    `python analysis.py`

    Os resultados serão adicionados à pasta `projeto_final/analysis/`.

    Ao final da execução, os gráficos interativos serão abertos no seu navegador e todos os arquivos de resultado estarão salvos no diretório de análise.

5. **Resultados e Visualizações**

    A análise gerou um conjunto de visualizações interativas para explorar os resultados da clusterização e os metadados dos TCCs.

    **Visualização 1: Mapa Interativo de Temas**

    Este gráfico de dispersão 2D, onde cada ponto representa um TCC, e sua posição no mapa é determinada pelo algoritmo t-SNE, que posiciona trabalhos similares próximos uns dos outros. As cores representam os 6 cluster de temas encontrados. Ao passar o mouse sobre um ponto, o título do TCC é exibido.

    ![Dispersão](img\dispersao.png)

    ![Dispersão](img\dispersao_comparacao1.png)

    ![Dispersão](img\dispersao_comparacao2.png)

    **Visualização 2: Painel de Análise de Orientadores**

    Para entender a contribuição dos docentes, foi criado um painel consolidado que apresenta três análises sobre os orientadores com mais trabalhos: o ranking de orientações, a distribuição de cluster de temas orientados por cada um e a produção anual em que orientaram.

    ![Orientadores Ranking](img\orientadores_ranking.png)

    ![Orientadores Temas](img\orientadores_temas.png)

    ![Orientadores Tempo](img\orientadores_tempo.png)

    **Visualização 3: Tabela Interativa de TCCs**
    
    Uma tabela completa foi gerada para permitir a visualização da lista de Trabalhos de Conclusão de Curso coletados e o cluster associado à ele.

    ![Tabela TCCs](img\tabela_geral_tccs.png)

    **Visualização 5: Nuvem de Palavras por Tema**
    
    Para cada um dos 6 cluster de temas, foi gerada uma nuvem de palavras que destaca visualmente os termos mais importantes. Palavras maiores indicam maior relevância para aquele tema, oferecendo um resumo visual rápido de cada cluster.

    ![Nuvem de Temas](img\nuvem_palavras_temas.png)

6. **Conclusão**

    A análise dos TCCs do curso de Tecnologia em Sistemas para Internet revelou informações valiosas sobre o corpo docente. Foi possível mapear as especialidades de cada professor, identificar os docentes de referência em cada tema e observar o volume de orientações ao longo dos anos.

    O modelo de aprendizado de máquina (K-Means) provou ser uma ferramenta eficaz para agrupar os trabalhos de forma coesa e coerente. Adicionalmente, a nomeação automática dos clusters facilitou enormemente a interpretação dos resultados, pois traduz os agrupamentos matemáticos em temas compreensíveis.

    Como limitação, reconhece-se que a qualidade e a completude dos dados extraídos dependem da consistência do preenchimento no repositório original. Para trabalhos futuros, sugere-se a aplicação de modelos de tópicos mais avançados e a análise da evolução da complexidade técnica dos projetos ao longo dos anos.
