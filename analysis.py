"""
Este script implementa um pipeline completo para análise de dados de Trabalhos de
Conclusão de Curso (TCCs). O processo inclui:
1. Carregamento de dados de TCCs (CSV ou JSON).
2. Pré-processamento de texto, incluindo limpeza e lematização para análise semântica.
3. Vetorização dos textos usando a técnica TF-IDF.
4. Aplicação de algoritmos de clustering para agrupar TCCs por similaridade temática.
5. Geração de nomes descritivos e temáticos para cada tema (cluster).
6. Redução de dimensionalidade para visualização.
7. Geração e salvamento de saídas visuais:
    - Gráfico interativo de dispersão dos temas (HTML).
    - Painel consolidado com análises sobre os orientadores (HTML).
    - Gráfico hierárquico de títulos por orientador e ano (HTML).
    - Nuvens de palavras para os principais temas (PNG).
8. Salvamento dos resultados consolidados em CSV.
"""

# Importação de Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import hdbscan
import re
import warnings
from wordcloud import WordCloud
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from typing import List, Dict, Optional, Tuple, Any
import spacy
import os

# Ignora avisos comuns de bibliotecas para uma saída mais limpa
warnings.filterwarnings('ignore')

# Configurações Globais

# Tenta carregar o modelo de linguagem do spaCy para lematização em português.
# A lematização é crucial para agrupar variações de uma palavra (ex: 'programa', 'programar')
# em um único token, melhorando a qualidade do clustering.
try:
    NLP = spacy.load("pt_core_news_sm")
except OSError:
    print("AVISO: Modelo 'pt_core_news_sm' do spaCy não encontrado.")
    print("Para instalar, execute no seu terminal: python -m spacy download pt_core_news_sm")
    print("A análise continuará sem lematização, o que pode afetar a qualidade dos resultados.")
    NLP = None

# Stopwords genéricas da língua portuguesa (palavras comuns sem valor semântico).
GENERIC_STOP_WORDS = [
    'a', 'ao', 'aos', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aquilo', 'as', 'até', 'com', 'como',
    'da', 'das', 'de', 'dela', 'delas', 'dele', 'deles', 'depois', 'do', 'dos', 'e', 'ela', 'elas',
    'ele', 'eles', 'em', 'entre', 'era', 'eram', 'dessa', 'essa', 'essas', 'desse', 'esse', 'esses',
    'desta', 'esta', 'estas', 'deste', 'este', 'estes', 'eu', 'foi', 'for', 'foram', 'há', 'isso', 'isto',
    'já', 'mais', 'mas', 'me', 'mesmo', 'meu', 'meus', 'minha', 'minhas', 'muito', 'na', 'nas', 'não',
    'no', 'nos', 'nossa', 'nossas', 'nosso', 'nossos', 'num', 'numa', 'o', 'os', 'ou', 'para', 'pela',
    'pelas', 'pelo', 'pelos', 'por', 'qual', 'quando', 'que', 'quem', 'se', 'sem', 'ser', 'seu', 'seus',
    'só', 'sua', 'suas', 'também', 'tambem', 'te', 'tem', 'têm', 'tu', 'tua', 'tuas', 'um', 'uma', 'você', 'vocês',
    'vos', 'à', 'às', 'é', 'sido', 'tendo', 'ter', 'será', 'está', 'estão', 'estar', 'estou',
    'tinha', 'tinham', 'tive', 'teve', 'tivemos', 'tiveram', 'tenho', 'tens', 'temos', 'sim', 'cada',
    'mesma', 'mesmas', 'mesmos', 'mesmo', 'nela', 'nele', 'neles', 'nelas', 'nessa', 'nesse', 'nesta', 'neste',
    'nesses', 'nessas', 'deste', 'desta', 'desses', 'dessas', 'disso', 'nisso', 'disto', 'nisto', 'lá',
    'aí', 'ainda', 'assim', 'então', 'depois', 'antes', 'sempre', 'nunca', 'vez', 'vezes', 'tão',
    'bem', 'mal', 'já', 'logo', 'ali', 'cá', 'onde', 'sim', 'não', 'apesar', 'sao', 'nao'
]

# Stopwords específicas do domínio acadêmico/TCC (frequentes mas não discriminativas para os temas).
DOMAIN_STOP_WORDS = [
    'sistema', 'sistemas', 'desenvolvimento', 'projeto', 'trabalho', 'aplicação', 'uso', 'utilização',
    'utilizando', 'implementação', 'análise', 'estudo', 'pesquisa', 'apresenta', 'apresentar',
    'objetivo', 'objetivos', 'proposta', 'proposto', 'forma', 'podem', 'pode', 'sendo', 'partir',
    'tcc', 'conclusão', 'curso', 'bacharelado', 'artigo', 'cientifico', 'ifb', 'instituto', 'federal'
]

# Lista final de stopwords a serem removidas dos textos.
PORTUGUESE_STOP_WORDS = GENERIC_STOP_WORDS + DOMAIN_STOP_WORDS


class TCCClusteringAnalyzer:
    """
    Encapsula todo o pipeline de análise de clustering temático de TCCs.

    Esta classe gerencia o carregamento de dados, pré-processamento, clusterização,
    e a geração de todas as visualizações e resultados.
    """

    def __init__(self, data_path: Optional[str] = None, df: Optional[pd.DataFrame] = None):
        """
        Inicializa o analisador, carregando dados de um arquivo ou DataFrame.

        Args:
            data_path (Optional[str]): Caminho para o arquivo de dados (CSV ou JSON).
            df (Optional[pd.DataFrame]): DataFrame pandas já carregado com os dados dos TCCs.
        """
        self.df: Optional[pd.DataFrame] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.X_tfidf: Optional[np.ndarray] = None
        self.X_reduced: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        self.cluster_keywords: Dict[int, List[Tuple[str, float]]] = {}
        self.cluster_names: Dict[int, str] = {}
        self.reduction_method: Optional[str] = None
        self.nlp = NLP

        if df is not None:
            self.df = df.copy()
        elif data_path:
            if not os.path.exists(data_path):
                print(f"ERRO: O arquivo especificado não foi encontrado em '{data_path}'")
            else:
                self.load_data(data_path)

    def load_data(self, data_path: str) -> bool:
        """
        Carrega os dados dos TCCs a partir de um arquivo CSV ou JSON.

        Args:
            data_path (str): O caminho para o arquivo de dados.

        Returns:
            bool: True se os dados foram carregados com sucesso, False caso contrário.
        """
        try:
            if data_path.endswith('.csv'):
                self.df = pd.read_csv(data_path, encoding='utf-8')
            elif data_path.endswith('.json'):
                self.df = pd.read_json(data_path, encoding='utf-8')
            else:
                raise ValueError("Formato de arquivo não suportado. Use CSV ou JSON.")
            
            print(f"Dados carregados com sucesso de '{data_path}': {len(self.df)} TCCs")
            return True
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            return False

    def _lemmatize_text(self, text: str) -> str:
        """
        Função auxiliar interna para lematizar texto usando spaCy.

        Args:
            text (str): O texto a ser lematizado.

        Returns:
            str: O texto com as palavras em sua forma de lema.
        """
        if not self.nlp or not isinstance(text, str):
            return ""
        doc = self.nlp(text.lower())
        lemmas = [token.lemma_ for token in doc if token.pos_ not in ['PUNCT', 'SPACE', 'SYM', 'NUM']]
        return " ".join(lemmas)

    def preprocess_text(self, use_lemmatization: bool = True):
        """
        Executa o pipeline de pré-processamento de texto. Combina colunas de texto,
        limpa, lematiza e filtra documentos muito curtos.
        """
        if self.df is None: return

        print("\n--- Etapa 1: Pré-processamento de Textos")
        
        text_columns = [col for col in ['titulo', 'resumo', 'palavras_chave'] if col in self.df.columns]
        self.df['texto_combinado'] = self.df[text_columns].fillna('').agg('. '.join, axis=1)
        self.df['texto_limpo'] = self.df['texto_combinado'].apply(lambda x: re.sub(r'[^\w\s\.]', '', x).lower())
        self.df['texto_limpo'] = self.df['texto_limpo'].apply(lambda x: re.sub(r'\s+', ' ', x))
        
        if use_lemmatization and self.nlp:
            print("Aplicando lematização... (Isso pode levar alguns minutos)")
            self.df['texto_processado'] = self.df['texto_limpo'].apply(self._lemmatize_text)
        else:
            self.df['texto_processado'] = self.df['texto_limpo']

        # Remove TCCs com texto processado muito curto, pois podem adicionar ruído.
        initial_count = len(self.df)
        self.df.dropna(subset=['texto_processado'], inplace=True)
        self.df = self.df[self.df['texto_processado'].str.strip().str.len() > 50]
        final_count = len(self.df)
        
        if initial_count != final_count:
            print(f"Removidos {initial_count - final_count} TCCs com texto insuficiente.")
        
        print(f"Pré-processamento concluído. {len(self.df)} TCCs prontos para análise.")
    
    def vectorize_texts(self, max_features: int = 2000, ngram_range: Tuple[int, int] = (1, 2), min_df: int = 3, max_df: float = 0.85):
        """
        Converte os textos processados em uma matriz numérica usando TF-IDF.

        Args:
            max_features (int): Número máximo de termos a serem considerados.
            ngram_range (Tuple[int, int]): Faixa de n-gramas a serem extraídos.
            min_df (int): Frequência mínima de documento para um termo ser considerado.
            max_df (float): Frequência máxima de documento para um termo ser considerado.
        """
        if self.df is None or 'texto_processado' not in self.df.columns: return

        print("\n--- Etapa 2: Vetorização com TF-IDF")
        
        self.vectorizer = TfidfVectorizer(
            stop_words=PORTUGUESE_STOP_WORDS,
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            strip_accents='unicode'
        )
        
        try:
            self.X_tfidf = self.vectorizer.fit_transform(self.df['texto_processado'])
            self.feature_names = self.vectorizer.get_feature_names_out()
            print(f"Matriz TF-IDF criada com sucesso. Shape: {self.X_tfidf.shape}")
        except ValueError as e:
            print(f"ERRO na vetorização: {e}.")
            raise

    def perform_clustering(self, method: str = 'kmeans', **kwargs) -> Any:
        """
        Executa o algoritmo de clustering nos dados vetorizados.

        Args:
            method (str): O algoritmo a ser usado ('kmeans' ou 'hdbscan').
            **kwargs: Argumentos específicos do algoritmo (ex: n_clusters).

        Returns:
            O objeto do clusterizador treinado.
        """
        if self.X_tfidf is None: return None
        
        print(f"\n--- Etapa 3: Clustering com {method.upper()}")
        clusterer: Any

        if method == 'kmeans':
            n_clusters = kwargs.get('n_clusters', 8)
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            labels = clusterer.fit_predict(self.X_tfidf)
        elif method == 'hdbscan':
            X_dense = self.X_tfidf.toarray()
            min_cluster_size = kwargs.get('min_cluster_size', max(5, len(self.df) // 20))
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
            labels = clusterer.fit_predict(X_dense)
        else:
            raise ValueError("Método de clustering não suportado.")
        
        self.df['cluster'] = labels
        self._print_cluster_stats(labels)
        return clusterer

    def _print_cluster_stats(self, labels: np.ndarray):
        """
        (Auxiliar) Imprime estatísticas sobre os clusters gerados.
        """
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        
        print("\nEstatísticas do Clustering:")
        print(f"- Número de clusters encontrados: {n_clusters}")
        if n_noise > 0:
            print(f"- Documentos considerados ruído (outliers): {n_noise} ({n_noise / len(labels):.1%})")
        print("\nDistribuição de TCCs por cluster:")
        print(pd.Series(labels).value_counts().sort_index().to_string())
        print("-" * 35)

    def extract_cluster_keywords(self, top_n: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """
        Extrai as palavras-chave mais representativas de cada cluster com base
        no score médio de TF-IDF dos termos dentro de cada cluster.

        Args:
            top_n (int): O número de palavras-chave a serem extraídas por cluster.

        Returns:
            Um dicionário mapeando ID de cluster para sua lista de (palavra, score).
        """
        if self.df is None or 'cluster' not in self.df.columns or self.X_tfidf is None: return {}
        
        print("\n--- Etapa 4: Extração de Palavras-Chave por Tema")
        self.cluster_keywords = {}
        unique_clusters = sorted([c for c in self.df['cluster'].unique() if c != -1])
        
        for cluster_id in unique_clusters:
            cluster_mask = (self.df['cluster'] == cluster_id).to_numpy()
            mean_tfidf_in_cluster = np.asarray(self.X_tfidf[cluster_mask].mean(axis=0)).flatten()
            
            top_indices = mean_tfidf_in_cluster.argsort()[-top_n:][::-1]
            top_words = [(self.feature_names[i], mean_tfidf_in_cluster[i]) for i in top_indices]
            self.cluster_keywords[cluster_id] = top_words
            
        return self.cluster_keywords

    def name_clusters(self) -> Dict[int, str]:
        """
        Gera nomes descritivos para cada cluster. Tenta identificar um tema
        predefinido (ex: "Desenvolvimento Web") e, se não encontrar, usa as
        palavras-chave mais importantes como um nome genérico.

        Returns:
            Um dicionário mapeando ID de cluster para seu nome temático.
        """
        if not self.cluster_keywords:
            self.extract_cluster_keywords()

        print("\n--- Etapa 5: Nomeando os Temas (Clusters)")

        # Dicionário de temas com palavras-chave associadas (em minúsculas e lematizadas)
        THEME_KEYWORDS = {
            'Desenvolvimento Web & API': ['web', 'api', 'serviço', 'rest', 'front-end', 'backend', 'internet', 'site', 'online', 'plataforma', 'javascript', 'css', 'html', 'requisição'],
            'Ciência de Dados & I.A.': ['dado', 'aprendizagem', 'máquina', 'inteligência', 'artificial', 'análise', 'predição', 'modelo', 'rede', 'neural', 'algoritmo', 'classificação'],
            'Aplicações Móveis': ['móvel', 'aplicativo', 'android', 'ios', 'dispositivo', 'celular'],
            'Educação & Acessibilidade': ['educacional', 'ensino', 'aprendizagem', 'acessibilidade', 'educação', 'objeto', 'professor', 'aluno', 'ferramenta', 'apoio'],
            'Gestão & Processos': ['gestão', 'processo', 'gerenciamento', 'controle', 'monitoramento', 'empresarial', 'negócio', 'estoque', 'serviço'],
            'IoT, Redes & Hardware': ['iot', 'dispositivo', 'sensor', 'hardware', 'monitoramento', 'embarcado', 'automação', 'rede', 'segurança']
        }

        cluster_names = {}
        for cluster_id, keywords in self.cluster_keywords.items():
            # Pega as 10 palavras mais importantes para a identificação do tema
            top_words_only = [word for word, score in keywords[:10]]
            
            best_theme = None
            max_score = 0

            # Tenta encontrar o melhor tema predefinido contando palavras em comum
            for theme, theme_kws in THEME_KEYWORDS.items():
                score = len(set(top_words_only) & set(theme_kws))
                if score > max_score:
                    max_score = score
                    best_theme = theme
            
            # Se um tema for encontrado com uma pontuação mínima, usa-o.
            if max_score >= 2:
                cluster_names[cluster_id] = f"{best_theme} (C{cluster_id})"
            else:
                # Fallback: se nenhum tema claro for encontrado, usa as 2 palavras-chave mais importantes.
                fallback_words = [word.capitalize() for word, score in keywords[:2]]
                cluster_names[cluster_id] = f"Tema Geral: {', '.join(fallback_words)} (C{cluster_id})"
        
        cluster_names[-1] = "Ruído / Outliers"
        self.cluster_names = cluster_names
        self.df['cluster_name'] = self.df['cluster'].map(self.cluster_names)
        
        print("Nomes dos temas gerados:")
        for cid, name in self.cluster_names.items():
            if cid != -1: print(f"  Cluster {cid}: {name}")
        
        return cluster_names

    def reduce_dimensions(self, method: str = 'tsne', **kwargs):
        """
        Reduz a dimensionalidade da matriz TF-IDF para visualização 2D.

        Args:
            method (str): O método de redução ('tsne' ou 'pca').
            **kwargs: Argumentos adicionais para o método de redução.
        """
        if self.X_tfidf is None: return
        
        self.reduction_method = method
        print(f"\n--- Etapa 6: Redução de Dimensionalidade com {method.upper()}")
        X_dense = self.X_tfidf.toarray()
        
        if method == 'tsne':
            # Um passo de PCA intermediário ajuda a estabilizar e acelerar o t-SNE.
            n_pca_components = min(50, X_dense.shape[1], len(self.df) - 1)
            X_pre_tsne = PCA(n_components=n_pca_components, random_state=42).fit_transform(X_dense) if n_pca_components >= 2 else X_dense
            perplexity = min(kwargs.get('perplexity', 30.0), X_pre_tsne.shape[0] - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=max(5.0, perplexity))
            self.X_reduced = tsne.fit_transform(X_pre_tsne)
        elif method == 'pca':
            self.X_reduced = PCA(n_components=2, random_state=42).fit_transform(X_dense)
        else:
            raise ValueError("Método de redução não suportado.")
            
        self.df['dim_x'] = self.X_reduced[:, 0]
        self.df['dim_y'] = self.X_reduced[:, 1]
        print("Redução de dimensionalidade concluída.")

    def plot_clusters_interactive(self, output_dir: str):
        """
        Gera e salva um gráfico de dispersão interativo dos clusters (mapa de temas).

        Args:
            output_dir (str): O diretório para salvar o arquivo HTML do gráfico.
        """
        if self.df is None or 'dim_x' not in self.df.columns or 'cluster_name' not in self.df.columns: return

        print("\n[Visualização] Gerando mapa de temas interativo...")
        method_name = getattr(self, 'reduction_method', 'Redução').upper()
        xaxis_title = ""
        yaxis_title = ""
        hover_text = self.df.get('titulo', pd.Series(['' for _ in self.df.index]))
        
        fig = px.scatter(
            self.df, x='dim_x', y='dim_y', color='cluster_name',
            hover_name=hover_text, hover_data={'dim_x': False, 'dim_y': False, 'cluster_name': True},
            title='Visualização Interativa dos Clusters Temáticos de TCCs',
            color_discrete_map={"Ruído / Outliers": "lightgrey"},
            category_orders={"cluster_name": sorted(self.df['cluster_name'].unique())}
        )
        fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
        fig.update_layout(xaxis_title=xaxis_title, yaxis_title=yaxis_title, legend_title_text='Temas dos Clusters')
        fig.show()
        
        save_path = os.path.join(output_dir, "mapa_temas_interativo.html")
        os.makedirs(output_dir, exist_ok=True)
        fig.write_html(save_path)
        print(f"-> Gráfico interativo salvo em: {save_path}")

    def analyze_advisors(self, output_dir: str, top_n: int = 15):
        """
        Gera e salva análises gráficas individuais sobre os orientadores.

        Esta função cria e salva três visualizações separadas em arquivos HTML:
        1. Ranking de orientadores por número de TCCs orientados.
        2. Distribuição dos temas (clusters) por orientador.
        3. Histórico de orientações ao longo do tempo para os principais orientadores.

        Args:
            output_dir (str): O diretório para salvar os arquivos HTML dos gráficos.
            top_n (int): O número de orientadores a serem incluídos nas análises.
        """
        if self.df is None or 'orientador' not in self.df.columns:
            print("\nAVISO: A coluna 'orientador' não foi encontrada. Análises de orientador serão puladas.")
            return

        print("\n" + "="*50)
        print("--- Etapa 7: Análise de Orientadores (Gráficos Individuais) ---")
        print("="*50)

        # --- Preparação dos Dados ---
        # Garante que o diretório de saída exista
        os.makedirs(output_dir, exist_ok=True)
        
        df_analysis = self.df.copy()
        df_analysis['orientador_principal'] = df_analysis['orientador'].str.split(';').str[0].str.strip()
        
        if 'ano' in df_analysis.columns:
            df_analysis['ano'] = pd.to_numeric(df_analysis['ano'], errors='coerce')
            df_analysis.dropna(subset=['orientador_principal', 'ano'], inplace=True)
        else:
            df_analysis.dropna(subset=['orientador_principal'], inplace=True)

        top_advisors = df_analysis['orientador_principal'].value_counts().nlargest(top_n).index
        df_top = df_analysis[df_analysis['orientador_principal'].isin(top_advisors)]

        # --- 1. Gráfico: Quantidade de trabalhos orientados ---
        print(f"\n[Análise Orientadores 1/3] Gerando gráfico: Ranking de Orientações...")
        advisor_counts = df_top['orientador_principal'].value_counts().reset_index()
        advisor_counts.columns = ['Orientador', 'Quantidade']
        
        fig1 = px.bar(
            advisor_counts.sort_values('Quantidade', ascending=True),
            x='Quantidade',
            y='Orientador',
            orientation='h',
            title=f'Top {top_n} Orientadores por Número de TCCs Orientados',
            text='Quantidade',
            labels={'Quantidade': 'Número de TCCs', 'Orientador': 'Nome do Orientador'}
        )
        fig1.update_traces(textposition='outside')
        fig1.update_layout(yaxis={'categoryorder':'total ascending'})
        fig1.show()
        
        # Salvamento do Gráfico 1
        save_path1 = os.path.join(output_dir, "analise_orientadores_ranking.html")
        fig1.write_html(save_path1)
        print(f"-> Gráfico de ranking salvo em: '{save_path1}'")

        # --- 2. Gráfico: Temáticas que cada orientador orienta ---
        print(f"\n[Análise Orientadores 2/3] Gerando gráfico: Distribuição de Temas por Orientador...")
        themes_by_advisor = df_top.groupby(['orientador_principal', 'cluster_name']).size().reset_index(name='count')
        
        fig2 = px.bar(
            themes_by_advisor,
            x='orientador_principal',
            y='count',
            color='cluster_name',
            title=f'Distribuição de Temas por Orientador (Top {top_n})',
            labels={'count': 'Número de TCCs', 'orientador_principal': 'Orientador', 'cluster_name': 'Tema do TCC'},
            category_orders={"orientador_principal": advisor_counts['Orientador'].tolist()}
        )
        fig2.update_layout(xaxis={'categoryorder':'total descending'})
        fig2.show()
        
        # Salvamento do Gráfico 2
        save_path2 = os.path.join(output_dir, "analise_orientadores_temas.html")
        fig2.write_html(save_path2)
        print(f"-> Gráfico de temas salvo em: '{save_path2}'")


        # --- 3. Gráfico: Quantidade de trabalhos orientados por período ---
        if 'ano' in df_top.columns:
            print(f"\n[Análise Orientadores 3/3] Gerando gráfico: Histórico de Orientações por Ano...")
            orientations_per_year = df_top.groupby(['ano', 'orientador_principal']).size().reset_index(name='count')
            
            fig3 = px.line(
                orientations_per_year,
                x='ano',
                y='count',
                color='orientador_principal',
                title=f'Produção Anual por Orientador (Top {top_n})',
                markers=True,
                labels={'count': 'Número de TCCs', 'ano': 'Ano de Publicação', 'orientador_principal': 'Orientador'}
            )
            fig3.update_xaxes(dtick=1)
            fig3.show()

            # Salvamento do Gráfico 3
            save_path3 = os.path.join(output_dir, "analise_orientadores_tempo.html")
            fig3.write_html(save_path3)
            print(f"-> Gráfico de histórico salvo em: '{save_path3}'")

    def create_tcc_table(self, output_dir: str):
        """
        Gera e salva uma tabela interativa em HTML com os detalhes dos TCCs.

        A tabela inclui orientador, autor, ano, título e o tema (cluster) associado,
        e permite ordenação interativa por colunas.

        Args:
            output_dir (str): O diretório para salvar o arquivo HTML da tabela.
        """
        if self.df is None:
            print("\nAVISO: DataFrame não disponível. Tabela de TCCs não pode ser gerada.")
            return

        print("\n[Visualização] Gerando tabela interativa de TCCs...")

        # --- Preparação dos Dados para a Tabela ---
        # Seleciona e renomeia as colunas de interesse para a exibição
        cols_to_show = {
            'orientador': 'Orientador',
            'autor': 'Autor',
            'ano': 'Ano',
            'titulo': 'Título do TCC',
            'cluster_name': 'Tema (Cluster)'
        }
        
        # Garante que todas as colunas necessárias existam
        required_cols = [col for col in cols_to_show.keys() if col in self.df.columns]
        df_table = self.df[required_cols].rename(columns=cols_to_show)

        # Ordena a tabela para uma visualização inicial mais organizada
        df_table.sort_values(by=['Tema (Cluster)', 'Ano', 'Orientador'], inplace=True)

        # --- Criação da Figura da Tabela com Plotly ---
        fig = go.Figure(data=[go.Table(
            # Define os cabeçalhos da tabela
            header=dict(
                values=list(df_table.columns),
                fill_color='paleturquoise',
                align='left',
                font=dict(size=12, color='black')
            ),
            # Define as células com os dados do DataFrame
            cells=dict(
                values=[df_table[col] for col in df_table.columns],
                fill_color='lavender',
                align=['left', 'left', 'center', 'left', 'left'], # Alinhamento por coluna
                font=dict(size=11)
            ))
        ])

        # Ajusta o layout da tabela
        fig.update_layout(
            title_text="<b>Tabela de TCCs e Temas Associados</b>",
            height=800 # Aumenta a altura para exibir mais linhas
        )

        # Exibe a tabela interativa
        fig.show()

        # Salva a tabela como um arquivo HTML
        save_path = os.path.join(output_dir, "tabela_geral_tccs.html")
        os.makedirs(output_dir, exist_ok=True)
        fig.write_html(save_path)
        print(f"-> Tabela interativa de TCCs salva em: '{save_path}'")

    def create_wordclouds(self, output_dir: str, max_clusters_to_plot: int = 6):
        """
        Cria e salva nuvens de palavras para os clusters.

        Args:
            output_dir (str): O diretório para salvar o arquivo PNG.
            max_clusters_to_plot (int): O número máximo de nuvens a gerar.
        """
        if not self.cluster_keywords: return

        print("\n[Visualização Final] Gerando nuvens de palavras por tema...")
        valid_clusters = sorted(self.cluster_keywords.keys())
        n_to_plot = min(len(valid_clusters), max_clusters_to_plot)
        if n_to_plot == 0: return

        cols = min(3, n_to_plot)
        rows = (n_to_plot + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.array(axes).flatten()

        for i, cluster_id in enumerate(valid_clusters[:n_to_plot]):
            word_freq = {word: max(0.01, score) for word, score in self.cluster_keywords.get(cluster_id, [])}
            if not word_freq: continue
            
            cluster_title = self.cluster_names.get(cluster_id, f"Cluster {cluster_id}")
            wc = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(word_freq)
            axes[i].imshow(wc, interpolation='bilinear')
            axes[i].set_title(f"{cluster_title} ({sum(self.df.cluster == cluster_id)} TCCs)", fontsize=12)
            axes[i].axis('off')

        for i in range(n_to_plot, len(axes)): axes[i].axis('off')
        plt.tight_layout(pad=2.0)
        
        save_path = os.path.join(output_dir, "nuvem_palavras_temas.png")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"-> Nuvens de palavras salvas em: {save_path}")
        plt.show() # Este comando bloqueia a execução até a janela ser fechada

    def save_results(self, output_path: str):
        """
        Salva o DataFrame com os resultados do clustering em um arquivo.

        Args:
            output_path (str): Caminho do arquivo de saída.
        """
        if self.df is None: return

        print(f"\n--- Etapa Final: Salvando Resultados")
        cols_to_save = ['titulo', 'autor', 'orientador', 'ano', 'resumo', 'palavras_chave', 'cluster', 'cluster_name', 'dim_x', 'dim_y']
        output_df = self.df[[col for col in cols_to_save if col in self.df.columns]].copy()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if output_path.endswith('.csv'):
            output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        elif output_path.endswith('.json'):
            output_df.to_json(output_path, orient='records', indent=2, force_ascii=False)
        print(f"-> Resultados consolidados salvos com sucesso em: {output_path}")


# Função Principal de Execução
def main():
    """
    Função principal que orquestra e executa todo o pipeline de análise.
    """
    # Parâmetros da Análise (Ajuste aqui)
    DATA_PATH = "projeto_final/data/tccs_sistemas_internet_ifb.json"
    CLUSTERING_METHOD = 'kmeans'
    NUM_CLUSTERS_KMEANS = 6
    OUTPUT_DIR = "projeto_final/analysis"
    
    # Início do Pipeline
    print("="*80)
    print("INICIANDO PIPELINE DE ANÁLISE DE CLUSTERING DE TCCs")
    print("="*80)
    analyzer = TCCClusteringAnalyzer(data_path=DATA_PATH)
    if analyzer.df is None: return

    # Etapas de Processamento e Análise Temática
    analyzer.preprocess_text()
    analyzer.vectorize_texts()
    if analyzer.X_tfidf is None: return

    if CLUSTERING_METHOD == 'kmeans':
        analyzer.perform_clustering(method='kmeans', n_clusters=NUM_CLUSTERS_KMEANS)
    
    analyzer.extract_cluster_keywords(top_n=15)
    analyzer.name_clusters() 
    
    # Etapa de Visualização
    analyzer.reduce_dimensions()
    analyzer.plot_clusters_interactive(output_dir=OUTPUT_DIR)
    analyzer.analyze_advisors(output_dir=OUTPUT_DIR, top_n=15)
    analyzer.create_tcc_table(output_dir=OUTPUT_DIR)
    
    # Etapa de Salvamento dos Resultados
    analyzer.save_results(os.path.join(OUTPUT_DIR, "tccs_clusterizados_resultado.csv"))
    
    # Visualização Final (Nuvem de Palavras)
    analyzer.create_wordclouds(output_dir=OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("ANÁLISE COMPLETA CONCLUÍDA COM SUCESSO!")
    print(f"Todos os resultados foram salvos no diretório: '{OUTPUT_DIR}'")
    print("="*80)

if __name__ == "__main__":
    main()
