"""
Este script realiza web scraping de Trabalhos de Conclusão de Curso (TCCs) 
do curso de Sistemas para Internet do Instituto Federal de Brasília (IFB)
a partir do repositório BDTCBra (Omeka). Ele extrai metadados de cada TCC,
como título, autor, orientador, ano, resumo e link para o PDF, e salva os
dados coletados em arquivos CSV e JSON.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from urllib.parse import urljoin, urlparse
import logging
import json
import os

# --- Configurações ---
# Caminho para salvar o arquivo JSON de saída.
DATA_PATH_JSON = "projeto_final/data/tccs_sistemas_internet_ifb.json"
# Caminho para salvar o arquivo CSV de saída.
DATA_PATH_CSV = "projeto_final/data/tccs_sistemas_internet_ifb.csv"

# Configuração de logging para monitorar a execução do script.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TCCScraper:
    """
    Classe para realizar o scraping de TCCs de um repositório baseado em Omeka.
    """
    def __init__(self, base_url="https://bdtcbra.omeka.net"):
        """
        Inicializa o scraper.

        Args:
            base_url (str): A URL base do site Omeka.
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        self.tccs_data = []
        self.processed_urls = set()

    def get_page_content(self, url, retries=5):
        """
        Faz uma requisição HTTP GET para a URL especificada com retentativas.

        Args:
            url (str): A URL para acessar.
            retries (int): O número máximo de tentativas em caso de falha.

        Returns:
            requests.Response or None: O objeto de resposta se a requisição for
                                       bem-sucedida, caso contrário, None.
        """
        for attempt in range(retries):
            try:
                logger.info(f"Acessando: {url} (tentativa {attempt + 1})")
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                if len(response.content) < 100:
                    logger.warning(f"Resposta muito pequena para {url}")
                    continue
                return response
            except requests.RequestException as e:
                logger.warning(f"Tentativa {attempt + 1} falhou para {url}: {e}")
                if attempt == retries - 1:
                    logger.error(f"Falha ao acessar {url} após {retries} tentativas")
                    return None
                time.sleep(min(2 ** attempt, 10))  # Backoff exponencial
        return None

    def extract_tcc_links_from_page(self, page_url):
        """
        Extrai todos os links de TCCs de uma página de listagem.

        Args:
            page_url (str): A URL da página de listagem de TCCs.

        Returns:
            list: Uma lista de URLs absolutas para as páginas de detalhes dos TCCs.
        """
        response = self.get_page_content(page_url)
        if not response:
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        tcc_links = set()

        # Estratégias para encontrar links de TCCs
        item_patterns = [r'/items/show/\d+', r'/item/\d+', r'/items/\d+']
        for pattern in item_patterns:
            links = soup.find_all('a', href=re.compile(pattern))
            for link in links:
                full_url = urljoin(self.base_url, link['href'])
                tcc_links.add(full_url)

        # Remove duplicatas e URLs inválidas
        valid_links = [url for url in tcc_links if self.is_valid_tcc_url(url)]
        logger.info(f"Encontrados {len(valid_links)} links de TCCs na página {page_url}")
        return valid_links

    def is_valid_tcc_url(self, url):
        """
        Verifica se uma URL parece ser uma página de detalhes de TCC válida.

        Args:
            url (str): A URL a ser validada.

        Returns:
            bool: True se a URL for válida, False caso contrário.
        """
        try:
            parsed = urlparse(url)
            return (parsed.netloc and 'omeka' in parsed.netloc and '/items/show/' in parsed.path)
        except Exception:
            return False

    def extract_tcc_details(self, tcc_url):
        """
        Extrai os metadados de uma página de detalhes de um TCC.

        Args:
            tcc_url (str): A URL da página do TCC.

        Returns:
            dict or None: Um dicionário contendo os dados do TCC ou None se a
                          extração falhar.
        """
        if tcc_url in self.processed_urls:
            logger.info(f"URL já processada: {tcc_url}")
            return None
        self.processed_urls.add(tcc_url)

        response = self.get_page_content(tcc_url)
        if not response:
            return None

        soup = BeautifulSoup(response.content, 'html.parser')
        
        def clean_text(text):
            return re.sub(r'\s+', ' ', text.strip()) if text else ""

        tcc_data = {
            'titulo': '', 'autor': '', 'orientador': '', 'coorientador': '',
            'ano': '', 'curso': 'Tecnologia em Sistemas para Internet',
            'instituicao': 'Instituto Federal de Brasília - Campus Brasília',
            'palavras_chave': '', 'resumo': '', 'abstract': '',
            'link_acesso': tcc_url, 'arquivo_pdf': '', 'tipo_trabalho': 'TCC'
        }
        
        # Mapeamento de metadados Dublin Core para os campos do dicionário
        dublin_core_mapping = {
            'title': 'titulo', 'creator': 'autor', 'contributor': 'orientador',
            'date': 'ano', 'subject': 'palavras_chave', 'description': 'resumo',
            'abstract': 'abstract', 'advisor': 'orientador', 'co-advisor': 'coorientador'
        }

        # Extração baseada em seletores CSS e mapeamento Dublin Core
        for dc_field, data_field in dublin_core_mapping.items():
            selector = f'#dublin-core-{dc_field} .element-text'
            elements = soup.select(selector)
            if elements:
                texts = [clean_text(elem.get_text()) for elem in elements if clean_text(elem.get_text())]
                if texts:
                    tcc_data[data_field] = '; '.join(texts)

        # Extração de título (fallback)
        if not tcc_data['titulo']:
            title_tag = soup.select_one('h1.item-title, h1, title')
            if title_tag:
                tcc_data['titulo'] = clean_text(title_tag.get_text())

        # Extração específica de ano a partir do campo de data
        if tcc_data['ano']:
            year_match = re.search(r'(\d{4})', tcc_data['ano'])
            if year_match:
                tcc_data['ano'] = year_match.group(1)

        # Busca por links de PDF
        pdf_link = soup.select_one('a[href$=".pdf"], a[href*=".pdf"]')
        if pdf_link:
            tcc_data['arquivo_pdf'] = urljoin(self.base_url, pdf_link['href'])
        
        # Validação final
        if not tcc_data.get('titulo'):
            logger.warning(f"Título não encontrado para {tcc_url}")
            return None

        logger.info(f"TCC extraído: {tcc_data['titulo'][:50]}...")
        return tcc_data

    def get_all_pages_urls(self, start_url):
        """
        Encontra todas as URLs de paginação a partir de uma URL inicial.

        Args:
            start_url (str): A URL da primeira página da coleção.

        Returns:
            list: Uma lista ordenada de todas as URLs de página encontradas.
        """
        pages_to_visit = [start_url]
        visited_pages = set()
        all_pages = {start_url}

        while pages_to_visit:
            current_url = pages_to_visit.pop(0)
            if current_url in visited_pages:
                continue
            
            visited_pages.add(current_url)
            response = self.get_page_content(current_url)
            if not response:
                continue

            soup = BeautifulSoup(response.content, 'html.parser')
            pagination_links = soup.select('a[href*="page="]')
            for link in pagination_links:
                page_url = urljoin(self.base_url, link['href'])
                if page_url not in all_pages:
                    all_pages.add(page_url)
                    pages_to_visit.append(page_url)

        sorted_pages = sorted(list(all_pages))
        logger.info(f"Total de páginas encontradas: {len(sorted_pages)}")
        return sorted_pages

    def scrape_all_tccs(self, collection_url):
        """
        Orquestra o processo completo de scraping de todos os TCCs de uma coleção.

        Args:
            collection_url (str): A URL inicial da coleção de TCCs.

        Returns:
            list: Uma lista de dicionários, cada um representando um TCC.
        """
        logger.info("Iniciando scraping dos TCCs...")
        if not self.get_page_content(collection_url):
            logger.error("Não foi possível acessar a URL inicial. Verifique a conectividade.")
            return []

        all_pages = self.get_all_pages_urls(collection_url)
        all_tcc_links = set()

        for i, page_url in enumerate(all_pages, 1):
            logger.info(f"Processando página {i}/{len(all_pages)}: {page_url}")
            tcc_links = self.extract_tcc_links_from_page(page_url)
            all_tcc_links.update(tcc_links)
            time.sleep(1)

        unique_tcc_links = list(all_tcc_links)
        logger.info(f"Total de {len(unique_tcc_links)} TCCs únicos encontrados para processar.")
        
        successful_extractions = 0
        for i, tcc_url in enumerate(unique_tcc_links, 1):
            logger.info(f"Processando TCC {i}/{len(unique_tcc_links)}: {tcc_url}")
            tcc_data = self.extract_tcc_details(tcc_url)
            if tcc_data:
                self.tccs_data.append(tcc_data)
                successful_extractions += 1
            else:
                logger.warning(f"Falha na extração de dados para: {tcc_url}")
            time.sleep(1.5)

        logger.info(f"Scraping concluído! {successful_extractions}/{len(unique_tcc_links)} TCCs processados com sucesso.")
        return self.tccs_data
    
    def save_to_json(self, filepath=DATA_PATH_JSON):
        """
        Salva os dados coletados em um arquivo JSON.

        Args:
            filepath (str): O caminho completo do arquivo onde os dados serão salvos.
                            O padrão é 'projeto_final/data/tccs_sistemas_internet_ifb.json'.
        """
        if not self.tccs_data:
            logger.warning("Nenhum dado para salvar em JSON.")
            return
        
        try:
            # Garante que o diretório de destino exista
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.tccs_data, f, ensure_ascii=False, indent=4)
            logger.info(f"Dados salvos com sucesso em {filepath}")
        except IOError as e:
            logger.error(f"Erro ao salvar arquivo JSON em {filepath}: {e}")

    def save_to_csv(self, filepath=DATA_PATH_CSV):
        """
        Salva os dados coletados em um arquivo CSV.

        Args:
            filepath (str): O caminho completo do arquivo onde os dados serão salvos.
                            O padrão é 'projeto_final/data/tccs_sistemas_internet_ifb.csv'.
        """
        if not self.tccs_data:
            logger.warning("Nenhum dado para salvar em CSV.")
            return

        try:
            # Garante que o diretório de destino exista
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
            df = pd.DataFrame(self.tccs_data)
            df.to_csv(filepath, index=False, encoding='utf-8')
            logger.info(f"Dados salvos com sucesso em {filepath}")
        except IOError as e:
            logger.error(f"Erro ao salvar arquivo CSV em {filepath}: {e}")

def main():
    """
    Função principal para executar o scraper de TCCs.
    """
    scraper = TCCScraper()
    collection_url = "https://bdtcbra.omeka.net/items/browse?collection=9"
    
    try:
        tccs_data = scraper.scrape_all_tccs(collection_url)
        
        if not tccs_data:
            print("Nenhum TCC foi coletado. Verifique a URL e a estrutura do site.")
            return
        
        # Salva os resultados nos caminhos definidos
        scraper.save_to_json()
        scraper.save_to_csv()
        
        print("\n--- AMOSTRA DOS DADOS COLETADOS ---")
        for i, tcc in enumerate(tccs_data[:3]):  # Primeiros 3 TCCs
            print(f"\nTCC {i+1}:")
            for key, value in tcc.items():
                if value and str(value).strip():
                    print(f"  {key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
        
        print("\nScraping concluído com sucesso!")
        print(f"Arquivos salvos em: {os.path.dirname(DATA_PATH_JSON)}")

    except KeyboardInterrupt:
        logger.info("Scraping interrompido pelo usuário.")
        if scraper.tccs_data:
            print("Salvando dados parciais...")
            scraper.save_to_json()
            scraper.save_to_csv()
            print("Dados parciais salvos.")
    except Exception as e:
        logger.error(f"Ocorreu um erro inesperado durante o scraping: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        import requests
        import pandas as pd
        from bs4 import BeautifulSoup
    except ImportError as e:
        print(f"Erro de dependência: {e}")
        print("Instale as dependências com:")
        print("pip install requests beautifulsoup4 pandas lxml")
        exit(1)
    
    main()
