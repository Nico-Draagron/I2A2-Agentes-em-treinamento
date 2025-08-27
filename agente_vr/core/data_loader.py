"""
Módulo data_reader.py
Responsável pela leitura e padronização de todos os arquivos Excel do sistema VR.

Autor: Agente VR
Data: 2025-08
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union, Any

class DataReader:
    """
    Classe responsável pela leitura e padronização de arquivos Excel
    para o sistema de automação de VR.
    """
    def __init__(self, pasta_input: Optional[str] = None):
        """
        Inicializa o leitor de dados.
        
        Args:
            pasta_input: Caminho para a pasta com os arquivos Excel (opcional)
        """
        # Caminho absoluto para agente_vr/data/input, sempre relativo ao arquivo data_reader.py
        if pasta_input is None:
            # Vai para a pasta pai do módulo (agente_vr) e depois para data/input
            self.pasta_input = Path(__file__).parent.parent / "data" / "input"
        else:
            # Se o caminho passado for relativo, torna relativo à pasta do projeto
            p = Path(pasta_input)
            if not p.is_absolute():
                self.pasta_input = (Path(__file__).parent.parent / p)
            else:
                self.pasta_input = p
        self.logger = logging.getLogger(__name__)
        
        # Mapeamento de arquivos esperados
        self.arquivos_mapeamento = {
            'ativos': 'ATIVOS.xlsx',
            'ferias': 'FÉRIAS.xlsx', 
            'desligados': 'DESLIGADOS.xlsx',
            'admissoes': 'ADMISSÃO ABRIL.xlsx',
            'sindicatos_valores': 'Base sindicato x valor.xlsx',
            'dias_uteis': 'Base dias uteis.xlsx',
            'afastamentos': 'AFASTAMENTOS.xlsx',
            'aprendizes': 'APRENDIZ.xlsx',
            'estagiarios': 'ESTÁGIO.xlsx',
            'exterior': 'EXTERIOR.xlsx'
        }
        
    def carregar_todas_bases(self) -> Dict[str, Any]:
        """
        Carrega todas as bases de dados necessárias para o processamento de VR.
        
        Returns:
            Dict com todas as bases organizadas e padronizadas
        """
        self.logger.info("🔄 Iniciando carregamento de todas as bases...")
        bases = {}
        try:
            # Bases principais
            bases['ativos'] = self.carregar_ativos()
            bases['ferias'] = self.carregar_ferias()
            bases['desligados'] = self.carregar_desligados()
            bases['admissoes'] = self.carregar_admissoes()
            # Bases de configuração
            bases['sindicatos_valores'] = self.carregar_sindicatos_valores()
            bases['dias_uteis'] = self.carregar_dias_uteis()
            # Bases de exclusão
            bases['exclusoes'] = {
                'afastamentos': self.carregar_afastamentos(),
                'aprendizes': self.carregar_aprendizes(),
                'estagiarios': self.carregar_estagiarios(),
                'exterior': self.carregar_exterior()
            }
            self.logger.info("✅ Todas as bases carregadas com sucesso!")
            self._log_resumo_bases(bases)
            return bases
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar bases: {e}")
            raise
    
    def carregar_ativos(self) -> pd.DataFrame:
        """
        Carrega a base de colaboradores ativos.
        
        Returns:
            DataFrame com colunas padronizadas: matricula, empresa, cargo, situacao, sindicato
        """
        self.logger.info("📊 Carregando base ATIVOS...")
        
        arquivo = self.pasta_input / self.arquivos_mapeamento['ativos']
        df = self._ler_excel_seguro(arquivo, sheet_name='ATIVOS')
        
        # Padronizar colunas
        df = df.rename(columns={
            'MATRICULA': 'matricula',
            'EMPRESA': 'empresa', 
            'TITULO DO CARGO': 'cargo',
            'DESC. SITUACAO': 'situacao',
            'Sindicato': 'sindicato'
        })
        
        # Validações básicas
        df = self._validar_matriculas(df, 'ATIVOS')
        df = self._limpar_sindicatos(df)
        
        self.logger.info(f"✅ ATIVOS carregado: {len(df)} registros")
        return df
    
    def carregar_ferias(self) -> pd.DataFrame:
        """
        Carrega a base de colaboradores em férias.
        
        Returns:
            DataFrame com colunas: matricula, situacao, dias_ferias
        """
        self.logger.info("🏖️ Carregando base FÉRIAS...")
        
        arquivo = self.pasta_input / self.arquivos_mapeamento['ferias']
        df = self._ler_excel_seguro(arquivo)
        
        # Padronizar colunas
        df = df.rename(columns={
            'MATRICULA': 'matricula',
            'DESC. SITUACAO': 'situacao',
            'DIAS DE FÉRIAS': 'dias_ferias'
        })
        
        # Validações
        df = self._validar_matriculas(df, 'FÉRIAS')
        
        # Converter dias de férias para numérico
        df['dias_ferias'] = pd.to_numeric(df['dias_ferias'], errors='coerce').fillna(0)
        
        self.logger.info(f"✅ FÉRIAS carregado: {len(df)} registros")
        return df
    
    def carregar_desligados(self) -> pd.DataFrame:
        """
        Carrega a base de colaboradores desligados.
        
        Returns:
            DataFrame com colunas: matricula, data_demissao, comunicado_ok
        """
        self.logger.info("👋 Carregando base DESLIGADOS...")
        
        arquivo = self.pasta_input / self.arquivos_mapeamento['desligados']
        df = self._ler_excel_seguro(arquivo, sheet_name='DESLIGADOS ')
        
        # Padronizar colunas (note os espaços extras no original)
        df = df.rename(columns={
            'MATRICULA ': 'matricula',
            'DATA DEMISSÃO': 'data_demissao',
            'COMUNICADO DE DESLIGAMENTO': 'comunicado_ok'
        })
        
        # Validações
        df = self._validar_matriculas(df, 'DESLIGADOS')
        
        # Converter datas do formato Excel
        df['data_demissao'] = self._converter_data_excel(df['data_demissao'])
        
        # Padronizar campo comunicado
        df['comunicado_ok'] = df['comunicado_ok'].str.upper().str.strip()
        
        self.logger.info(f"✅ DESLIGADOS carregado: {len(df)} registros")
        return df
    
    def carregar_admissoes(self) -> pd.DataFrame:
        """
        Carrega a base de colaboradores admitidos no mês.
        
        Returns:
            DataFrame com colunas: matricula, data_admissao, cargo
        """
        self.logger.info("🆕 Carregando base ADMISSÕES...")
        
        arquivo = self.pasta_input / self.arquivos_mapeamento['admissoes']
        df = self._ler_excel_seguro(arquivo)
        
        # Padronizar colunas
        df = df.rename(columns={
            'MATRICULA': 'matricula',
            'Admissão': 'data_admissao',
            'Cargo': 'cargo'
        })
        
        # Validações
        df = self._validar_matriculas(df, 'ADMISSÕES')
        
        # Converter datas do formato Excel
        df['data_admissao'] = self._converter_data_excel(df['data_admissao'])
        
        self.logger.info(f"✅ ADMISSÕES carregado: {len(df)} registros")
        return df
    
    def carregar_sindicatos_valores(self) -> pd.DataFrame:
        """
        Carrega a base de valores por sindicato/estado.
        
        Returns:
            DataFrame com colunas: estado, valor_diario
        """
        self.logger.info("💰 Carregando base SINDICATOS x VALORES...")
        
        arquivo = self.pasta_input / self.arquivos_mapeamento['sindicatos_valores']
        df = self._ler_excel_seguro(arquivo)
        
        # O arquivo tem headers mal formatados, vamos corrigir
        if len(df.columns) >= 2:
            # Renomear para colunas padronizadas
            df.columns = ['estado', 'valor_diario']  # type: ignore
            
            # Limpar dados
            df['estado'] = df['estado'].str.strip()
            df['valor_diario'] = pd.to_numeric(df['valor_diario'], errors='coerce')
            
            # Remover linhas inválidas
            df = df.dropna(subset=['estado', 'valor_diario'])
            
        else:
            raise ValueError("Arquivo de sindicatos x valores com estrutura inválida")
        
        self.logger.info(f"✅ SINDICATOS x VALORES carregado: {len(df)} registros")
        return df
    
    def carregar_dias_uteis(self) -> pd.DataFrame:
        """
        Carrega a base de dias úteis por sindicato.
        
        Returns:
            DataFrame com colunas: sindicato, dias_uteis
        """
        self.logger.info("📅 Carregando base DIAS ÚTEIS...")
        
        arquivo = self.pasta_input / self.arquivos_mapeamento['dias_uteis']
        df = self._ler_excel_seguro(arquivo)
        
        # Corrigir headers mal formatados
        if len(df.columns) >= 2:
            df.columns = ['sindicato', 'dias_uteis']
            
            # Limpar primeira linha se for header
            if df.iloc[0]['sindicato'] == 'SINDICADO':
                df = df.drop(0).reset_index(drop=True)
            
            # Limpar dados
            df['sindicato'] = df['sindicato'].str.strip()
            df['dias_uteis'] = pd.to_numeric(df['dias_uteis'], errors='coerce')
            
            # Remover linhas inválidas
            df = df.dropna(subset=['sindicato', 'dias_uteis'])
            
        else:
            raise ValueError("Arquivo de dias úteis com estrutura inválida")
        
        self.logger.info(f"✅ DIAS ÚTEIS carregado: {len(df)} registros")
        return df
    
    def carregar_afastamentos(self) -> pd.DataFrame:
        """
        Carrega a base de colaboradores afastados.
        
        Returns:
            DataFrame com colunas: matricula, situacao
        """
        return self._carregar_base_exclusao('afastamentos', 'AFASTAMENTOS')
    
    def carregar_aprendizes(self) -> pd.DataFrame:
        """
        Carrega a base de aprendizes (para exclusão).
        
        Returns:
            DataFrame com colunas: matricula, cargo
        """
        return self._carregar_base_exclusao('aprendizes', 'APRENDIZES')
    
    def carregar_estagiarios(self) -> pd.DataFrame:
        """
        Carrega a base de estagiários (para exclusão).
        
        Returns:
            DataFrame com colunas: matricula, cargo
        """
        return self._carregar_base_exclusao('estagiarios', 'ESTAGIÁRIOS')
    
    def carregar_exterior(self) -> pd.DataFrame:
        """
        Carrega a base de colaboradores no exterior (para exclusão).
        
        Returns:
            DataFrame com colunas: matricula, valor, situacao
        """
        self.logger.info("🌍 Carregando base EXTERIOR...")
        
        arquivo = self.pasta_input / self.arquivos_mapeamento['exterior']
        df = self._ler_excel_seguro(arquivo)
        
        # Padronizar colunas
        if len(df.columns) >= 3:
            df.columns = ['matricula', 'valor', 'situacao']
        else:
            df.columns = ['matricula', 'valor']
            df['situacao'] = 'exterior'
        
        # Validações
        df = self._validar_matriculas(df, 'EXTERIOR')
        
        self.logger.info(f"✅ EXTERIOR carregado: {len(df)} registros")
        return df
    
    def _carregar_base_exclusao(self, tipo: str, nome_log: str) -> pd.DataFrame:
        """
        Método auxiliar para carregar bases de exclusão.
        
        Args:
            tipo: Tipo da base (ex: 'afastamentos', 'aprendizes')
            nome_log: Nome para logs
            
        Returns:
            DataFrame padronizado
        """
        self.logger.info(f"🚫 Carregando base {nome_log}...")
        
        arquivo = self.pasta_input / self.arquivos_mapeamento[tipo]
        df = self._ler_excel_seguro(arquivo)
        
        # Padronizar colunas baseado na estrutura
        if len(df.columns) == 2:
            if 'CARGO' in df.columns[1].upper():
                df.columns = ['matricula', 'cargo']
            else:
                df.columns = ['matricula', 'situacao']
        else:
            df.columns = ['matricula'] + [f'col_{i}' for i in range(1, len(df.columns))]
        
        # Validações
        df = self._validar_matriculas(df, nome_log)
        
        self.logger.info(f"✅ {nome_log} carregado: {len(df)} registros")
        return df
    
    def _ler_excel_seguro(self, arquivo: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Lê arquivo Excel com tratamento de erros.
        
        Args:
            arquivo: Caminho do arquivo
            sheet_name: Nome da planilha (opcional)
            
        Returns:
            DataFrame lido
        """
        if not arquivo.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {arquivo}")
        
        try:
            # Ler Excel com parâmetros robustos
            result = pd.read_excel(
                arquivo,
                sheet_name=sheet_name,
                dtype=str,  # Ler tudo como string inicialmente
                na_filter=False  # Não converter valores para NaN automaticamente
            )
            
            # Se sheet_name não foi especificado e retornou um dict, pegar a primeira planilha
            if isinstance(result, dict):
                df = next(iter(result.values()))  # Primeira planilha
            else:
                df = result
            
            # Remover linhas completamente vazias
            df = df.dropna(how='all')
            
            # Remover colunas completamente vazias
            df = df.dropna(axis=1, how='all')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Erro ao ler {arquivo}: {e}")
            raise
    
    def _validar_matriculas(self, df: pd.DataFrame, nome_base: str) -> pd.DataFrame:
        """
        Valida e limpa matrículas.
        
        Args:
            df: DataFrame a validar
            nome_base: Nome da base para logs
            
        Returns:
            DataFrame com matrículas válidas
        """
        if 'matricula' not in df.columns:
            raise ValueError(f"Coluna 'matricula' não encontrada em {nome_base}")
        
        # Converter para numérico
        df['matricula'] = pd.to_numeric(df['matricula'], errors='coerce')
        
        # Remover matrículas inválidas
        antes = len(df)
        df = df.dropna(subset=['matricula'])
        df['matricula'] = df['matricula'].astype(int)
        depois = len(df)
        
        if antes != depois:
            self.logger.warning(f"⚠️ {nome_base}: {antes - depois} registros com matrícula inválida removidos")
        
        return df
    
    def _converter_data_excel(self, serie_datas: pd.Series) -> pd.Series:
        """
        Converte datas do formato Excel (número) para datetime.
        
        Args:
            serie_datas: Série com datas em formato Excel
            
        Returns:
            Série com datas convertidas
        """
        def converter_data(valor):
            try:
                if pd.isna(valor) or valor == '':
                    return None
                
                # Se já é datetime, retornar como está
                if isinstance(valor, datetime):
                    return valor
                
                # Converter número Excel para datetime
                if isinstance(valor, (int, float)):
                    # Excel usa 1900-01-01 como base (com bug do ano 1900)
                    return datetime(1899, 12, 30) + timedelta(days=valor)
                
                # Tentar converter string
                return pd.to_datetime(valor, errors='coerce')
                
            except:
                return None
        
        return serie_datas.apply(converter_data)
    
    def _limpar_sindicatos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpa e padroniza nomes de sindicatos.
        
        Args:
            df: DataFrame com coluna sindicato
            
        Returns:
            DataFrame com sindicatos limpos
        """
        if 'sindicato' in df.columns:
            df['sindicato'] = df['sindicato'].str.strip()
            df['sindicato'] = df['sindicato'].fillna('SEM_SINDICATO')
        
        return df
    
    def _log_resumo_bases(self, bases: Dict[str, Any]) -> None:
        """
        Registra resumo das bases carregadas.
        
        Args:
            bases: Dicionário com todas as bases
        """
        self.logger.info("📊 RESUMO DAS BASES CARREGADAS:")
        self.logger.info(f"   📋 Ativos: {len(bases['ativos'])} registros")
        self.logger.info(f"   🏖️ Férias: {len(bases['ferias'])} registros")
        self.logger.info(f"   👋 Desligados: {len(bases['desligados'])} registros")
        self.logger.info(f"   🆕 Admissões: {len(bases['admissoes'])} registros")
        self.logger.info(f"   💰 Sindicatos: {len(bases['sindicatos_valores'])} registros")
        self.logger.info(f"   📅 Dias úteis: {len(bases['dias_uteis'])} registros")
        
        total_exclusoes = sum(len(base) for base in bases['exclusoes'].values())
        self.logger.info(f"   🚫 Total exclusões: {total_exclusoes} registros")


# Função utilitária para uso direto
def carregar_bases_vr(pasta_input: str = "data/input") -> Dict[str, pd.DataFrame]:
    """
    Função utilitária para carregar todas as bases de VR.
    
    Args:
        pasta_input: Pasta com os arquivos Excel
        
    Returns:
        Dicionário com todas as bases carregadas
    """
    reader = DataReader(pasta_input)
    return reader.carregar_todas_bases()


# Exemplo de uso
if __name__ == "__main__":
    import logging
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Testar carregamento
    try:
        bases = carregar_bases_vr()
        print("✅ Todas as bases carregadas com sucesso!")
        
        # Mostrar resumo
        for nome, df in bases.items():
            if nome != 'exclusoes':
                print(f"{nome}: {len(df)} registros")
            else:
                for sub_nome, sub_df in df.items():
                    print(f"exclusoes.{sub_nome}: {len(sub_df)} registros")
                    

    except Exception as e:
        print(f"❌ Erro: {e}")

# Alias para compatibilidade com o main.py
DataLoader = DataReader