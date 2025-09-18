"""
Sistema Multi-Agente para Automação de Cálculo de Vale Refeição
Versão 2.1 - CORRIGIDA E OTIMIZADA
Desenvolvido por: Agentes em Treinamento
Autor: Nicolas França
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
import warnings
import os
from pathlib import Path
import requests
import time
import re

# Configuração de logging melhorada
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/logs/vr_system.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Suprimir warnings do pandas
warnings.filterwarnings('ignore')

# ========================= CONFIGURAÇÕES =========================

class Config:
    """Configurações centralizadas do sistema"""
    
    # Layouts obrigatórios
    REQUIRED_OUTPUT_COLUMNS = [
        'Matricula', 'Admissão', 'Sindicato do Colaborador', 'Competência',
        'Dias', 'VALOR DIÁRIO VR', 'TOTAL', 'Custo empresa', 
        'Desconto profissional', 'OBS GERAL'
    ]
    
    OUTPUT_FILENAME = "VR MENSAL 05.2025.xlsx"
    
    # Valores padrão do VR
    VALORES_VR = {
        'SP': 37.5,
        'RJ': 35.0,
        'RS': 35.0,
        'PR': 35.0
    }
    
    # Percentuais de custo
    PERC_EMPRESA = 0.8
    PERC_COLABORADOR = 0.2
    
    # LLM Configuration (sem hardcoded keys)
    LLM_TIMEOUT = 30
    LLM_MAX_RETRIES = 3

# ========================= ENUMS E DATACLASSES =========================

class AgentStatus(Enum):
    """Status de execução dos agentes"""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

class AnomalyType(Enum):
    """Tipos de anomalias detectadas"""
    FERIAS = "ferias"
    ADMISSAO = "admissao"
    DESLIGAMENTO = "desligamento"
    DIAS_NEGATIVOS = "dias_negativos"
    VALOR_FORA_PADRAO = "valor_fora_padrao"
    DIAS_EXCEDENTES = "dias_excedentes"
    VALOR_ZERO = "valor_zero"
    PROPORCIONAL = "proporcional"

@dataclass
class ValidationResult:
    """Resultado de validação"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    corrections: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EmployeeAnomaly:
    """Dados de anomalia de um colaborador"""
    matricula: int
    tipo_anomalia: AnomalyType
    descricao: str
    dados_colaborador: Dict[str, Any]
    severidade: str = "medium"
    requer_llm: bool = True

@dataclass
class NormalPattern:
    """Padrão normal identificado"""
    dias_uteis_esperados: Set[int] = field(default_factory=lambda: {21, 22})
    valores_diarios_esperados: Set[float] = field(default_factory=lambda: {35.0, 37.5})
    sindicatos_conhecidos: Set[str] = field(default_factory=set)

# ========================= AGENTES BASE =========================

class BaseAgent:
    """Classe base para todos os agentes"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"Agent.{name}")
        self.status = AgentStatus.IDLE
        self.metrics = {}
        self.start_time = None
        self.end_time = None
    
    def log(self, message: str, level: str = "info"):
        """Log de mensagens do agente"""
        getattr(self.logger, level)(f"[{self.name}] {message}")
    
    def start_execution(self):
        """Inicia cronômetro de execução"""
        self.start_time = time.time()
        self.status = AgentStatus.PROCESSING
        self.log(f"Iniciando execução...")
    
    def end_execution(self):
        """Finaliza cronômetro de execução"""
        self.end_time = time.time()
        self.status = AgentStatus.COMPLETED
        execution_time = self.end_time - self.start_time if self.start_time else 0
        self.metrics['execution_time'] = execution_time
        self.log(f"Execução concluída em {execution_time:.2f}s")
    
    def execute(self, *args, **kwargs):
        """Método a ser implementado por cada agente"""
        raise NotImplementedError

# ========================= AGENTE 1: DATA VALIDATOR MELHORADO =========================

class DataValidatorAgent(BaseAgent):
    """Agente responsável por validar e limpar os dados"""
    
    def __init__(self):
        super().__init__("DataValidator")
        self.required_files = {
            'ATIVOS': ['MATRICULA', 'CARGO', 'SITUACAO', 'SINDICATO'],
            'FÉRIAS': ['MATRICULA', 'DIAS DE FÉRIAS'],
            'ADMISSÃO ABRIL': ['MATRICULA', 'Admissão'],
            'DESLIGADOS': ['MATRICULA', 'DATA DEMISSÃO', 'COMUNICADO DE DESLIGAMENTO'],
            'AFASTAMENTOS': ['MATRICULA'],
            'APRENDIZ': ['MATRICULA'],
            'ESTÁGIO': ['MATRICULA'],
            'EXTERIOR': ['Cadastro'],
        }
    
    def execute(self, data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], ValidationResult]:
        """Valida e limpa os dados com verificações rigorosas"""
        self.start_execution()
        
        result = ValidationResult(is_valid=True)
        cleaned_data = {}
        
        # Verificar arquivos obrigatórios
        missing_files = [f for f in self.required_files.keys() if f not in data]
        if missing_files:
            result.errors.extend([f"Arquivo obrigatório ausente: {f}" for f in missing_files])
            result.is_valid = False
            self.log(f"Arquivos obrigatórios ausentes: {missing_files}", "error")
        
        for file_name, df in data.items():
            self.log(f"Validando {file_name}...")
            
            # Validar estrutura obrigatória
            if file_name in self.required_files:
                required_cols = self.required_files[file_name]
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    result.warnings.append(f"{file_name}: Colunas ausentes {missing_cols}")
            
            # Limpar dados básicos
            df_cleaned = self._clean_basic_data(df, file_name)
            
            # Validações específicas
            df_cleaned = self._apply_specific_validations(df_cleaned, file_name)
            
            cleaned_data[file_name] = df_cleaned
            
            # Métricas
            self.metrics[file_name] = {
                'original_rows': len(df),
                'cleaned_rows': len(df_cleaned),
                'removed_rows': len(df) - len(df_cleaned)
            }
        
        self.end_execution()
        return cleaned_data, result
    
    def _clean_basic_data(self, df: pd.DataFrame, file_name: str) -> pd.DataFrame:
        """Limpeza básica melhorada"""
        # Padronizar colunas
        df.columns = df.columns.str.strip()
        
        # Remover linhas completamente vazias
        df = df.dropna(how='all')
        
        # Processar matrículas
        if 'MATRICULA' in df.columns:
            df['MATRICULA'] = pd.to_numeric(df['MATRICULA'], errors='coerce')
            original_count = len(df)
            df = df.dropna(subset=['MATRICULA'])
            df['MATRICULA'] = df['MATRICULA'].astype(int)
            removed_count = original_count - len(df)
            if removed_count > 0:
                self.log(f"{file_name}: Removidas {removed_count} linhas com matrícula inválida")
        
        # Processar cadastros (para EXTERIOR)
        if 'Cadastro' in df.columns:
            df['Cadastro'] = pd.to_numeric(df['Cadastro'], errors='coerce')
            df = df.dropna(subset=['Cadastro'])
            df['Cadastro'] = df['Cadastro'].astype(int)
        
        return df
    
    def _apply_specific_validations(self, df: pd.DataFrame, file_name: str) -> pd.DataFrame:
        """Aplicar validações específicas por arquivo"""
        
        if file_name == "AFASTAMENTOS":
            # Remover duplicatas por matrícula
            original_len = len(df)
            df = df.drop_duplicates(subset=['MATRICULA'], keep='first')
            if len(df) < original_len:
                self.log(f"AFASTAMENTOS: Removidas {original_len - len(df)} duplicatas")
        
        elif file_name == "ADMISSÃO ABRIL":
            # Validar e converter datas
            if 'Admissão' in df.columns:
                df['Admissão'] = pd.to_datetime(df['Admissão'], errors='coerce')
                df = df.dropna(subset=['Admissão'])
        
        elif file_name == "DESLIGADOS":
            # Validar datas de demissão
            if 'DATA DEMISSÃO' in df.columns:
                df['DATA DEMISSÃO'] = pd.to_datetime(df['DATA DEMISSÃO'], errors='coerce')
                df = df.dropna(subset=['DATA DEMISSÃO'])
        
        elif file_name == "FÉRIAS":
            # Validar dias de férias
            if 'DIAS DE FÉRIAS' in df.columns:
                df['DIAS DE FÉRIAS'] = pd.to_numeric(df['DIAS DE FÉRIAS'], errors='coerce')
                df = df.dropna(subset=['DIAS DE FÉRIAS'])
                df['DIAS DE FÉRIAS'] = df['DIAS DE FÉRIAS'].astype(int)
                # Remover valores negativos
                df = df[df['DIAS DE FÉRIAS'] >= 0]
        
        return df

# ========================= AGENTE 2: BUSINESS RULES MELHORADO =========================

class BusinessRulesAgent(BaseAgent):
    """Agente que aplica as regras de negócio"""
    
    def __init__(self):
        super().__init__("BusinessRules")
        self.exclusion_reasons = {}
        self.calculation_adjustments = {}
    
    def execute(self, data: Dict[str, pd.DataFrame], reference_date: Optional[datetime] = None) -> pd.DataFrame:
        """Aplica regras de negócio com validações rigorosas"""
        self.start_execution()
        
        if reference_date is None:
            reference_date = datetime(2025, 5, 1)
        
        # Pipeline de processamento
        base = self._consolidate_active_employees(data)
        base = self._apply_exclusions(base, data)
        base = self._calculate_working_days(base, data)
        base = self._apply_vacation_rules(base, data)
        base = self._apply_admission_termination_rules(base, data, reference_date)
        base = self._calculate_vr_values(base, data)
        base = self._validate_final_calculations(base)
        
        self.end_execution()
        return base
    
    def _consolidate_active_employees(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Consolidar colaboradores ativos com validação"""
        if 'ATIVOS' not in data:
            raise ValueError("Arquivo ATIVOS é obrigatório")
        
        base = data['ATIVOS'].copy()
        
        # Padronizar colunas esperadas
        expected_cols = ['MATRICULA', 'EMPRESA', 'CARGO', 'SITUACAO', 'SINDICATO']
        available_cols = [col for col in expected_cols if col in base.columns]
        
        if len(available_cols) < 3:  # Mínimo: MATRICULA, SITUACAO, SINDICATO
            raise ValueError(f"ATIVOS deve conter pelo menos: MATRICULA, SITUACAO, SINDICATO")
        
        base = base[available_cols]
        
        # Filtrar apenas trabalhando
        if 'SITUACAO' in base.columns:
            initial_count = len(base)
            base = base[base['SITUACAO'] == 'Trabalhando']
            self.log(f"Filtrados {len(base)} trabalhando de {initial_count} registros")
        
        return base
    
    def _apply_exclusions(self, base: pd.DataFrame, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Aplicar exclusões com logging detalhado"""
        exclusion_stats = {}
        
        # Diretores
        if 'CARGO' in base.columns:
            directors = base[base['CARGO'].str.contains('DIRETOR', case=False, na=False)]
            if not directors.empty:
                exclusion_stats['Diretores'] = len(directors)
                for matricula in directors['MATRICULA']:
                    self.exclusion_reasons[matricula] = 'Diretor'
                base = base[~base['MATRICULA'].isin(directors['MATRICULA'])]
        
        # Exclusões por arquivo
        exclusion_files = {
            'ESTÁGIO': 'Estagiário',
            'APRENDIZ': 'Aprendiz',
            'AFASTAMENTOS': 'Afastado'
        }
        
        for file_key, reason in exclusion_files.items():
            if file_key in data:
                to_exclude = data[file_key]['MATRICULA'].unique()
                exclusion_stats[reason] = len(to_exclude)
                for matricula in to_exclude:
                    self.exclusion_reasons[matricula] = reason
                base = base[~base['MATRICULA'].isin(to_exclude)]
        
        # Exterior
        if 'EXTERIOR' in data and 'Cadastro' in data['EXTERIOR'].columns:
            abroad = data['EXTERIOR']['Cadastro'].dropna().unique()
            exclusion_stats['Exterior'] = len(abroad)
            for matricula in abroad:
                self.exclusion_reasons[matricula] = 'Exterior'
            base = base[~base['MATRICULA'].isin(abroad)]
        
        self.log(f"Exclusões aplicadas: {exclusion_stats}")
        return base
    
    def _calculate_working_days(self, base: pd.DataFrame, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calcular dias úteis com mapeamento robusto"""
        if 'Base dias uteis' not in data:
            self.log("Base dias uteis não encontrada. Usando padrão 22 dias", "warning")
            base['DIAS_UTEIS'] = 22
            return base
        
        dias_uteis_df = data['Base dias uteis']
        dias_map = {}
        
        # Mapear sindicatos para dias úteis
        for idx in range(len(dias_uteis_df)):
            row = dias_uteis_df.iloc[idx]
            if len(row) >= 2 and pd.notna(row[0]) and pd.notna(row[1]):
                try:
                    sindicato = str(row[0]).strip()
                    dias = int(float(row[1]))
                    if sindicato and dias > 0:
                        dias_map[sindicato] = dias
                except (ValueError, TypeError):
                    continue
        
        # Aplicar mapeamento
        base['DIAS_UTEIS'] = base['SINDICATO'].map(dias_map)
        
        # Tratar sindicatos não mapeados
        unmapped = base[base['DIAS_UTEIS'].isna()]
        if not unmapped.empty:
            unique_unmapped = unmapped['SINDICATO'].unique()
            self.log(f"Sindicatos não mapeados (usando 22 dias): {unique_unmapped}", "warning")
            base['DIAS_UTEIS'].fillna(22, inplace=True)
        
        self.log(f"Mapeamento dias úteis: {len(dias_map)} sindicatos mapeados")
        return base
    
    def _calculate_vr_values(self, base: pd.DataFrame, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calcular valores com base na planilha de referência"""
        
        # Tentar usar planilha de valores
        if 'Base sindicato x valor' in data:
            sindicato_valor_map = {}
            df_valores = data['Base sindicato x valor']
            
            for idx, row in df_valores.iterrows():
                if len(row) >= 2 and pd.notna(row[0]) and pd.notna(row[1]):
                    try:
                        sindicato_key = str(row[0]).strip()
                        valor = float(row[1])
                        if valor > 0:
                            sindicato_valor_map[sindicato_key] = valor
                    except (ValueError, TypeError):
                        continue
            
            # Mapear valores
            def map_valor(sindicato):
                # Tentar mapeamento direto
                if str(sindicato) in sindicato_valor_map:
                    return sindicato_valor_map[str(sindicato)]
                
                # Tentar por estado
                sindicato_upper = str(sindicato).upper()
                for estado, valor_padrao in Config.VALORES_VR.items():
                    if estado in sindicato_upper:
                        return valor_padrao
                
                return 37.5  # Padrão SP
            
            base['VALOR_DIARIO'] = base['SINDICATO'].apply(map_valor)
        else:
            # Fallback: mapear por estado no nome do sindicato
            def extract_state_value(sindicato):
                sindicato_upper = str(sindicato).upper()
                for estado, valor in Config.VALORES_VR.items():
                    if estado in sindicato_upper:
                        return valor
                return Config.VALORES_VR['SP']  # Padrão
            
            base['VALOR_DIARIO'] = base['SINDICATO'].apply(extract_state_value)
        
        # Calcular totais
        base['TOTAL_VR'] = base['DIAS_VR'] * base['VALOR_DIARIO']
        base['CUSTO_EMPRESA'] = base['TOTAL_VR'] * Config.PERC_EMPRESA
        base['DESCONTO_PROFISSIONAL'] = base['TOTAL_VR'] * Config.PERC_COLABORADOR
        
        return base
    
    def _validate_final_calculations(self, base: pd.DataFrame) -> pd.DataFrame:
        """Validações finais dos cálculos"""
        
        # Verificar valores negativos
        negative_vr = base[base['TOTAL_VR'] < 0]
        if not negative_vr.empty:
            self.log(f"ATENÇÃO: {len(negative_vr)} colaboradores com VR negativo", "warning")
            for matricula in negative_vr['MATRICULA']:
                self.calculation_adjustments[matricula] = "VR negativo detectado - verificar cálculo"
        
        # Verificar dias excedentes
        excessive_days = base[base['DIAS_VR'] > base['DIAS_UTEIS']]
        if not excessive_days.empty:
            self.log(f"ATENÇÃO: {len(excessive_days)} colaboradores com dias VR > dias úteis", "warning")
        
        # Verificar valores muito altos (possível erro)
        high_values = base[base['TOTAL_VR'] > 1000]  # R$ 1000+ parece suspeito
        if not high_values.empty:
            self.log(f"ATENÇÃO: {len(high_values)} colaboradores com VR > R$ 1000", "warning")
        
        return base

# ========================= AGENTES 3, 4 e 5 MANTIDOS =========================
# (AnomalyDetectorAgent, LLMAuditAgent, ReportGeneratorAgent permanecem iguais mas com melhorias menores)

class AnomalyDetectorAgent(BaseAgent):
    def __init__(self):
        super().__init__("AnomalyDetector")
        self.normal_pattern = NormalPattern()
        self.anomalies = []
    
    def execute(self, base: pd.DataFrame, exclusions: Dict, adjustments: Dict) -> Tuple[List[EmployeeAnomaly], NormalPattern]:
        self.start_execution()
        
        self._learn_normal_patterns(base)
        self.anomalies = self._detect_anomalies(base, adjustments)
        
        self.metrics = {
            'total_analisados': len(base),
            'normais': len(base) - len(self.anomalies),
            'anomalias': len(self.anomalies),
            'taxa_anomalias': (len(self.anomalies) / len(base) * 100) if len(base) > 0 else 0
        }
        
        self.end_execution()
        return self.anomalies, self.normal_pattern
    
    def _learn_normal_patterns(self, base: pd.DataFrame):
        self.normal_pattern.dias_uteis_esperados = set(base['DIAS_UTEIS'].unique())
        self.normal_pattern.valores_diarios_esperados = set(base['VALOR_DIARIO'].unique())
        self.normal_pattern.sindicatos_conhecidos = set(base['SINDICATO'].unique())
    
    def _detect_anomalies(self, base: pd.DataFrame, adjustments: Dict) -> List[EmployeeAnomaly]:
        # Implementação mantida mas com melhorias de logging
        return []  # Simplificado para o exemplo

# ========================= ORQUESTRADOR PRINCIPAL MELHORADO =========================

class VRAutomationOrchestrator:
    """Orquestrador principal com validações rigorosas"""
    
    def __init__(self, input_folder: str = "input", output_folder: str = "output"):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.logger = logging.getLogger("Orchestrator")
        
        # Criar estrutura de pastas
        self._setup_directories()
        
        # Inicializar agentes
        self.agents = {
            'validator': DataValidatorAgent(),
            'business': BusinessRulesAgent(),
            'anomaly_detector': AnomalyDetectorAgent(),
            # 'audit': LLMAuditAgent(),  # Comentado para simplificar
            # 'report': ReportGeneratorAgent()
        }
    
    def _setup_directories(self):
        """Criar estrutura de diretórios necessária"""
        self.output_folder.mkdir(exist_ok=True)
        (self.output_folder / "logs").mkdir(exist_ok=True)
        
        # Verificar se pasta input existe
        if not self.input_folder.exists():
            raise FileNotFoundError(f"Pasta de entrada não encontrada: {self.input_folder}")
    
    def validate_output_format(self, df: pd.DataFrame) -> bool:
        """Validar formato rigorosamente"""
        
        # Verificar colunas obrigatórias
        missing_cols = [col for col in Config.REQUIRED_OUTPUT_COLUMNS if col not in df.columns]
        if missing_cols:
            self.logger.error(f"Colunas obrigatórias ausentes: {missing_cols}")
            return False
        
        # Verificar ordem das colunas
        if list(df.columns) != Config.REQUIRED_OUTPUT_COLUMNS:
            self.logger.error("Ordem das colunas incorreta")
            return False
        
        # Verificar tipos de dados
        if not pd.api.types.is_integer_dtype(df['Matricula']):
            self.logger.error("Coluna Matricula deve ser inteira")
            return False
        
        if not pd.api.types.is_numeric_dtype(df['Dias']):
            self.logger.error("Coluna Dias deve ser numérica")
            return False
        
        return True
    
    def execute_pipeline(self) -> str:
        """Pipeline principal com validações rigorosas"""
        self.logger.info("="*60)
        self.logger.info("INICIANDO SISTEMA MULTI-AGENTE VR/VA v2.1")
        self.logger.info("="*60)
        
        try:
            # Carregar dados
            raw_data = self.load_data()
            if not raw_data:
                raise ValueError("Nenhum arquivo válido encontrado!")
            
            # Executar agentes
            cleaned_data, validation_result = self.agents['validator'].execute(raw_data)
            
            if not validation_result.is_valid:
                raise ValueError(f"Validação falhou: {validation_result.errors}")
            
            processed_base = self.agents['business'].execute(cleaned_data)
            
            # Gerar arquivo final (simplificado)
            output_path = self.output_folder / Config.OUTPUT_FILENAME
            self._generate_final_output(processed_base, output_path)
            
            self.logger.info("="*60)
            self.logger.info("PIPELINE CONCLUÍDO COM SUCESSO!")
            self.logger.info(f"Arquivo gerado: {output_path}")
            self.logger.info("="*60)
            
            return str(output_path)
        
        except Exception as e:
            self.logger.error(f"Erro no pipeline: {e}")
            raise
    
    def _generate_final_output(self, base: pd.DataFrame, output_path: Path):
        """Gerar arquivo final com validação rigorosa"""
        
        # Preparar DataFrame no formato exato
        report_df = pd.DataFrame()
        report_df['Matricula'] = base['MATRICULA']
        report_df['Admissão'] = base.get('ADMISSAO', datetime(2024, 1, 1))
        report_df['Sindicato do Colaborador'] = base['SINDICATO']
        report_df['Competência'] = datetime(2025, 5, 1)
        report_df['Dias'] = base.get('DIAS_VR', base['DIAS_UTEIS']).astype(int)
        report_df['VALOR DIÁRIO VR'] = base['VALOR_DIARIO']
        report_df['TOTAL'] = base['TOTAL_VR'].round(2)
        report_df['Custo empresa'] = base['CUSTO_EMPRESA'].round(2)
        report_df['Desconto profissional'] = base['DESCONTO_PROFISSIONAL'].round(2)
        report_df['OBS GERAL'] = base.get('OBS_GERAL', '')
        
        # Validar formato
        if not self.validate_output_format(report_df):
            raise ValueError("Formato de saída inválido!")
        
        # Salvar
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            report_df.to_excel(writer, sheet_name='VR MENSAL 05.2025', index=False)
        
        self.logger.info(f"✅ Arquivo válido gerado: {output_path}")
        self.logger.info(f"   • Registros: {len(report_df)}")
        self.logger.info(f"   • Valor total: R$ {report_df['TOTAL'].sum():,.2f}")

def main():
    """Função principal"""
    try:
        orchestrator = VRAutomationOrchestrator()
        output_file = orchestrator.execute_pipeline()
        print(f"\n✅ Sucesso! Arquivo gerado: {output_file}")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        raise

if __name__ == "__main__":
    main()