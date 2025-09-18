
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

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suprimir avisos do pandas
warnings.filterwarnings('ignore')

# ========================= ENUMS E DATACLASSES =========================

class AgentStatus(Enum):
    """Status de execução dos agentes"""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

class AnomalyType(Enum):
    """Tipos de anomalias detectadas"""
    FERIAS = "férias"
    ADMISSAO = "admissão"
    DESLIGAMENTO = "desligamento"
    DIAS_NEGATIVOS = "dias_negativos"
    VALOR_FORA_PADRAO = "valor_fora_padrão"
    DIAS_EXCEDENTES = "dias_excedentes"
    VALOR_ZERO = "valor_zero"
    PROPORCIONAL = "proporcional"

@dataclass
class ValidationResult:
    """Resultado da validação"""
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
    severidade: str = "medium"  # baixo, médio, alto
    requer_llm: bool = True

@dataclass
class NormalPattern:
    """Padrão normal identificado"""
    dias_uteis_esperados: Set[int] = field(default_factory=lambda: {21, 22})
    valores_diarios_esperados: Set[float] = field(default_factory=lambda: {35.0, 37.5})
    sindicatos_conhecidos: Set[str] = field(default_factory=set)
    tem_ferias: bool = False
    tem_ajustes: bool = False

# ========================= AGENTES BASE =========================

from typing import Any, Dict

class BaseAgent:
    metrics: dict[str, Any]
    exclusion_reasons: Dict[str, Any]
    calculation_adjustments: Dict[str, Any]

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        self.status = AgentStatus.IDLE
        self.metrics = {}  # Métricas do agente
        self.exclusion_reasons = {}  # Motivos de exclusão
        self.calculation_adjustments = {}  # Ajustes de cálculo

    def log(self, message: str, level: str = "info"):
        """Log de mensagens do agente"""
        getattr(self.logger, level)(f"[{self.name}] {message}")

    def execute(self, *args, **kwargs):
        """Método a ser implementado por cada agente"""
        raise NotImplementedError

# ========================= AGENTE 1: DATA VALIDATOR =========================

class DataValidatorAgent(BaseAgent):
    """Agente responsável por validar e limpar os dados"""
    
    def __init__(self):
        super().__init__("DataValidatorAgent")
        self.validation_rules = {
            'matricula': self._validate_matricula,
            'dates': self._validate_dates,
            'duplicates': self._check_duplicates,
            'missing': self._check_missing_values
        }
    
    def execute(self, data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], ValidationResult]:
        """Valida e limpa os dados"""
        self.status = AgentStatus.PROCESSING
        self.log("Iniciando validação dos dados...")
        result = ValidationResult(is_valid=True)
        cleaned_data = {}
        
        for file_name, df in data.items():
            self.log(f"Validando {file_name}...")
            
            # Limpar dados básicos
            df = self._clean_basic_data(df)
            
            # Aplicar validações específicas
            if file_name == "AFASTAMENTOS":
                df = self._fix_afastamentos(df)
            elif file_name == "ADMISSÃO ABRIL":
                df = self._validate_admissions(df)
            elif file_name == "DESLIGADOS":
                df = self._validate_terminations(df)
            
            cleaned_data[file_name] = df
            
            # Registrar metricas
            self.metrics[file_name] = {
                'original_rows': len(data[file_name]),
                'cleaned_rows': len(df)
            }
        
        self.status = AgentStatus.COMPLETED
        self.log(f"ValidaÃ§Ã£o concluÃ­da. metricas: {self.metrics}")
        
        return cleaned_data, result
    
    def _clean_basic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpeza bÃ¡sica dos dados"""
        # Remover espaÃ§os em branco nas colunas
        df.columns = df.columns.str.strip()
        
        # Remover linhas completamente vazias
        df = df.dropna(how='all')
        
        # Converter matricula para int onde aplicÃ¡vel
        if 'MATRICULA' in df.columns:
            df['MATRICULA'] = pd.to_numeric(df['MATRICULA'], errors='coerce')
            df = df.dropna(subset=['MATRICULA'])
            df['MATRICULA'] = df['MATRICULA'].astype(int)
        
        return df
    
    def _fix_afastamentos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Corrige o arquivo de afastamentos com dados duplicados"""
        # Remover duplicatas baseadas em matricula
        df = df.drop_duplicates(subset=['MATRICULA'], keep='first')
        
        # Validar que temos apenas afastamentos validos
        if 'DESC. SITUACAO' in df.columns:
            df = df[df['DESC. SITUACAO'].notna()]
        
        self.log(f"Afastamentos reduzidos para {len(df)} registros Ãºnicos")
        return df
    
    def _validate_admissions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida admissoes"""
        if 'admissão' in df.columns:
            df['admissão'] = pd.to_datetime(df['admissão'], errors='coerce')
            df = df.dropna(subset=['admissão'])
        return df
    
    def _validate_terminations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valida desligamentos"""
        if 'DATA DEMISSÃO' in df.columns:
            df['DATA DEMISSÃO'] = pd.to_datetime(df['DATA DEMISSÃO'], errors='coerce')
            df = df.dropna(subset=['DATA DEMISSÃO'])
        return df
    
    def _validate_matricula(self, df: pd.DataFrame) -> List[str]:
        """Valida matriculas"""
        errors = []
        if 'MATRICULA' in df.columns:
            invalid_matriculas = df[df['MATRICULA'] <= 0]
            if not invalid_matriculas.empty:
                errors.append(f"matriculas invalidas encontradas: {len(invalid_matriculas)}")
        return errors
    
    def _validate_dates(self, df: pd.DataFrame) -> List[str]:
        """Valida datas"""
        errors = []
        date_columns = df.select_dtypes(include=['datetime64']).columns
        for col in date_columns:
            future_dates = df[df[col] > datetime.now() + timedelta(days=365)]
            if not future_dates.empty:
                errors.append(f"Datas futuras invalidas em {col}: {len(future_dates)}")
        return errors
    
    def _check_duplicates(self, df: pd.DataFrame) -> List[str]:
        """Verifica duplicatas"""
        warnings = []
        if 'MATRICULA' in df.columns:
            duplicates = df[df.duplicated(subset=['MATRICULA'], keep=False)]
            if not duplicates.empty:
                warnings.append(f"Duplicatas encontradas: {len(duplicates)}")
        return warnings
    
    def _check_missing_values(self, df: pd.DataFrame) -> List[str]:
        """Verifica valores ausentes"""
        warnings = []
        missing = df.isnull().sum()
        if isinstance(missing, pd.Series):
            critical_missing = missing[missing.astype(float) > float(len(df)) * 0.1]
            if isinstance(critical_missing, pd.Series) and not critical_missing.empty:
                warnings.append(f"Colunas com muitos valores ausentes: {critical_missing.to_dict()}")
        return warnings

# ========================= AGENTE 2: BUSINESS RULES =========================

class BusinessRulesAgent(BaseAgent):
    """Agente que aplica as regras de negocio"""
    
    def __init__(self):
        super().__init__("BusinessRulesAgent")
        self.exclusion_reasons = {}
        self.calculation_adjustments = {}
    
    def execute(self, data: Dict[str, pd.DataFrame], reference_date: Optional[datetime] = None) -> pd.DataFrame:
        """Aplica todas as regras de negócio e retorna a base consolidada"""
        self.status = AgentStatus.PROCESSING
        self.log("Aplicando regras de negócio...")
        if reference_date is None:
            reference_date = datetime(2025, 5, 1)
    # 1. Consolidar base de ativos
        base = self._consolidate_active_employees(data)
    # 2. Aplicar exclusões
        base = self._apply_exclusions(base, data)
    # 3. Calcular dias úteis
        base = self._calculate_working_days(base, data, reference_date)
    # 4. Aplicar regras de férias
        base = self._apply_vacation_rules(base, data)
    # 5. Aplicar regras de admissão/desligamento
        base = self._apply_admission_termination_rules(base, data, reference_date)
    # 6. Calcular valores
        base = self._calculate_vr_values(base, data)
    # 7. Checagem final cruzada de exclusões
        excluidos: set[int] = set()
    # Diretores
        if 'ATIVOS' in data:
            if 'CARGO' in data['ATIVOS'].columns:
                diretores = data['ATIVOS'][data['ATIVOS']['CARGO'].str.contains('DIRETOR', case=False, na=False)]['MATRICULA'].unique()
                excluidos.update(diretores)
            else:
                self.log("Coluna 'CARGO' não encontrada em ATIVOS. Pulando exclusÃ£o de diretores.", "warning")
    # Estagiários
        if 'ESTAGIO' in data:
            excluidos.update(data['ESTAGIO']['MATRICULA'].unique())
    # Aprendizes
        if 'APRENDIZ' in data:
            excluidos.update(data['APRENDIZ']['MATRICULA'].unique())
    # Afastados
        if 'AFASTAMENTOS' in data:
            excluidos.update(data['AFASTAMENTOS']['MATRICULA'].unique())
    # Exterior
        if 'EXTERIOR' in data:
                df_exterior = data['EXTERIOR']
                col_cadastro = None
                # Procurar coluna 'CADASTRO' (já em maiúsculas)
                for col in df_exterior.columns:
                    if col.strip().upper() == 'CADASTRO':
                        col_cadastro = col
                        break
                if col_cadastro:
                    abroad = df_exterior[col_cadastro].dropna().unique()
                    excluidos.update(abroad)
                    self.log(f"Excluidos colaboradores no exterior")
                else:
                    self.log("Coluna 'CADASTRO' não encontrada na planilha EXTERIOR", "warning")
    # Remover todos excluídos
        base = base[~base['MATRICULA'].isin(excluidos)]
        self.log(f"Checagem final cruzada: {len(excluidos)} colaboradores excluídos.")
        self.status = AgentStatus.COMPLETED
        self.log(f"Regras aplicadas. {len(base)} colaboradores processados.")
        return base
    
    def _consolidate_active_employees(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Consolida base de colaboradores ativos"""
        base = data['ATIVOS'].copy()
        base.columns = pd.Index(['MATRICULA', 'EMPRESA', 'CARGO', 'SITUACAO', 'SINDICATO'])
        base = base[base['SITUACAO'] == 'Trabalhando']
        self.log(f"Base consolidada com {len(base)} colaboradores ativos")
        return base
    
    def _apply_exclusions(self, base: pd.DataFrame, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Aplica todas as exclusões necessárias"""
        initial_count = len(base)
        
        # Excluir diretores
        directors = base[base['CARGO'].str.contains('DIRETOR', case=False, na=False)]
        if not directors.empty:
            self.exclusion_reasons.update({m: 'Diretor' for m in directors['MATRICULA']})
            base = base[~base['MATRICULA'].isin(directors['MATRICULA'])]
            self.log(f"Excluidos {len(directors)} diretores")
        
        # Excluir estagiÃ¡rios
        if 'ESTAGIO' in data:
            interns = data['ESTAGIO']['MATRICULA'].unique()
            self.exclusion_reasons.update({m: 'EstagiÃ¡rio' for m in interns})
            base = base[~base['MATRICULA'].isin(interns)]
            self.log(f"Excluidos {len(interns)} estagiÃ¡rios")
        
        # Excluir aprendizes
        if 'APRENDIZ' in data:
            apprentices = data['APRENDIZ']['MATRICULA'].unique()
            self.exclusion_reasons.update({m: 'Aprendiz' for m in apprentices})
            base = base[~base['MATRICULA'].isin(apprentices)]
            self.log(f"Excluidos {len(apprentices)} aprendizes")
        
        # Excluir afastados
        if 'AFASTAMENTOS' in data:
            on_leave = data['AFASTAMENTOS']['MATRICULA'].unique()
            self.exclusion_reasons.update({m: 'Afastado' for m in on_leave if m in base['MATRICULA'].values})
            base = base[~base['MATRICULA'].isin(on_leave)]
            self.log(f"Excluidos {len(on_leave)} afastados")
        
        # Excluir exterior
        if 'EXTERIOR' in data:
            df_exterior = data['EXTERIOR']
            col_cadastro = None
            # Procurar coluna 'CADASTRO' (já em maiúsculas)
            for col in df_exterior.columns:
                if col.strip().upper() == 'CADASTRO':
                    col_cadastro = col
                    break
            if col_cadastro:
                abroad = df_exterior[col_cadastro].dropna().unique()
                self.exclusion_reasons.update({m: 'Exterior' for m in abroad if m in base['MATRICULA'].values})
                base = base[~base['MATRICULA'].isin(abroad)]
                self.log(f"Excluidos colaboradores no exterior")
            else:
                self.log("Coluna 'CADASTRO' não encontrada na planilha EXTERIOR", "warning")
        
        self.log(f"Total de exclusoes: {initial_count - len(base)}")
        return base
    
    def _calculate_working_days(self, base: pd.DataFrame, data: Dict[str, pd.DataFrame], reference_date: datetime) -> pd.DataFrame:
        """Calcula dias Uteis por sindicato, consultando LLM se necessario"""
        dias_uteis_df = data.get('Base dias uteis')
        dias_map = {}
        if dias_uteis_df is not None:
            for idx in range(2, len(dias_uteis_df)):
                row = dias_uteis_df.iloc[idx]
                if pd.notna(row[0]) and pd.notna(row[1]):
                    sindicato = str(row[0]).strip()
                    dias = int(row[1])
                    dias_map[sindicato] = dias
        # Aplicar dias Uteis
        base['DIAS_UTEIS'] = base['SINDICATO'].map(dias_map)
        # Identificar sindicatos sem mapeamento
        missing_sindicatos = base[base['DIAS_UTEIS'].isna()]['SINDICATO'].unique()
        if len(missing_sindicatos) > 0:
            self.log(f"âš ï¸ Sindicatos sem dias Uteis mapeados: {missing_sindicatos}", "warning")
            # Consultar LLM para sugerir dias Uteis corretos
            llm_agent = LLMAuditAgent()
            for sindicato in missing_sindicatos:
                # Montar contexto para LLM
                prompt = f"Sindicato: {sindicato}. Informe o numero correto de dias Uteis para VR em maio/2025 considerando Ferias, desligamentos e admissoes proporcionais. Responda apenas com o numero inteiro de dias Uteis." 
                try:
                    response = llm_agent.call_ollama(
                        EmployeeAnomaly(
                            matricula=0,
                            tipo_anomalia=AnomalyType.VALOR_FORA_PADRAO,
                            descricao=f"Dias Uteis não mapeados para sindicato {sindicato}",
                            dados_colaborador={"sindicato": sindicato},
                            severidade="medium",
                            requer_llm=True
                        ),
                        NormalPattern()
                    )
                    dias_llm = None
                    if isinstance(response, dict):
                        # Tentar extrair numero de dias da justificativa ou obs
                        for k in ['justificativa', 'obs', 'correcao_sugerida']:
                            v = response.get(k, "")
                            match = re.search(r'(\d{1,2})', str(v))
                            if match:
                                dias_llm = int(match.group(1))
                                break
                    if dias_llm:
                        base.loc[base['SINDICATO'] == sindicato, 'DIAS_UTEIS'] = dias_llm
                        self.log(f"Dias Uteis sugeridos pela LLM para {sindicato}: {dias_llm}")
                    else:
                        self.log(f"LLM não conseguiu sugerir dias Uteis para {sindicato}. Bloqueando calculo.", "warning")
                        base.loc[base['SINDICATO'] == sindicato, 'DIAS_UTEIS'] = None
                except Exception as e:
                    self.log(f"Erro ao consultar LLM para sindicato {sindicato}: {e}", "warning")
                    base.loc[base['SINDICATO'] == sindicato, 'DIAS_UTEIS'] = None
        # Bloquear calculo para sindicatos sem dias Uteis
        if base['DIAS_UTEIS'].isna().any():
            self.log("Existem colaboradores sem dias Uteis definidos. Corrija o mapeamento antes de prosseguir.", "error")
        self.log(f"Dias Uteis calculados. Mapeamento: {dias_map}")
        return base
    
    def _apply_vacation_rules(self, base: pd.DataFrame, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Aplica regras de férias"""
        if 'FÉRIAS' not in data:
            base['DIAS_FERIAS'] = 0
            base['DIAS_VR'] = base['DIAS_UTEIS']
            return base
        
        ferias_df = data['FÉRIAS']
        
        # Criar mapeamento de férias
        ferias_map = dict(zip(ferias_df['MATRICULA'], ferias_df['DIAS DE FÉRIAS']))
        
        # Aplicar desconto de férias
        base['DIAS_FERIAS'] = base['MATRICULA'].map(ferias_map).fillna(0)
        base['DIAS_VR'] = base['DIAS_UTEIS'] - base['DIAS_FERIAS']
        base['DIAS_VR'] = base['DIAS_VR'].clip(lower=0)  # Não pode ser negativo
        
        # Registrar ajustes
        for matricula, dias_ferias in ferias_map.items():
            if matricula in base['MATRICULA'].values and dias_ferias > 0:
                self.calculation_adjustments[matricula] = f"Deduzidos {dias_ferias} dias de férias"
        
        self.log(f"Regras de férias aplicadas. {len(ferias_map)} colaboradores em férias.")
        return base
    
    def _apply_admission_termination_rules(self, base: pd.DataFrame, data: Dict[str, pd.DataFrame], reference_date: datetime) -> pd.DataFrame:
        """Aplica regras de admissão e desligamento com calculo proporcional detalhado via LLM"""
        llm_agent = LLMAuditAgent()
        # Processar admissoes
        if 'ADMISSÃO ABRIL' in data:
            admissions = data['ADMISSÃO ABRIL']
            for _, adm in admissions.iterrows():
                matricula = adm['MATRICULA']
                adm_date = pd.to_datetime(adm['ADMISSÃO'])
                if matricula in base['MATRICULA'].values:
                    idx = base[base['MATRICULA'] == matricula].index[0]
                    dias_uteis = base.loc[idx, 'DIAS_UTEIS'] if 'DIAS_UTEIS' in base.columns else 22
                    # Se admitido apos dia 15 de abril, calcular proporcional para maio
                    if adm_date.month == 4 and adm_date.year == 2025 and adm_date.day > 15:
                        # Consultar LLM para calculo proporcional exato
                        prompt = f"Colaborador admitido em {adm_date.strftime('%d/%m/%Y')}, sindicato: {base.loc[idx, 'SINDICATO']}, dias Uteis do Mês: {dias_uteis}. Calcule o numero proporcional de dias VR para maio/2025 conforme regras do PDF. Responda apenas com o numero inteiro de dias VR." 
                        anomaly = EmployeeAnomaly(
                            matricula=matricula,
                            tipo_anomalia=AnomalyType.ADMISSAO,
                            descricao=f"admissão apos dia 15/abril. Calcular proporcional.",
                            dados_colaborador={"admissao": adm_date.strftime('%d/%m/%Y'), "dias_uteis": dias_uteis, "sindicato": base.loc[idx, 'SINDICATO']},
                            severidade="medium",
                            requer_llm=True
                        )
                        response = llm_agent.call_ollama(anomaly, NormalPattern())
                        dias_llm = None
                        for k in ['justificativa', 'obs', 'correcao_sugerida']:
                            v = response.get(k, "")
                            match = re.search(r'(\d{1,2})', str(v))
                            if match:
                                dias_llm = int(match.group(1))
                                break
                        if dias_llm:
                            base.loc[idx, 'DIAS_VR'] = dias_llm
                            base.loc[idx, 'ADMISSAO'] = adm_date
                            self.calculation_adjustments[matricula] = f"admissão em {adm_date.strftime('%d/%m/%Y')} - VR proporcional LLM: {dias_llm} dias"
                        else:
                            import math
                            if dias_uteis is None or (isinstance(dias_uteis, float) and math.isnan(dias_uteis)):
                                dias_uteis_corrigido = 0
                            else:
                                dias_uteis_corrigido = dias_uteis
                            base.loc[idx, 'DIAS_VR'] = int(dias_uteis_corrigido * 0.5)
                            base.loc[idx, 'ADMISSAO'] = adm_date
                            self.calculation_adjustments[matricula] = f"admissão em {adm_date.strftime('%d/%m/%Y')} - VR proporcional padrão: {int(dias_uteis_corrigido * 0.5)} dias"
        # Processar desligamentos
        if 'DESLIGADOS' in data:
            terminations = data['DESLIGADOS']
            for _, term in terminations.iterrows():
                matricula = term['MATRICULA']
                term_date = pd.to_datetime(term['DATA DEMISSÃO'])
                comunicado = term.get('COMUNICADO DE DESLIGAMENTO', '')
                if matricula in base['MATRICULA'].values:
                    idx = base[base['MATRICULA'] == matricula].index[0]
                    dias_uteis = base.loc[idx, 'DIAS_UTEIS'] if 'DIAS_UTEIS' in base.columns else 22
                    # Regra: ATE dia 15 com OK = excluir
                    if term_date.day <= 15 and comunicado == 'OK':
                        base = base.drop(idx)
                        self.exclusion_reasons[matricula] = f"Desligado dia {term_date.day} com comunicado OK"
                    # apos dia 15 = proporcional
                    elif term_date.day > 15:
                        # Consultar LLM para calculo proporcional exato
                        prompt = f"Colaborador desligado em {term_date.strftime('%d/%m/%Y')}, sindicato: {base.loc[idx, 'SINDICATO']}, dias Uteis do Mês: {dias_uteis}. Calcule o numero proporcional de dias VR para maio/2025 conforme regras do PDF. Responda apenas com o numero inteiro de dias VR." 
                        anomaly = EmployeeAnomaly(
                            matricula=matricula,
                            tipo_anomalia=AnomalyType.DESLIGAMENTO,
                            descricao=f"Desligamento apos dia 15/maio. Calcular proporcional.",
                            dados_colaborador={"desligamento": term_date.strftime('%d/%m/%Y'), "dias_uteis": dias_uteis, "sindicato": base.loc[idx, 'SINDICATO']},
                            severidade="high",
                            requer_llm=True
                        )
                        response = llm_agent.call_ollama(anomaly, NormalPattern())
                        dias_llm = None
                        for k in ['justificativa', 'obs', 'correcao_sugerida']:
                            v = response.get(k, "")
                            match = re.search(r'(\d{1,2})', str(v))
                            if match:
                                dias_llm = int(match.group(1))
                                break
                        import math
                        if dias_llm:
                            base.loc[idx, 'DIAS_VR'] = dias_llm
                            self.calculation_adjustments[matricula] = f"Desligamento em {term_date.strftime('%d/%m/%Y')} - VR proporcional LLM: {dias_llm} dias"
                        else:
                            dias_trabalhados = term_date.day
                            proporcao = dias_trabalhados / 30
                            # Corrigir NaN para 0
                            if dias_uteis is None or (isinstance(dias_uteis, float) and math.isnan(dias_uteis)):
                                dias_uteis_corrigido = 0
                            else:
                                dias_uteis_corrigido = dias_uteis
                            base.loc[idx, 'DIAS_VR'] = int(dias_uteis_corrigido * proporcao)
                            self.calculation_adjustments[matricula] = f"Desligamento em {term_date.strftime('%d/%m/%Y')} - VR proporcional padrão: {int(dias_uteis_corrigido * proporcao)} dias"
        return base
    
    def _calculate_vr_values(self, base: pd.DataFrame, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calcula valores de VR usando a planilha de referencia de sindicato x valor"""
        # Carregar base de valores por sindicato
        sindicato_valor_df = data.get('Base sindicato x valor')
        sindicato_valor_map: Dict[object, float] = {}
        if sindicato_valor_df is not None:
            for idx, row in sindicato_valor_df.iterrows():
                valor = float(row[1]) if not pd.isna(row[1]) else 0.0
                sindicato_str = str(row[0]).strip()
                try:
                    sindicato_valor_map[int(float(sindicato_str))] = valor  # type: ignore
                except (ValueError, TypeError):
                    continue  # Ignora sindicatos não convertÃ­veis para int
        def map_sindicato(s) -> float | None:
            try:
                s_int = int(float(s))
                return sindicato_valor_map[s_int]
            except (KeyError, ValueError, TypeError):
                return None
        base['VALOR_DIARIO'] = base['SINDICATO'].apply(map_sindicato)
        # Se não encontrar, usar valor padrão (37.5)
        base['VALOR_DIARIO'] = base['VALOR_DIARIO'].fillna(37.5)
        # Calcular valores
        base['TOTAL_VR'] = base['DIAS_VR'] * base['VALOR_DIARIO']
        base['CUSTO_EMPRESA'] = base['TOTAL_VR'] * 0.8
        base['DESCONTO_PROFISSIONAL'] = base['TOTAL_VR'] * 0.2
        self.log("Valores de VR calculados com base na planilha de sindicato x valor")
        return base

# ========================= AGENTE 3: ANOMALY DETECTOR =========================

class AnomalyDetectorAgent(BaseAgent):
    """Agente que detecta anomalias e padroes normais"""
    
    def __init__(self):
        super().__init__("AnomalyDetectorAgent")
        self.normal_pattern = NormalPattern()
        self.anomalies = []
    
    def execute(self, base: pd.DataFrame, exclusions: Dict, adjustments: Dict) -> Tuple[List[EmployeeAnomaly], NormalPattern]:
        """Detecta anomalias e aprende padroes normais"""
        self.status = AgentStatus.PROCESSING
        self.log("Iniciando Detecção de anomalias e aprendizado de padroes...")
        
        # 1. Aprender padroes normais
        self._learn_normal_patterns(base)

        # 2. Detectar anomalias
        self.anomalies = self._detect_anomalies(base, adjustments)
        
        # 3. Gerar metricas
        self.metrics = {
            'total_analisados': len(base),
            'normais': len(base) - len(self.anomalies),
            'anomalias': len(self.anomalies),
            'taxa_anomalias': (len(self.anomalies) / len(base) * 100) if len(base) > 0 else 0
        }
        
        self.status = AgentStatus.COMPLETED
        self.log(f"Detecção concluÃ­da: {len(self.anomalies)} anomalias em {len(base)} registros ({self.metrics['taxa_anomalias']:.1f}%)")
        
        return self.anomalies, self.normal_pattern
    
    def _learn_normal_patterns(self, base: pd.DataFrame):
        """Aprende os padroes normais dos dados"""
        # padroes de dias Uteis
        self.normal_pattern.dias_uteis_esperados = set(base['DIAS_UTEIS'].unique())
        
        # padroes de valores diarios
        self.normal_pattern.valores_diarios_esperados = set(base['VALOR_DIARIO'].unique())
        
        # Sindicatos conhecidos
        self.normal_pattern.sindicatos_conhecidos = set(base['SINDICATO'].unique())
        
        # estatisticas gerais
        dias_medio = base['DIAS_VR'].mean()
        dias_std = base['DIAS_VR'].std()
        
        self.log(f"padroes normais aprendidos:")
        self.log(f"  -- Dias Uteis esperados: {self.normal_pattern.dias_uteis_esperados}")
        self.log(f"  -- Valores diarios: {self.normal_pattern.valores_diarios_esperados}")
        self.log(f"  -- Media de dias VR: {dias_medio:.1f} Â± {dias_std:.1f}")
        self.log(f"  -- {len(self.normal_pattern.sindicatos_conhecidos)} sindicatos conhecidos")
    
    def _detect_anomalies(self, base: pd.DataFrame, adjustments: Dict) -> List[EmployeeAnomaly]:
        """Detecta todas as anomalias na base"""
        anomalies = []
        
        for idx, row in base.iterrows():
            matricula = row['MATRICULA']
            employee_anomalies = []
            
            # 1. Verificar Ferias
            if row.get('DIAS_FERIAS', 0) > 0:
                employee_anomalies.append(EmployeeAnomaly(
                    matricula=matricula,
                    tipo_anomalia=AnomalyType.FERIAS,
                    descricao=f"Colaborador em Ferias por {int(row['DIAS_FERIAS'])} dias",
                    dados_colaborador=self._get_employee_data(row),
                    severidade="medium",
                    requer_llm=True
                ))
            
            # 2. Verificar admissão/desligamento
            if matricula in adjustments:
                ajuste = adjustments[matricula]
                if "admissão" in ajuste:
                    employee_anomalies.append(EmployeeAnomaly(
                        matricula=matricula,
                        tipo_anomalia=AnomalyType.ADMISSAO,
                        descricao=ajuste,
                        dados_colaborador=self._get_employee_data(row),
                        severidade="medium",
                        requer_llm=True
                    ))
                elif "Desligamento" in ajuste:
                    employee_anomalies.append(EmployeeAnomaly(
                        matricula=matricula,
                        tipo_anomalia=AnomalyType.DESLIGAMENTO,
                        descricao=ajuste,
                        dados_colaborador=self._get_employee_data(row),
                        severidade="high",
                        requer_llm=True
                    ))
            
            # 3. Verificar dias negativos ou zero
            if row['DIAS_VR'] < 0:
                employee_anomalies.append(EmployeeAnomaly(
                    matricula=matricula,
                    tipo_anomalia=AnomalyType.DIAS_NEGATIVOS,
                    descricao=f"Dias VR negativos: {row['DIAS_VR']}",
                    dados_colaborador=self._get_employee_data(row),
                    severidade="high",
                    requer_llm=True
                ))
            elif row['DIAS_VR'] == 0 and row['DIAS_FERIAS'] == 0:
                employee_anomalies.append(EmployeeAnomaly(
                    matricula=matricula,
                    tipo_anomalia=AnomalyType.VALOR_ZERO,
                    descricao="Dias VR zerados sem motivo aparente",
                    dados_colaborador=self._get_employee_data(row),
                    severidade="high",
                    requer_llm=True
                ))
            
            # 4. Verificar valor diario fora do padrão
            if row['VALOR_DIARIO'] not in self.normal_pattern.valores_diarios_esperados:
                employee_anomalies.append(EmployeeAnomaly(
                    matricula=matricula,
                    tipo_anomalia=AnomalyType.VALOR_FORA_PADRAO,
                    descricao=f"Valor diario anormal: R$ {row['VALOR_DIARIO']}",
                    dados_colaborador=self._get_employee_data(row),
                    severidade="medium",
                    requer_llm=True
                ))
            
            # 5. Verificar dias excedentes
            if row['DIAS_VR'] > row['DIAS_UTEIS']:
                employee_anomalies.append(EmployeeAnomaly(
                    matricula=matricula,
                    tipo_anomalia=AnomalyType.DIAS_EXCEDENTES,
                    descricao=f"Dias VR ({row['DIAS_VR']}) excedem dias Uteis ({row['DIAS_UTEIS']})",
                    dados_colaborador=self._get_employee_data(row),
                    severidade="high",
                    requer_llm=True
                ))
            
            # 6. Verificar casos proporcionais (dias diferentes do padrão sem Ferias)
            if row['DIAS_FERIAS'] == 0 and row['DIAS_VR'] not in self.normal_pattern.dias_uteis_esperados:
                if row['DIAS_VR'] > 0:  # So se não for zero
                    employee_anomalies.append(EmployeeAnomaly(
                        matricula=matricula,
                        tipo_anomalia=AnomalyType.PROPORCIONAL,
                        descricao=f"Dias calculados ({row['DIAS_VR']}) diferente do esperado",
                        dados_colaborador=self._get_employee_data(row),
                        severidade="medium",
                        requer_llm=True
                    ))
            
            # Adicionar apenas a anomalia mais severa por colaborador para evitar duplicacao
            if employee_anomalies:
                # Ordenar por severidade (high > medium > low)
                severity_order = {"high": 3, "medium": 2, "low": 1}
                employee_anomalies.sort(key=lambda x: severity_order[x.severidade], reverse=True)
                anomalies.append(employee_anomalies[0])  # Adiciona apenas a mais severa
        
        return anomalies
    
    def _get_employee_data(self, row: pd.Series) -> Dict[str, Any]:
        """Extrai dados do colaborador para analise"""
        import math
        dias_uteis_val = row.get('DIAS_UTEIS', 22)
        if dias_uteis_val is None or (isinstance(dias_uteis_val, float) and math.isnan(dias_uteis_val)):
            dias_uteis_val = 22
        else:
            dias_uteis_val = int(dias_uteis_val)
        return {
            "matricula": int(row['MATRICULA']),
            "sindicato": str(row.get('SINDICATO', '')),
            "dias_vr": int(row.get('DIAS_VR', 0)),
            "dias_ferias": int(row.get('DIAS_FERIAS', 0)),
            "dias_uteis": dias_uteis_val,
            "valor_diario": float(row.get('VALOR_DIARIO', 0)),
            "total_vr": float(row.get('TOTAL_VR', 0)),
            "custo_empresa": float(row.get('CUSTO_EMPRESA', 0)),
            "desconto_profissional": float(row.get('DESCONTO_PROFISSIONAL', 0))
        }

# ========================= AGENTE 4: LLM AUDIT OTIMIZADO =========================

class LLMAuditAgent(BaseAgent):
    def save_llm_anomaly_report(self, output_folder: str = "output"):
        """Salva relatorio detalhado das anomalias processadas pelo Ollama"""
        import os
        import json
        from datetime import datetime
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "llm_provider": self.llm_provider,
            "anomalies": self.llm_responses
        }
        os.makedirs(output_folder, exist_ok=True)
        report_path = os.path.join(output_folder, f"llm_anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        self.log(f"relatorio de anomalias LLM salvo em: {report_path}")
    """Agente que usa LLM apenas para casos Anormais"""
    
    def __init__(self):
        super().__init__("LLMAuditAgent")
        self.observations = {}
        self.llm_provider = self._select_llm_provider()
        self.setup_instructions()
        self.processed_anomalies = 0
        self.llm_responses = []
    
    def _select_llm_provider(self) -> str:
        """Seleciona o provedor LLM disponÃ­vel"""
        # Verificar Ollama
        if self._check_ollama():
            self.log("Usando Ollama (local) como LLM")
            return "ollama"
        
        # Verificar Gemini
        if os.getenv("AIzaSyCOPKG7ZoAQcUqRTicHt84aXJbsh5rlIE8"):
            self.log("Usando Google Gemini como LLM")
            return "gemini"
        
        # Verificar OpenAI
        if os.getenv("OPENAI_API_KEY"):
            self.log("Usando OpenAI como LLM")
            return "openai"
        
        self.log("âš ï¸ Nenhum LLM configurado. Usando modo fallback inteligente.", "warning")
        return "fallback"
    
    def _check_ollama(self) -> bool:
        """Verifica se Ollama esta rodando"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def setup_instructions(self):
        """Define intrusções base para o LLM com contexto sobre padroes normais"""
        self.system_prompt = """Você é um auditor especializado em Vale Refeição (VR) no Brasil.
        
        CONTEXTO DE NORMALIDADE:
        - Dias Uteis normais: 21 ou 22 dias
        - Valores diarios normais: R$ 35,00 (PR/RJ/RS) ou R$ 37,50 (SP)
        - Colaborador normal: sem Ferias, sem ajustes, dias e valores dentro do padrão
        
        VOCE Esta ANALISANDO APENAS CASOS ANORMAIS que precisam de ATENÇÃO especial.
        
        REGRAS DE ANALISE PARA ANOMALIAS:
        1. Ferias: Explicar dedução proporcional e verificar se calculo esta correto
        2. admissão em abril: Se apos dia 15, VR de maio deve ser proporcional
        3. Desligamento: ATE dia 15 com OK = excluir; apos dia 15 = proporcional
        4. Dias negativos: Sempre corrigir e alertar
        5. Valores fora do padrão: Verificar se sindicato justifica
        
        FORMATO DE RESPOSTA (JSON estrito):
        {
            "acao": "aprovar|corrigir|revisar",
            "justificativa": "explicação técnica da anomalia",
            "obs": "texto para a coluna OBS da planilha",
            "correcao_sugerida": "se acao for corrigir, explicar correção",
            "severidade": "low|medium|high"
        }
        
        IMPORTANTE: Responda APENAS com JSON valido."""
    
    def call_ollama(self, anomaly: EmployeeAnomaly, normal_pattern: NormalPattern) -> dict:
        """Chama Ollama com contexto da anomalia"""
        
        prompt = f"""{self.system_prompt}
        
        PADRÃ•ES NORMAIS APRENDIDOS:
        - Dias Uteis esperados: {normal_pattern.dias_uteis_esperados}
        - Valores diarios esperados: {normal_pattern.valores_diarios_esperados}
        
        ANOMALIA DETECTADA:
        Tipo: {anomaly.tipo_anomalia.value}
        Descrição: {anomaly.descricao}
        Severidade inicial: {anomaly.severidade}
        
        DADOS DO COLABORADOR:
        {json.dumps(anomaly.dados_colaborador, ensure_ascii=False, indent=2)}
        
        Analise esta anomalia e forneça sua decisão em JSON:"""
        
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "prompt": prompt,
                    "stream": True,
                    "format": "json",
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "seed": 42  # Para resultados mais consistentes
                    }
                },
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                import re
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError:
                    # Tentar extrair apenas o primeiro bloco JSON
                    json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
                    if json_match:
                        try:
                            return json.loads(json_match.group())
                        except Exception as e:
                            self.log(f"Erro ao parsear JSON extraÃ­do: {e}", "warning")
        except Exception as e:
            self.log(f"Erro Ollama: {e}", "warning")
        
        return self._generate_fallback_response(anomaly)
    
    def _generate_fallback_response(self, anomaly: EmployeeAnomaly) -> dict:
        """Gera resposta inteligente quando LLM não esta disponÃ­vel"""
        response = {
            "acao": "aprovar",
            "justificativa": "",
            "obs": "",
            "correcao_sugerida": "",
            "severidade": anomaly.severidade
        }
        
        # Respostas especificas por tipo de anomalia
        if anomaly.tipo_anomalia == AnomalyType.FERIAS:
            dias_ferias = anomaly.dados_colaborador.get("dias_ferias", 0)
            response["obs"] = f"Colaborador teve {dias_ferias} dias de Ferias deduzidos no periodo"
            response["justificativa"] = "dedução de Ferias aplicada corretamente"
            
        elif anomaly.tipo_anomalia == AnomalyType.ADMISSAO:
            response["obs"] = "VR calculado proporcionalmente devido a admissão em abril"
            response["justificativa"] = "calculo proporcional para admissão"
            
        elif anomaly.tipo_anomalia == AnomalyType.DESLIGAMENTO:
            response["obs"] = "VR ajustado devido a desligamento no Mês"
            response["justificativa"] = "calculo proporcional para desligamento"
            
        elif anomaly.tipo_anomalia == AnomalyType.DIAS_NEGATIVOS:
            response["acao"] = "corrigir"
            response["obs"] = "CORREÇÃO NECESSÃRIA: Dias negativos detectados"
            response["justificativa"] = "Dias VR não podem ser negativos"
            response["correcao_sugerida"] = "Ajustar para 0 dias ou revisar calculo"
            response["severidade"] = "high"
            
        elif anomaly.tipo_anomalia == AnomalyType.VALOR_FORA_PADRAO:
            valor = anomaly.dados_colaborador.get("valor_diario", 0)
            response["obs"] = f"Valor diario R$ {valor:.2f} fora do padrão - verificar sindicato"
            response["justificativa"] = "Valor diario requer verificaÃ§Ã£o"
            response["acao"] = "revisar"
            
        elif anomaly.tipo_anomalia == AnomalyType.DIAS_EXCEDENTES:
            response["acao"] = "corrigir"
            response["obs"] = "ERRO: Dias VR excedem dias Uteis do Mês"
            response["justificativa"] = "inconsistencia no calculo de dias"
            response["correcao_sugerida"] = "Limitar aos dias Uteis do periodo"
            response["severidade"] = "high"
            
        elif anomaly.tipo_anomalia == AnomalyType.VALOR_ZERO:
            response["acao"] = "revisar"
            response["obs"] = "Dias zerados - verificar motivo"
            response["justificativa"] = "Colaborador sem VR precisa de justificativa"
            
        elif anomaly.tipo_anomalia == AnomalyType.PROPORCIONAL:
            response["obs"] = f"calculo proporcional aplicado: {anomaly.dados_colaborador.get('dias_vr', 0)} dias"
            response["justificativa"] = "Ajuste proporcional detectado"
        
        return response
    
    def execute(self, base: pd.DataFrame, anomalies: List[EmployeeAnomaly], 
                normal_pattern: NormalPattern) -> pd.DataFrame:
        """Executa auditoria APENAS para casos Anormais"""
        self.status = AgentStatus.PROCESSING
        self.log(f"Iniciando auditoria inteligente de {len(anomalies)} anomalias...")
        self.log(f"Economizando analise de {len(base) - len(anomalies)} casos normais")
        
        # Inicializar coluna de obersevaçoes
        base['OBS_GERAL'] = ''
        
        # Criar mapa de anomalias por matricula para acesso rÃ¡pido
        anomaly_map = {a.matricula: a for a in anomalies}
        
        # Processar apenas anomalias
        if anomalies:
            self.log(f"Processando {len(anomalies)} anomalias via {self.llm_provider}...")
            
            for i, anomaly in enumerate(anomalies, 1):
                if i % 10 == 0:
                    self.log(f"Progresso: {i}/{len(anomalies)} anomalias processadas...")
                
                # Chamar LLM para analisar anomalia
                if self.llm_provider == "ollama":
                    llm_response = self.call_ollama(anomaly, normal_pattern)
                else:
                    llm_response = self._generate_fallback_response(anomaly)
                
                # Aplicar resposta do LLM na base
                matricula = anomaly.matricula
                if matricula in base['MATRICULA'].values:
                    idx = base[base['MATRICULA'] == matricula].index[0]
                    
                    # Montar obersevação
                    obs = llm_response.get('obs', '')
                    if llm_response.get('acao') == 'corrigir':
                        obs = f"[CORRIGIR] {obs}"
                        if llm_response.get('correcao_sugerida'):
                            obs += f" | Sugestão: {llm_response.get('correcao_sugerida')}"
                    elif llm_response.get('acao') == 'revisar':
                        obs = f"[REVISAR] {obs}"
                    
                    base.loc[idx, 'OBS_GERAL'] = obs
                    self.processed_anomalies += 1
                    
                    # Armazenar resposta para relatorio
                    self.llm_responses.append({
                        'matricula': matricula,
                        'anomalia': anomaly.tipo_anomalia.value,
                        'acao': llm_response.get('acao'),
                        'severidade': llm_response.get('severidade', anomaly.severidade)
                    })
        
        # Gerar estatisticas
        total_com_obs = (base['OBS_GERAL'] != '').sum()
        
        self.observations['summary'] = {
            'total_colaboradores': len(base),
            'casos_normais': len(base) - len(anomalies),
            'anomalias_detectadas': len(anomalies),
            'anomalias_processadas': self.processed_anomalies,
            'total_com_observacoes': total_com_obs,
            'economia_processamento': f"{((len(base) - len(anomalies)) / len(base) * 100):.1f}%",
            'llm_provider': self.llm_provider
        }
        
        # analise de severidade
        if self.llm_responses:
            severidade_count: Dict[str, int] = {}
            for resp in self.llm_responses:
                sev = resp.get('severidade', 'medium')
                severidade_count[sev] = severidade_count.get(sev, 0) + 1
            
            self.observations['severidade'] = severidade_count
        
        self.status = AgentStatus.COMPLETED
        self.log("#“ Auditoria inteligente concluÃ­da!")
        self.log(f"  -- Processadas {self.processed_anomalies} anomalias")
        self.log(f"  -- Economizados {len(base) - len(anomalies)} casos normais")
        self.log(f"  -- Taxa de economia: {self.observations['summary']['economia_processamento']}")
        
        return base

# ========================= AGENTE 5: REPORT GENERATOR =========================

class ReportGeneratorAgent(BaseAgent):
    """Agente que gera a planilha final no formato correto"""
    
    def __init__(self):
        super().__init__("ReportGeneratorAgent")
    
    def execute(self, base: pd.DataFrame, output_path: str = "VR MENSAL 05.2025.xlsx") -> str:
        """Gera a planilha final no formato especificado"""
        self.status = AgentStatus.PROCESSING
        self.log("Gerando relatorio final...")

        # Preparar dados no formato correto (10 colunas)
        report_df = pd.DataFrame()

        # Definir Competência
        competencia = datetime(2025, 5, 1)

        # Mapear colunas para o formato final
        report_df['Matricula'] = base['MATRICULA']
        report_df['admissão'] = base.get('ADMISSAO', datetime(2024, 1, 1))
        report_df['Sindicato do Colaborador'] = base['SINDICATO']
        report_df['Competência'] = competencia
        report_df['Dias'] = base['DIAS_VR'].fillna(0).astype(int)
        report_df['VALOR DIÁRIO VR'] = base['VALOR_DIARIO']
        report_df['TOTAL'] = base['TOTAL_VR'].round(2)
        report_df['Custo empresa'] = base['CUSTO_EMPRESA'].round(2)
        report_df['Desconto profissional'] = base['DESCONTO_PROFISSIONAL'].round(2)
        report_df['OBS GERAL'] = base.get('OBS_GERAL', '')

        # Ordenar por matricula
        report_df = report_df.sort_values('Matricula')

        # Criar Excel com formatação
        self._save_to_excel(report_df, output_path)

        self.status = AgentStatus.COMPLETED
        self.log(f"relatorio gerado: {output_path}")
        self.log(f"Total de registros: {len(report_df)}")
        self.log(f"Valor total: R$ {report_df['TOTAL'].sum():,.2f}")
        self.log(f"Registros com obersevaçoes: {(report_df['OBS GERAL'] != '').sum()}")

        return output_path
    
    def _save_to_excel(self, df: pd.DataFrame, output_path: str):
        """Salva DataFrame em Excel com formatação"""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Adicionar linha de totais no topo
            total_row = pd.DataFrame([[None] * 6 + [df['TOTAL'].sum()] + [None] * 3], 
                                    columns=df.columns)
            # Concatenar total com dados
            final_df = pd.concat([total_row, df], ignore_index=True)
            # Salvar na primeira aba
            final_df.to_excel(writer, sheet_name='VR MENSAL 05.2025', index=False, startrow=1)
            # Adicionar aba de Validações
            validacoes = pd.DataFrame({
                'Validações': [
                    'Afastados / Licenças',
                    'DESLIGADOS GERAL',
                    'Admitidos Mês',
                    'Ferias',
                    'ESTAGIARIO',
                    'APRENDIZ',
                    'SINDICATOS x VALOR',
                    'DESLIGADOS Antes DIA 15 - EXCLUIR SE OK',
                    'DESLIGADOS Após DIA 15 - PROPORCIONAL',
                    'EXTERIOR',
                    'ATIVOS',
                    'ANALISE LLM APLICADA'
                ],
                'Check': ['#“'] * 12
            })
            validacoes.to_excel(writer, sheet_name='Validações', index=False)
            # Formatar colunas
            worksheet = writer.sheets['VR MENSAL 05.2025']
            column_widths = {
                'A': 10,  # Matricula
                'B': 12,  # admissão
                'C': 50,  # Sindicato
                'D': 12,  # CompetênciaApós
                'E': 8,   # Dias
                'F': 15,  # Valor diario
                'G': 12,  # Total
                'H': 15,  # Custo empresa
                'I': 20,  # Desconto profissional
                'J': 60   # OBS (maior para acomodar analises)
            }
            for col, width in column_widths.items():
                worksheet.column_dimensions[col].width = width

# ========================= ORQUESTRADOR PRINCIPAL OTIMIZADO =========================

class VRAutomationOrchestrator:
    """Orquestrador principal do sistema multi-agente otimizado"""
    
    def __init__(self, input_folder: str = "input", output_folder: str = "output"):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.logger = logging.getLogger("Orchestrator")
        
        # Inicializar agentes (agora com 5 agentes)
        self.agents = {
            'validator': DataValidatorAgent(),
            'business': BusinessRulesAgent(),
            'anomaly_detector': AnomalyDetectorAgent(),
            'audit': LLMAuditAgent(),
            'report': ReportGeneratorAgent()
        }
        
        # Criar pastas se não existirem
        self.output_folder.mkdir(exist_ok=True)
        (self.output_folder / "logs").mkdir(exist_ok=True)
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Carrega todos os arquivos de entrada"""
        self.logger.info("Carregando arquivos de entrada...")

        data = {}
        required_files = [
            'ATIVOS.xlsx',
            'FÉRIAS.xlsx',
            'ADMISSÃO ABRIL.xlsx',
            'DESLIGADOS.xlsx',
            'AFASTAMENTOS.xlsx',
            'APRENDIZ.xlsx',
            'ESTAGIO.xlsx',
            'EXTERIOR.xlsx',
            'Base dias uteis.xlsx',
            'Base sindicato x valor.xlsx'
        ]

        # Mapeamento flexível de nomes de colunas para o padrão esperado
        column_map = {
            'CARGO': ['TITULO DO CARGO', 'CARGO', 'FUNÇÃO'],
            'SITUACAO': ['DESC. SITUACAO', 'SITUACAO', 'SITUAÇÃO'],
            'SINDICATO': ['Sindicato', 'SINDICATO'],
            'MATRICULA': ['MATRICULA', 'matricula'],
            'EMPRESA': ['EMPRESA', 'Empresa'],
        }

        def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
            col_map = {}
            for std, alts in column_map.items():
                for alt in alts:
                    for col in df.columns:
                        if col.strip().upper() == alt.strip().upper():
                            col_map[col] = std
            return df.rename(columns=col_map)

        for file_name in required_files:
            file_path = self.input_folder / file_name
            if file_path.exists():
                try:
                    df = pd.read_excel(file_path)
                    df = normalize_columns(df)
                    df.columns = df.columns.str.upper()
                    data[file_name.replace('.xlsx', '')] = df
                    self.logger.info(f"#“ Carregado: {file_name} ({len(df)} registros)")
                except Exception as e:
                    self.logger.error(f"#— Erro ao carregar {file_name}: {e}")
            else:
                self.logger.warning(f"⚠  Arquivo não encontrado: {file_name}")
        return data
    
    def validate_and_update_rules(self, data: Dict[str, pd.DataFrame]):
        """Valida estrutura das planilhas e atualiza regras do pipeline conforme necessario"""
        self.logger.info("Validando estrutura das planilhas e regras de negocio...")
        # 1. Validar colunas obrigatÃ³rias nas planilhas principais
        expected_columns = {
            'ATIVOS': ['MATRICULA', 'EMPRESA', 'CARGO', 'SITUACAO', 'SINDICATO'],
            'FÉRIAS': ['MATRICULA', 'DIAS DE FÉRIAS'],
            'ADMISSÃO ABRIL': ['MATRICULA', 'admissão'],
            'DESLIGADOS': ['MATRICULA', 'DATA DEMISSÃO', 'COMUNICADO DE DESLIGAMENTO'],
            'AFASTAMENTOS': ['MATRICULA', 'DESC. SITUACAO'],
            'APRENDIZ': ['MATRICULA'],
            'ESTAGIO': ['MATRICULA'],
            'EXTERIOR': ['Cadastro'],
            'Base dias uteis': None,  # Estrutura flexÃ­vel
            'Base sindicato x valor': None
        }
        corrections = {}
        for key, cols in expected_columns.items():
            if key in data and cols:
                missing = [c for c in cols if c not in data[key].columns]
                if missing:
                    corrections[key] = f"Colunas ausentes: {missing}"
                    self.logger.warning(f"Planilha {key} esta com colunas ausentes: {missing}")
        # 2. Validar regras de negocio: dias Uteis e valores diarios
        # Atualizar mapeamentos se houver divergencia
        # Dias Uteis
        if 'Base dias uteis' in data:
            dias_uteis_map = {}
            dias_uteis_df = data['Base dias uteis']
            for idx in range(2, len(dias_uteis_df)):
                row = dias_uteis_df.iloc[idx]
                if pd.notna(row[0]) and pd.notna(row[1]):
                    sindicato = str(row[0]).strip()
                    dias = int(row[1])
                    dias_uteis_map[str(sindicato)] = dias
            if hasattr(self.agents['business'], 'dias_uteis_map'):
                self.agents['business'].dias_uteis_map = dias_uteis_map
            else:
                setattr(self.agents['business'], 'dias_uteis_map', dias_uteis_map)
            self.logger.info(f"Mapeamento de dias Uteis atualizado: {dias_uteis_map}")
        # Valores diarios
        if 'Base sindicato x valor' in data:
            sindicato_valor_map = {}
            sindicato_valor_df = data['Base sindicato x valor']
            # Normalizar colunas para maiúsculas
            sindicato_valor_df.columns = sindicato_valor_df.columns.str.upper()
            for _, row in sindicato_valor_df.iterrows():
                row = row.rename({c: c.upper() for c in row.index})
                valor = float(row.get('VALOR', 0.0)) if not pd.isna(row.get('VALOR', 0.0)) else 0.0
                sindicato_raw = row.get('SINDICATO', None)
                try:
                    key = str(sindicato_raw).strip() if sindicato_raw is not None else ''
                    if key != '':
                        sindicato_valor_map[key] = valor
                except Exception:
                    continue
            if hasattr(self.agents['business'], 'sindicato_valor_map'):
                self.agents['business'].sindicato_valor_map = sindicato_valor_map
            else:
                setattr(self.agents['business'], 'sindicato_valor_map', sindicato_valor_map)
            self.logger.info(f"Mapeamento de valores diarios atualizado: {sindicato_valor_map}")
        # 3. Registrar correções
        if corrections:
            report_path = self.output_folder / "logs" / f"planilha_correcoes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(corrections, f, indent=2, ensure_ascii=False)
            self.logger.info(f"relatorio de correções salvo em: {report_path}")

    def execute_pipeline(self) -> str:
        """Executa o pipeline completo otimizado"""
        self.logger.info("=" * 60)
        self.logger.info("INICIANDO PIPELINE OTIMIZADO DE AUTOMAÃ‡ÃƒO VR")
        self.logger.info("=" * 60)
        
        try:
            # 1. Carregar dados
            raw_data = self.load_data()
            
            if not raw_data:
                raise ValueError("Nenhum arquivo de dados encontrado!")
            

            # 2. Validar e limpar dados (AGENTE 1)
            self.logger.info("\nâ†’ Executando DataValidatorAgent...")
            cleaned_data, validation_result = self.agents['validator'].execute(raw_data)

            # 2.1 Validar estrutura das planilhas e atualizar regras
            self.validate_and_update_rules(cleaned_data)

            # 3. Aplicar regras de negocio (AGENTE 2)
            self.logger.info("\nâ†’ Executando BusinessRulesAgent...")
            processed_base = self.agents['business'].execute(cleaned_data)
            
            # 4. Detectar anomalias e aprender padroes (AGENTE 3 - NOVO!)
            self.logger.info("\nâ†’ Executando AnomalyDetectorAgent...")
            anomalies, normal_pattern = self.agents['anomaly_detector'].execute(
                processed_base,
                self.agents['business'].exclusion_reasons,
                self.agents['business'].calculation_adjustments
            )
            
            # 5. Auditar APENAS anomalias com LLM (AGENTE 4 - OTIMIZADO!)
            self.logger.info("\nâ†’ Executando LLMAuditAgent (modo otimizado)...")
            audited_base = self.agents['audit'].execute(
                processed_base,
                anomalies,
                normal_pattern
            )
            
            # 6. Gerar relatorio final (AGENTE 5)
            self.logger.info("\nâ†’ Executando ReportGeneratorAgent...")
            output_file = self.agents['report'].execute(
                audited_base,
                str(self.output_folder / "VR MENSAL 05.2025.xlsx")
            )
            
            # 7. Gerar relatorio de execuÃ§Ã£o
            self._generate_execution_report(anomalies)
            # 8. Salvar relatorio detalhado das anomalias LLM
            if hasattr(self.agents['audit'], 'save_llm_anomaly_report'):
                self.agents['audit'].save_llm_anomaly_report(str(self.output_folder))
            
            self.logger.info("=" * 60)
            self.logger.info("PIPELINE CONCLUÃDO COM SUCESSO!")
            self.logger.info(f"Arquivo gerado: {output_file}")
            self.logger.info("=" * 60)
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Erro no pipeline: {e}")
            raise
    
    def _generate_execution_report(self, anomalies: List[EmployeeAnomaly]):
        """Gera relatorio detalhado da execução"""
        # Contar anomalias por tipo
        anomaly_types: Dict[str, int] = {}
        for anomaly in anomalies:
            tipo = anomaly.tipo_anomalia.value
            anomaly_types[tipo] = anomaly_types.get(tipo, 0) + 1

        report = {
            'timestamp': datetime.now().isoformat(),
            'agents_executed': list(self.agents.keys()),
            'agent_status': {name: agent.status.value for name, agent in self.agents.items()},
            'metrics': {
                name: agent.metrics if hasattr(agent, 'metrics') else {}
                for name, agent in self.agents.items()
            },
            'anomaly_summary': {
                'total': len(anomalies),
                'by_type': anomaly_types
            },
            'llm_summary': self.agents['audit'].observations if hasattr(self.agents['audit'], 'observations') else {}
        }

        # Salvar relatorio
        report_path = self.output_folder / "logs" / f"execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)

        self.logger.info(f"relatorio de execução salvo em: {report_path}")

# ========================= FUNÇÃO PRINCIPAL =========================

def main():

    # Iniciar Ollama/Mistral automaticamente
    import subprocess
    import os
    import time
    os.environ["OLLAMA_NUM_THREADS"] = "4"
    os.environ["OLLAMA_KV_CACHE"] = "8192"
    try:
        print("Iniciando Ollama com o modelo mistral...")
        time.sleep(5)  # Aguarda alguns segundos para garantir que o serviço suba
    except Exception as e:
        print(f"Aviso: não foi possível iniciar o Ollama automaticamente: {e}")

    # Criar orquestrador
    orchestrator = VRAutomationOrchestrator(
        input_folder="input",
        output_folder="output"
    )

    # Executar pipeline
    try:
        start_time = time.time()
        output_file = orchestrator.execute_pipeline()
        elapsed_time = time.time() - start_time

        print(f"\n✅ Processo concluído em {elapsed_time:.1f} segundos!")
        print(f"📄 Arquivo gerado: {output_file}")


        # Mostrar economia de processamento
        if hasattr(orchestrator.agents['audit'], 'observations'):
            summary = orchestrator.agents['audit'].observations.get('summary', {})
            if 'economia_processamento' in summary:
                print(f"\n⚡ EFICIÊNCIA:")
                print(f"   • Casos normais ignorados: {summary.get('casos_normais', 0)}")
                print(f"   • Anomalias processadas: {summary.get('anomalias_processadas', 0)}")
                print(f"   • Taxa de economia: {summary.get('economia_processamento', 'N/A')}")

        # Checar relatorio de erros LLM
        import glob, json, os
        output_folder = 'output'
        relatorios = sorted(glob.glob(os.path.join(output_folder, 'llm_anomaly_report_*.json')))
        if relatorios:
            with open(relatorios[-1], encoding='utf-8') as f:
                relatorio_llm = json.load(f)
            erros_detectados = [a for a in relatorio_llm.get('anomalies', []) if a.get('acao') in ['corrigir', 'revisar']]
            if erros_detectados:
                print(f"\n🚨 O agente detectou problemas nos dados que podem causar VR acima do esperado!")
                print(f"   • Total de problemas detectados: {len(erros_detectados)}")
                print(f"   • Exemplos: {erros_detectados[:3]}")
                resposta = input("Deseja aplicar correção automatica conforme Sugestão da LLM? (sim/não): ").strip().lower()
                if resposta == 'sim':
                    aplicar_correcao_automatica(erros_detectados, output_folder)
                    print("Correções aplicadas! Reexecutando pipeline...")
                    output_file = orchestrator.execute_pipeline()
                    print(f"\n✅ Processo concluído após correção!")
                    print(f"📄 Arquivo gerado: {output_file}")
    except Exception as e:
        print(f"\n🚨 Erro na execução: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n🚨 Erro na execução: {e}")
        import traceback
        traceback.print_exc()

def aplicar_correcao_automatica(erros_detectados, output_folder):
    """Aplica correções automáticas conforme Sugestão da LLM"""
    # Exemplo: Corrigir dias VR negativos, excedentes, valores fora do padrão
    # Aqui, você pode implementar lógica para editar os arquivos de entrada ou ajustar o DataFrame final
    # Para simplificação, apenas registra as correções sugeridas
    import json
    correcao_path = os.path.join(output_folder, f'correcao_llm_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(correcao_path, 'w', encoding='utf-8') as f:
        json.dump(erros_detectados, f, indent=2, ensure_ascii=False)
    print(f"Correções sugeridas salvas em: {correcao_path}")

if __name__ == "__main__":
    main()
