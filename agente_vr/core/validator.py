# Validações e limpeza de dados
"""
Módulo data_validator.py
Responsável por validações avançadas e correções automáticas dos dados do sistema VR.

Autor: Agente VR
Data: 2025-08
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Set
import logging
from dataclasses import dataclass
from enum import Enum
import re

class TipoInconsistencia(Enum):
    """Tipos de inconsistências encontradas nos dados"""
    COLABORADOR_ATIVO_E_AFASTADO = "colaborador_ativo_e_afastado"
    COLABORADOR_ATIVO_E_DESLIGADO = "colaborador_ativo_e_desligado"
    DATA_INVALIDA = "data_invalida"
    DATA_FUTURA = "data_futura"
    MATRICULA_DUPLICADA = "matricula_duplicada"
    SINDICATO_SEM_VALOR = "sindicato_sem_valor"
    SINDICATO_SEM_DIAS_UTEIS = "sindicato_sem_dias_uteis"
    FERIAS_INCONSISTENTES = "ferias_inconsistentes"
    CAMPO_OBRIGATORIO_VAZIO = "campo_obrigatorio_vazio"
    DESLIGAMENTO_SEM_DATA = "desligamento_sem_data"
    ADMISSAO_SEM_DATA = "admissao_sem_data"
    VALOR_NEGATIVO = "valor_negativo"
    DIAS_FERIAS_INVALIDOS = "dias_ferias_invalidos"

@dataclass
class Inconsistencia:
    """Representa uma inconsistência encontrada nos dados"""
    tipo: TipoInconsistencia
    matricula: Optional[int]
    base_origem: str
    campo: str
    valor_atual: Any
    valor_sugerido: Any
    descricao: str
    severidade: str  # 'CRITICA', 'ALTA', 'MEDIA', 'BAIXA'
    corrigivel_automaticamente: bool

class DataValidator:
    """
    Classe responsável por validações avançadas e correções automáticas
    dos dados do sistema VR.
    """
    
    def __init__(self, data_competencia: str = "2025-05"):
        """
        Inicializa o validador de dados.
        
        Args:
            data_competencia: Competência no formato YYYY-MM
        """
        self.logger = logging.getLogger(__name__)
        self.data_competencia = datetime.strptime(f"{data_competencia}-01", "%Y-%m-%d")
        self.inconsistencias: List[Inconsistencia] = []
        
        # Mapeamento de sindicatos para estados (baseado nos dados reais)
        self.mapeamento_sindicato_estado = {
            'SINDPD SP': 'São Paulo',
            'SINDPD RJ': 'Rio de Janeiro', 
            'SINDPPD RS': 'Rio Grande do Sul',
            'SITEPD PR': 'Paraná'
        }
        
        # Campos obrigatórios por base
        self.campos_obrigatorios = {
            'ativos': ['matricula', 'cargo', 'situacao', 'sindicato'],
            'ferias': ['matricula', 'dias_ferias'],
            'desligados': ['matricula', 'data_demissao'],
            'admissoes': ['matricula', 'data_admissao'],
            'sindicatos_valores': ['estado', 'valor_diario'],
            'dias_uteis': ['sindicato', 'dias_uteis']
        }
        
    def validar_e_limpar_dados(self, bases: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Inconsistencia]]:
        """
        Valida e corrige automaticamente os dados carregados.
        
        Args:
            bases: Dicionário com todas as bases carregadas
            
        Returns:
            Tuple com (bases_validadas, lista_inconsistencias)
        """
        self.logger.info("✅ Iniciando validações e correções dos dados...")
        self.inconsistencias = []  # Reset
        
        # Criar cópia profunda para não alterar os dados originais
        bases_validadas = self._criar_copia_bases(bases)
        
        try:
            # Etapa 1: Validações básicas de campos obrigatórios
            self._validar_campos_obrigatorios(bases_validadas)
            
            # Etapa 2: Validações de tipos e formatos
            self._validar_tipos_dados(bases_validadas)
            
            # Etapa 3: Validações de datas
            self._validar_datas(bases_validadas)
            
            # Etapa 4: Validações de consistência entre bases
            self._validar_consistencia_entre_bases(bases_validadas)
            
            # Etapa 5: Validações de regras de negócio
            self._validar_regras_negocio(bases_validadas)
            
            # Etapa 6: Validações de configurações (sindicatos, valores)
            self._validar_configuracoes(bases_validadas)
            
            # Etapa 7: Correções automáticas
            self._aplicar_correcoes_automaticas(bases_validadas)
            
            # Resumo das validações
            self._log_resumo_validacoes()
            
            return bases_validadas, self.inconsistencias
            
        except Exception as e:
            self.logger.error(f"❌ Erro durante validações: {e}")
            raise
    
    def _validar_campos_obrigatorios(self, bases: Dict[str, Any]) -> None:
        """Valida se todos os campos obrigatórios estão preenchidos"""
        self.logger.info("🔍 Validando campos obrigatórios...")
        
        for nome_base, campos in self.campos_obrigatorios.items():
            if nome_base not in bases:
                continue
                
            df = bases[nome_base]
            
            for campo in campos:
                if campo not in df.columns:
                    self._registrar_inconsistencia(
                        TipoInconsistencia.CAMPO_OBRIGATORIO_VAZIO,
                        None, nome_base, campo, None, None,
                        f"Campo obrigatório '{campo}' não encontrado na base '{nome_base}'",
                        'CRITICA', False
                    )
                    continue
                
                # Verificar valores vazios/nulos
                valores_vazios = df[df[campo].isna() | (df[campo] == '') | (df[campo] == 0)]
                
                for idx, row in valores_vazios.iterrows():
                    matricula = row.get('matricula', None)
                    self._registrar_inconsistencia(
                        TipoInconsistencia.CAMPO_OBRIGATORIO_VAZIO,
                        matricula, nome_base, campo, row[campo], "PREENCHIMENTO_OBRIGATORIO",
                        f"Campo obrigatório '{campo}' vazio para matrícula {matricula}",
                        'ALTA', False
                    )
    
    def _validar_tipos_dados(self, bases: Dict[str, Any]) -> None:
        """Valida tipos de dados e converte quando necessário"""
        self.logger.info("🔢 Validando tipos de dados...")
        
        # Validar matrículas (devem ser inteiros positivos)
        for nome_base in ['ativos', 'ferias', 'desligados', 'admissoes']:
            if nome_base in bases:
                df = bases[nome_base]
                if 'matricula' in df.columns:
                    # Matrículas inválidas
                    matriculas_invalidas = df[
                        ~df['matricula'].apply(lambda x: isinstance(x, (int, np.integer)) and x > 0)
                    ]
                    
                    for idx, row in matriculas_invalidas.iterrows():
                        self._registrar_inconsistencia(
                            TipoInconsistencia.CAMPO_OBRIGATORIO_VAZIO,
                            None, nome_base, 'matricula', row['matricula'], "MATRICULA_VALIDA",
                            f"Matrícula inválida: {row['matricula']}",
                            'CRITICA', True
                        )
        
        # Validar valores monetários
        if 'sindicatos_valores' in bases:
            df = bases['sindicatos_valores']
            if 'valor_diario' in df.columns:
                valores_negativos = df[df['valor_diario'] <= 0]
                
                for idx, row in valores_negativos.iterrows():
                    self._registrar_inconsistencia(
                        TipoInconsistencia.VALOR_NEGATIVO,
                        None, 'sindicatos_valores', 'valor_diario', 
                        row['valor_diario'], "VALOR_POSITIVO",
                        f"Valor diário negativo ou zero para estado {row.get('estado', 'N/A')}",
                        'ALTA', True
                    )
        
        # Validar dias de férias
        if 'ferias' in bases:
            df = bases['ferias']
            if 'dias_ferias' in df.columns:
                dias_invalidos = df[
                    (df['dias_ferias'] < 0) | (df['dias_ferias'] > 30)
                ]
                
                for idx, row in dias_invalidos.iterrows():
                    self._registrar_inconsistencia(
                        TipoInconsistencia.DIAS_FERIAS_INVALIDOS,
                        row.get('matricula'), 'ferias', 'dias_ferias',
                        row['dias_ferias'], "ENTRE_0_E_30",
                        f"Dias de férias inválidos: {row['dias_ferias']}",
                        'MEDIA', True
                    )
    
    def _validar_datas(self, bases: Dict[str, Any]) -> None:
        """Valida consistência e formato das datas"""
        self.logger.info("📅 Validando datas...")
        
        data_limite_futura = self.data_competencia + timedelta(days=365)  # 1 ano no futuro
        data_limite_passada = datetime(2000, 1, 1)  # Limite razoável no passado
        
        # Validar datas de desligamento
        if 'desligados' in bases:
            df = bases['desligados']
            if 'data_demissao' in df.columns:
                # Datas nulas
                datas_nulas = df[df['data_demissao'].isna()]
                for idx, row in datas_nulas.iterrows():
                    self._registrar_inconsistencia(
                        TipoInconsistencia.DESLIGAMENTO_SEM_DATA,
                        row.get('matricula'), 'desligados', 'data_demissao',
                        None, "DATA_VALIDA",
                        f"Data de desligamento não informada",
                        'CRITICA', False
                    )
                
                # Datas futuras
                datas_futuras = df[df['data_demissao'] > data_limite_futura]
                for idx, row in datas_futuras.iterrows():
                    self._registrar_inconsistencia(
                        TipoInconsistencia.DATA_FUTURA,
                        row.get('matricula'), 'desligados', 'data_demissao',
                        row['data_demissao'], self.data_competencia.strftime('%Y-%m-%d'),
                        f"Data de desligamento no futuro: {row['data_demissao']}",
                        'ALTA', True
                    )
                
                # Datas muito antigas
                datas_antigas = df[df['data_demissao'] < data_limite_passada]
                for idx, row in datas_antigas.iterrows():
                    self._registrar_inconsistencia(
                        TipoInconsistencia.DATA_INVALIDA,
                        row.get('matricula'), 'desligados', 'data_demissao',
                        row['data_demissao'], "DATA_RAZOAVEL",
                        f"Data de desligamento muito antiga: {row['data_demissao']}",
                        'MEDIA', False
                    )
        
        # Validar datas de admissão
        if 'admissoes' in bases:
            df = bases['admissoes']
            if 'data_admissao' in df.columns:
                # Datas nulas
                datas_nulas = df[df['data_admissao'].isna()]
                for idx, row in datas_nulas.iterrows():
                    self._registrar_inconsistencia(
                        TipoInconsistencia.ADMISSAO_SEM_DATA,
                        row.get('matricula'), 'admissoes', 'data_admissao',
                        None, "DATA_VALIDA",
                        f"Data de admissão não informada",
                        'CRITICA', False
                    )
                
                # Datas futuras
                datas_futuras = df[df['data_admissao'] > data_limite_futura]
                for idx, row in datas_futuras.iterrows():
                    self._registrar_inconsistencia(
                        TipoInconsistencia.DATA_FUTURA,
                        row.get('matricula'), 'admissoes', 'data_admissao',
                        row['data_admissao'], self.data_competencia.strftime('%Y-%m-%d'),
                        f"Data de admissão no futuro: {row['data_admissao']}",
                        'ALTA', True
                    )
    
    def _validar_consistencia_entre_bases(self, bases: Dict[str, Any]) -> None:
        """Valida consistência entre diferentes bases"""
        self.logger.info("🔄 Validando consistência entre bases...")
        
        # Extrair conjuntos de matrículas
        matriculas_ativos = set()
        matriculas_desligados = set()
        matriculas_afastados = set()
        matriculas_estagiarios = set()
        matriculas_aprendizes = set()
        matriculas_exterior = set()
        
        if 'ativos' in bases:
            matriculas_ativos = set(bases['ativos']['matricula'])
        
        if 'desligados' in bases:
            matriculas_desligados = set(bases['desligados']['matricula'])
        
        if 'exclusoes' in bases:
            if 'afastamentos' in bases['exclusoes']:
                matriculas_afastados = set(bases['exclusoes']['afastamentos']['matricula'])
            
            if 'estagiarios' in bases['exclusoes']:
                matriculas_estagiarios = set(bases['exclusoes']['estagiarios']['matricula'])
            
            if 'aprendizes' in bases['exclusoes']:
                matriculas_aprendizes = set(bases['exclusoes']['aprendizes']['matricula'])
            
            if 'exterior' in bases['exclusoes']:
                matriculas_exterior = set(bases['exclusoes']['exterior']['matricula'])
        
        # Validação 1: Colaboradores ativos E desligados
        ativos_e_desligados = matriculas_ativos.intersection(matriculas_desligados)
        for matricula in ativos_e_desligados:
            self._registrar_inconsistencia(
                TipoInconsistencia.COLABORADOR_ATIVO_E_DESLIGADO,
                matricula, 'ativos+desligados', 'matricula',
                matricula, "ESCOLHER_STATUS_CORRETO",
                f"Colaborador {matricula} aparece como ativo E desligado",
                'CRITICA', True
            )
        
        # Validação 2: Colaboradores ativos E afastados
        ativos_e_afastados = matriculas_ativos.intersection(matriculas_afastados)
        for matricula in ativos_e_afastados:
            self._registrar_inconsistencia(
                TipoInconsistencia.COLABORADOR_ATIVO_E_AFASTADO,
                matricula, 'ativos+afastamentos', 'matricula',
                matricula, "REMOVER_DE_ATIVOS",
                f"Colaborador {matricula} aparece como ativo E afastado",
                'ALTA', True
            )
        
        # Validação 3: Colaboradores ativos E estagiários/aprendizes
        ativos_e_estagiarios = matriculas_ativos.intersection(matriculas_estagiarios)
        ativos_e_aprendizes = matriculas_ativos.intersection(matriculas_aprendizes)
        
        for matricula in ativos_e_estagiarios.union(ativos_e_aprendizes):
            tipo_exclusao = "estagiário" if matricula in ativos_e_estagiarios else "aprendiz"
            self._registrar_inconsistencia(
                TipoInconsistencia.COLABORADOR_ATIVO_E_AFASTADO,
                matricula, f'ativos+{tipo_exclusao}', 'matricula',
                matricula, "REMOVER_DE_ATIVOS",
                f"Colaborador {matricula} é ativo mas está marcado como {tipo_exclusao}",
                'ALTA', True
            )
        
        # Validação 4: Matrículas duplicadas dentro da mesma base
        for nome_base in ['ativos', 'ferias', 'desligados', 'admissoes']:
            if nome_base in bases:
                df = bases[nome_base]
                if 'matricula' in df.columns:
                    duplicadas = df[df.duplicated('matricula', keep=False)]
                    for matricula in duplicadas['matricula'].unique():
                        self._registrar_inconsistencia(
                            TipoInconsistencia.MATRICULA_DUPLICADA,
                            matricula, nome_base, 'matricula',
                            matricula, "MANTER_APENAS_UM_REGISTRO",
                            f"Matrícula {matricula} duplicada na base {nome_base}",
                            'ALTA', True
                        )
    
    def _validar_regras_negocio(self, bases: Dict[str, Any]) -> None:
        """Valida regras específicas do negócio VR"""
        self.logger.info("📋 Validando regras de negócio...")
        
        # Validação: Férias não podem exceder dias úteis do sindicato
        if 'ferias' in bases and 'ativos' in bases and 'dias_uteis' in bases:
            ferias = bases['ferias']
            ativos = bases['ativos']
            dias_uteis = bases['dias_uteis']
            
            # Criar mapeamento matrícula → sindicato
            matricula_sindicato = dict(zip(ativos['matricula'], ativos['sindicato']))
            
            # Criar mapeamento sindicato → dias úteis
            sindicato_dias = {}
            for _, row in dias_uteis.iterrows():
                nome_sindicato = row['sindicato']
                # Mapear nome completo para sigla
                for sigla, _ in self.mapeamento_sindicato_estado.items():
                    if sigla in nome_sindicato:
                        sindicato_dias[sigla] = row['dias_uteis']
                        break
            
            # Validar férias por colaborador
            for _, row in ferias.iterrows():
                matricula = row['matricula']
                dias_ferias = row['dias_ferias']
                
                if matricula in matricula_sindicato:
                    sindicato_colaborador = matricula_sindicato[matricula]
                    
                    # Encontrar dias úteis do sindicato
                    dias_uteis_sindicato = None
                    for sigla in self.mapeamento_sindicato_estado.keys():
                        if sigla in sindicato_colaborador and sigla in sindicato_dias:
                            dias_uteis_sindicato = sindicato_dias[sigla]
                            break
                    
                    if dias_uteis_sindicato and dias_ferias > dias_uteis_sindicato:
                        self._registrar_inconsistencia(
                            TipoInconsistencia.FERIAS_INCONSISTENTES,
                            matricula, 'ferias', 'dias_ferias',
                            dias_ferias, dias_uteis_sindicato,
                            f"Férias ({dias_ferias}) excedem dias úteis do sindicato ({dias_uteis_sindicato})",
                            'MEDIA', True
                        )
    
    def _validar_configuracoes(self, bases: Dict[str, Any]) -> None:
        """Valida configurações de sindicatos e valores"""
        self.logger.info("⚙️ Validando configurações...")
        
        # Coletar sindicatos únicos dos colaboradores ativos
        sindicatos_colaboradores = set()
        if 'ativos' in bases:
            sindicatos_colaboradores = set(bases['ativos']['sindicato'].unique())
        
        # Verificar se todos os sindicatos têm valor diário
        estados_com_valor = set()
        if 'sindicatos_valores' in bases:
            estados_com_valor = set(bases['sindicatos_valores']['estado'].unique())
        
        for sindicato in sindicatos_colaboradores:
            estado_correspondente = None
            
            # Mapear sindicato para estado
            for sigla, estado in self.mapeamento_sindicato_estado.items():
                if sigla in sindicato:
                    estado_correspondente = estado
                    break
            
            if not estado_correspondente:
                self._registrar_inconsistencia(
                    TipoInconsistencia.SINDICATO_SEM_VALOR,
                    None, 'sindicatos_valores', 'estado',
                    sindicato, "MAPEAR_PARA_ESTADO",
                    f"Sindicato '{sindicato}' não mapeado para estado",
                    'ALTA', False
                )
            elif estado_correspondente not in estados_com_valor:
                self._registrar_inconsistencia(
                    TipoInconsistencia.SINDICATO_SEM_VALOR,
                    None, 'sindicatos_valores', 'valor_diario',
                    estado_correspondente, "DEFINIR_VALOR",
                    f"Estado '{estado_correspondente}' sem valor diário definido",
                    'CRITICA', False
                )
        
        # Verificar se todos os sindicatos têm dias úteis
        sindicatos_com_dias = set()
        if 'dias_uteis' in bases:
            for sindicato_completo in bases['dias_uteis']['sindicato']:
                for sigla in self.mapeamento_sindicato_estado.keys():
                    if sigla in sindicato_completo:
                        sindicatos_com_dias.add(sigla)
        
        for sindicato in sindicatos_colaboradores:
            tem_dias_uteis = False
            for sigla in self.mapeamento_sindicato_estado.keys():
                if sigla in sindicato and sigla in sindicatos_com_dias:
                    tem_dias_uteis = True
                    break
            
            if not tem_dias_uteis:
                self._registrar_inconsistencia(
                    TipoInconsistencia.SINDICATO_SEM_DIAS_UTEIS,
                    None, 'dias_uteis', 'dias_uteis',
                    sindicato, "DEFINIR_DIAS_UTEIS",
                    f"Sindicato '{sindicato}' sem dias úteis definidos",
                    'CRITICA', False
                )
    
    def _aplicar_correcoes_automaticas(self, bases: Dict[str, Any]) -> None:
        """Aplica correções automáticas onde possível"""
        self.logger.info("🔧 Aplicando correções automáticas...")
        
        correcoes_aplicadas = 0
        
        # Correção 1: Remover colaboradores afastados da base de ativos
        if 'ativos' in bases and 'exclusoes' in bases:
            matriculas_excluir = set()
            
            # Coletar todas as matrículas que devem ser excluídas
            for tipo_exclusao, df_exclusao in bases['exclusoes'].items():
                if 'matricula' in df_exclusao.columns:
                    matriculas_excluir.update(df_exclusao['matricula'])
            
            # Remover da base de ativos
            tamanho_original = len(bases['ativos'])
            bases['ativos'] = bases['ativos'][~bases['ativos']['matricula'].isin(matriculas_excluir)]
            removidos = tamanho_original - len(bases['ativos'])
            
            if removidos > 0:
                self.logger.info(f"   ✅ Removidos {removidos} colaboradores afastados da base de ativos")
                correcoes_aplicadas += removidos
        
        # Correção 2: Limitar dias de férias ao máximo de dias úteis
        if 'ferias' in bases and 'dias_uteis' in bases:
            # Para simplificar, usar 22 dias como limite máximo (SP e PR)
            dias_maximos = 22
            
            ferias_excessivas = bases['ferias']['dias_ferias'] > dias_maximos
            if ferias_excessivas.any():
                bases['ferias'].loc[ferias_excessivas, 'dias_ferias'] = dias_maximos
                corrigidas = ferias_excessivas.sum()
                self.logger.info(f"   ✅ Corrigidos {corrigidas} registros de férias excessivas")
                correcoes_aplicadas += corrigidas
        
        # Correção 3: Remover matrículas duplicadas (manter primeira ocorrência)
        for nome_base in ['ativos', 'ferias', 'desligados', 'admissoes']:
            if nome_base in bases and 'matricula' in bases[nome_base].columns:
                tamanho_original = len(bases[nome_base])
                bases[nome_base] = bases[nome_base].drop_duplicates('matricula', keep='first')
                removidos = tamanho_original - len(bases[nome_base])
                
                if removidos > 0:
                    self.logger.info(f"   ✅ Removidas {removidos} matrículas duplicadas de {nome_base}")
                    correcoes_aplicadas += removidos
        
        # Correção 4: Converter valores negativos em zero
        if 'sindicatos_valores' in bases:
            valores_negativos = bases['sindicatos_valores']['valor_diario'] <= 0
            if valores_negativos.any():
                bases['sindicatos_valores'].loc[valores_negativos, 'valor_diario'] = 35.0  # Valor padrão
                corrigidos = valores_negativos.sum()
                self.logger.info(f"   ✅ Corrigidos {corrigidos} valores negativos para valor padrão R$ 35,00")
                correcoes_aplicadas += corrigidos
        
        self.logger.info(f"🎯 Total de correções automáticas aplicadas: {correcoes_aplicadas}")
    
    def _registrar_inconsistencia(
        self, 
        tipo: TipoInconsistencia,
        matricula: Optional[int],
        base_origem: str,
        campo: str,
        valor_atual: Any,
        valor_sugerido: Any,
        descricao: str,
        severidade: str,
        corrigivel_automaticamente: bool
    ) -> None:
        """Registra uma inconsistência encontrada"""
        inconsistencia = Inconsistencia(
            tipo=tipo,
            matricula=matricula,
            base_origem=base_origem,
            campo=campo,
            valor_atual=valor_atual,
            valor_sugerido=valor_sugerido,
            descricao=descricao,
            severidade=severidade,
            corrigivel_automaticamente=corrigivel_automaticamente
        )
        
        self.inconsistencias.append(inconsistencia)
    
    def _criar_copia_bases(self, bases: Dict[str, Any]) -> Dict[str, Any]:
        """Cria uma cópia profunda das bases para não alterar os dados originais"""
        bases_copia = {}
        
        for nome, valor in bases.items():
            if isinstance(valor, pd.DataFrame):
                bases_copia[nome] = valor.copy()
            elif isinstance(valor, dict):
                bases_copia[nome] = {}
                for sub_nome, sub_valor in valor.items():
                    if isinstance(sub_valor, pd.DataFrame):
                        bases_copia[nome][sub_nome] = sub_valor.copy()
                    else:
                        bases_copia[nome][sub_nome] = sub_valor
            else:
                bases_copia[nome] = valor
        
        return bases_copia
    
    def _log_resumo_validacoes(self) -> None:
        """Registra resumo das validações realizadas"""
        total = len(self.inconsistencias)
        
        if total == 0:
            self.logger.info("✅ Nenhuma inconsistência encontrada!")
            return
        
        # Contar por severidade
        criticas = sum(1 for i in self.inconsistencias if i.severidade == 'CRITICA')
        altas = sum(1 for i in self.inconsistencias if i.severidade == 'ALTA')
        medias = sum(1 for i in self.inconsistencias if i.severidade == 'MEDIA')
        baixas = sum(1 for i in self.inconsistencias if i.severidade == 'BAIXA')
        
        # Contar corrigíveis automaticamente
        corrigiveis = sum(1 for i in self.inconsistencias if i.corrigivel_automaticamente)
        
        self.logger.info(f"📊 RESUMO DAS VALIDAÇÕES:")
        self.logger.info(f"   Total de inconsistências: {total}")
        self.logger.info(f"   🔴 Críticas: {criticas}")
        self.logger.info(f"   🟠 Altas: {altas}")
        self.logger.info(f"   🟡 Médias: {medias}")
        self.logger.info(f"   🟢 Baixas: {baixas}")
        self.logger.info(f"   🔧 Corrigíveis automaticamente: {corrigiveis}")
    
    def gerar_relatorio_inconsistencias(self) -> pd.DataFrame:
        """
        Gera relatório detalhado das inconsistências encontradas.
        
        Returns:
            DataFrame com todas as inconsistências
        """
        if not self.inconsistencias:
            return pd.DataFrame()
        
        dados_relatorio = []
        
        for inc in self.inconsistencias:
            dados_relatorio.append({
                'Tipo': inc.tipo.value,
                'Matrícula': inc.matricula,
                'Base': inc.base_origem,
                'Campo': inc.campo,
                'Valor Atual': inc.valor_atual,
                'Valor Sugerido': inc.valor_sugerido,
                'Descrição': inc.descricao,
                'Severidade': inc.severidade,
                'Corrigível Auto': inc.corrigivel_automaticamente
            })
        
        return pd.DataFrame(dados_relatorio)
    
    def salvar_relatorio_inconsistencias(self, caminho: str = "data/output/relatorio_inconsistencias.xlsx") -> None:
        """
        Salva relatório de inconsistências em Excel.
        
        Args:
            caminho: Caminho do arquivo de saída
        """
        relatorio = self.gerar_relatorio_inconsistencias()
        
        if not relatorio.empty:
            relatorio.to_excel(caminho, index=False)
            self.logger.info(f"📋 Relatório de inconsistências salvo em: {caminho}")
        else:
            self.logger.info("✅ Nenhuma inconsistência para salvar no relatório")


# Função utilitária para uso direto
def validar_bases_vr(bases: Dict[str, Any], data_competencia: str = "2025-05") -> Tuple[Dict[str, Any], List[Inconsistencia]]:
    """
    Função utilitária para validar bases de VR.
    
    Args:
        bases: Bases carregadas pelo data_reader
        data_competencia: Competência no formato YYYY-MM
        
    Returns:
        Tuple com (bases_validadas, inconsistencias)
    """
    validator = DataValidator(data_competencia)
    return validator.validar_e_limpar_dados(bases)


# Exemplo de uso
if __name__ == "__main__":
    import logging
    from data_reader import carregar_bases_vr
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Testar validações
    try:
        # Carregar dados
        bases = carregar_bases_vr("data/input")
        
        # Validar dados
        bases_validadas, inconsistencias = validar_bases_vr(bases)
        
        print("✅ Validações concluídas!")
        print(f"📊 Inconsistências encontradas: {len(inconsistencias)}")
        
        # Mostrar algumas inconsistências
        validator = DataValidator()
        relatorio = validator.gerar_relatorio_inconsistencias()
        
        if not relatorio.empty:
            print("\n📋 Primeiras inconsistências:")
            print(relatorio.head().to_string())
        
    except Exception as e:
        print(f"❌ Erro: {e}")