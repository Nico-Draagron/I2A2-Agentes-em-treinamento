"""
Módulo ai_analytics.py
Sistema híbrido de análises avançadas com IA para detecção de fraudes e inconsistências.

Autor: Agente VR + IA
Data: 2025-08
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import re
from difflib import SequenceMatcher
from decimal import Decimal

# Machine Learning para detecção de anomalias
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy import stats

# Visualização
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt


@dataclass
class InconsistenciaDetectada:
    """Representa uma inconsistência detectada pelo sistema"""
    tipo: str
    gravidade: str  # 'CRITICA', 'ALTA', 'MEDIA', 'BAIXA'
    matricula: Optional[str]
    campo_afetado: str
    valor_original: Any
    valor_sugerido: Any
    confianca: float  # 0-1
    detalhes: str
    corrigivel_automaticamente: bool
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RelatorioFraudes:
    """Relatório consolidado de fraudes e inconsistências"""
    total_inconsistencias: int
    inconsistencias_criticas: int
    inconsistencias_corrigidas: int
    score_integridade: float  # 0-100
    detalhes: List[InconsistenciaDetectada]
    recomendacoes: List[str] = field(default_factory=list)


class AIAnalyticsVR:
    """
    Sistema híbrido de análises avançadas com IA para VR
    Combina Machine Learning tradicional com análises estatísticas
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configurar modelos ML
        self.modelo_outliers = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
        # Dicionários para correção inteligente de nomes de colunas
        self.mapeamento_colunas_corretas = {
            # Variações comuns de nomes de colunas
            'matricula': ['matricula', 'matrícula', 'matricla', 'matricul', 'registro', 'codigo', 'id', 'identificacao', 'mat'],
            'nome': ['nome', 'nome_completo', 'funcionario', 'colaborador', 'empregado', 'nom', 'nme'],
            'admissao': ['admissao', 'admissão', 'data_admissao', 'dt_admissao', 'inicio', 'admisao', 'admisão'],
            'data_demissao': ['data_demissao', 'data_demissão', 'demissao', 'demissão', 'desligamento', 'saida', 'fim', 'termino'],
            'valor_diario': ['valor_diario', 'valor_diário', 'valor', 'valor_vr', 'vr', 'vale_refeicao', 'quantia', 'vlr_diario'],
            'estado': ['estado', 'uf', 'regiao', 'região', 'stad', 'est'],
            'sindicato': ['sindicato', 'sindical', 'sind', 'sindicto'],
            'situacao': ['situacao', 'situação', 'status', 'ativo', 'condicao', 'condicão', 'sit'],
            'empresa': ['empresa', 'emp', 'empres'],
            'cargo': ['cargo', 'funcao', 'função', 'carg'],
            'dias_uteis': ['dias_uteis', 'dias_úteis', 'dias', 'du', 'dias_trabalhados'],
            'dias_ferias': ['dias_ferias', 'dias_férias', 'ferias', 'férias', 'df', 'dias_f'],
            'comunicado_ok': ['comunicado_ok', 'comunicado', 'ok', 'comm_ok', 'comunicdo_ok']
        }
        
        self.logger.info("🤖 AI Analytics VR inicializado")
    
    
    def detectar_fraudes_inconsistencias(
        self, 
        bases_dados: Dict[str, pd.DataFrame],
        calculos_vr: Optional[List] = None
    ) -> RelatorioFraudes:
        """
        Detecta fraudes e inconsistências usando sistema híbrido ML + análise estatística
        """
        self.logger.info("🔍 Iniciando detecção de fraudes e inconsistências...")
        
        inconsistencias = []
        
        # 1. Correção automática de nomes de colunas
        bases_corrigidas = self._corrigir_nomes_colunas(bases_dados)
        inconsistencias.extend(self._detectar_inconsistencias_colunas(bases_dados, bases_corrigidas))
        
        # 2. Detecção de outliers financeiros
        if calculos_vr:
            inconsistencias.extend(self._detectar_outliers_financeiros(calculos_vr))
        
        # 3. Detecção de inconsistências entre bases
        inconsistencias.extend(self._detectar_inconsistencias_cruzadas(bases_corrigidas))
        
        # 4. Validação de integridade de dados
        inconsistencias.extend(self._validar_integridade_dados(bases_corrigidas))
        
        # 5. Detecção de padrões suspeitos
        inconsistencias.extend(self._detectar_padroes_suspeitos(bases_corrigidas))
        
        # 6. Análise estatística avançada
        inconsistencias.extend(self._analise_estatistica_avancada(bases_corrigidas))
        
        # 7. Gerar relatório consolidado
        relatorio = self._gerar_relatorio_fraudes(inconsistencias)
        
        self.logger.info(f"✅ Análise concluída: {len(inconsistencias)} inconsistências detectadas")
        return relatorio
    
    
    def _corrigir_nomes_colunas(self, bases_dados: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Corrige automaticamente nomes de colunas com erros de digitação
        """
        bases_corrigidas = {}
        
        for nome_base, df in bases_dados.items():
            df_corrigido = df.copy()
            colunas_originais = list(df.columns)
            colunas_corrigidas = []
            
            for coluna in colunas_originais:
                coluna_corrigida = self._encontrar_coluna_correta(coluna.lower().strip())
                colunas_corrigidas.append(coluna_corrigida)
            
            # Aplicar correções
            df_corrigido.columns = colunas_corrigidas
            bases_corrigidas[nome_base] = df_corrigido
            
            # Log das correções aplicadas
            if colunas_originais != colunas_corrigidas:
                for orig, corr in zip(colunas_originais, colunas_corrigidas):
                    if orig != corr:
                        self.logger.info(f"📝 {nome_base}: '{orig}' → '{corr}'")
        
        return bases_corrigidas
    
    
    def _encontrar_coluna_correta(self, nome_coluna: str) -> str:
        """
        Encontra o nome correto da coluna usando similaridade de strings
        """
        melhor_match = nome_coluna
        melhor_score = 0
        
        for coluna_padrao, variacoes in self.mapeamento_colunas_corretas.items():
            for variacao in variacoes:
                score = SequenceMatcher(None, nome_coluna, variacao.lower()).ratio()
                if score > melhor_score and score > 0.75:  # 75% de similaridade
                    melhor_score = score
                    melhor_match = coluna_padrao
        
        return melhor_match
    
    
    def _detectar_inconsistencias_colunas(
        self, 
        bases_originais: Dict[str, pd.DataFrame],
        bases_corrigidas: Dict[str, pd.DataFrame]
    ) -> List[InconsistenciaDetectada]:
        """
        Detecta inconsistências nos nomes das colunas
        """
        inconsistencias = []
        
        for nome_base in bases_originais:
            colunas_orig = list(bases_originais[nome_base].columns)
            colunas_corr = list(bases_corrigidas[nome_base].columns)
            
            for orig, corr in zip(colunas_orig, colunas_corr):
                if orig != corr:
                    score = SequenceMatcher(None, orig.lower(), corr.lower()).ratio()
                    inconsistencias.append(InconsistenciaDetectada(
                        tipo="ERRO_NOME_COLUNA",
                        gravidade="MEDIA",
                        matricula=None,
                        campo_afetado=f"{nome_base}.{orig}",
                        valor_original=orig,
                        valor_sugerido=corr,
                        confianca=score,
                        detalhes=f"Nome de coluna corrigido automaticamente (similaridade: {score:.2f})",
                        corrigivel_automaticamente=True
                    ))
        
        return inconsistencias
    
    
    def _detectar_outliers_financeiros(self, calculos_vr: List) -> List[InconsistenciaDetectada]:
        """
        Detecta outliers financeiros usando Machine Learning
        """
        inconsistencias = []
        
        if not calculos_vr:
            return inconsistencias
        
        # Converter para DataFrame
        dados_financeiros = []
        for calculo in calculos_vr:
            if hasattr(calculo, 'matricula') and hasattr(calculo, 'valor_total_vr'):
                dados_financeiros.append({
                    'matricula': calculo.matricula,
                    'valor_total': float(calculo.valor_total_vr),
                    'dias_pagos': getattr(calculo, 'dias_vr_pagos', 0),
                    'valor_diario': float(calculo.valor_diario_vr) if hasattr(calculo, 'valor_diario_vr') else 0
                })
        
        if len(dados_financeiros) < 5:
            return inconsistencias
        
        df_financeiro = pd.DataFrame(dados_financeiros)
        
        # Detectar outliers usando Isolation Forest
        X = df_financeiro[['valor_total', 'dias_pagos', 'valor_diario']].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        outliers = self.modelo_outliers.fit_predict(X_scaled)
        scores = self.modelo_outliers.decision_function(X_scaled)
        
        # Analisar outliers
        for i, (is_outlier, score) in enumerate(zip(outliers, scores)):
            if is_outlier == -1:  # É outlier
                registro = dados_financeiros[i]
                
                # Determinar gravidade baseada no score
                if score < -0.5:
                    gravidade = "CRITICA"
                elif score < -0.3:
                    gravidade = "ALTA"
                else:
                    gravidade = "MEDIA"
                
                valor_mediano = df_financeiro['valor_total'].median()
                inconsistencias.append(InconsistenciaDetectada(
                    tipo="OUTLIER_FINANCEIRO",
                    gravidade=gravidade,
                    matricula=str(registro['matricula']),
                    campo_afetado="valor_vr",
                    valor_original=registro['valor_total'],
                    valor_sugerido=valor_mediano,
                    confianca=abs(score),
                    detalhes=f"Valor suspeito: R$ {registro['valor_total']:.2f} (score: {score:.3f})",
                    corrigivel_automaticamente=False
                ))
        
        return inconsistencias
    
    
    def _detectar_inconsistencias_cruzadas(self, bases_dados: Dict[str, pd.DataFrame]) -> List[InconsistenciaDetectada]:
        """
        Detecta inconsistências entre diferentes bases de dados
        """
        inconsistencias = []
        
        # Verificar se colaborador ativo tem desligamento
        if 'ativos' in bases_dados and 'desligados' in bases_dados:
            df_ativos = bases_dados['ativos']
            df_desligados = bases_dados['desligados']
            
            if 'matricula' in df_ativos.columns and 'matricula' in df_desligados.columns:
                matriculas_ativas = set(df_ativos['matricula'].astype(str))
                matriculas_desligadas = set(df_desligados['matricula'].astype(str))
                
                # Colaboradores em ambas as listas
                duplicatas = matriculas_ativas.intersection(matriculas_desligadas)
                
                for matricula in duplicatas:
                    inconsistencias.append(InconsistenciaDetectada(
                        tipo="COLABORADOR_ATIVO_E_DESLIGADO",
                        gravidade="CRITICA",
                        matricula=matricula,
                        campo_afetado="situacao",
                        valor_original="ATIVO_E_DESLIGADO",
                        valor_sugerido="VERIFICAR_MANUALMENTE",
                        confianca=1.0,
                        detalhes=f"Colaborador {matricula} está nas bases ATIVOS e DESLIGADOS",
                        corrigivel_automaticamente=False
                    ))
        
        # Verificar valores negativos ou zerados suspeitos
        for nome_base, df in bases_dados.items():
            colunas_valor = [col for col in df.columns if 'valor' in col.lower()]
            
            for col_valor in colunas_valor:
                if col_valor in df.columns:
                    valores_suspeitos = df[
                        (df[col_valor] <= 0) | 
                        (df[col_valor].isna()) | 
                        (df[col_valor] > 1000)  # Valor muito alto para VR
                    ]
                    
                    for _, row in valores_suspeitos.iterrows():
                        valor_mediano = df[col_valor].median()
                        inconsistencias.append(InconsistenciaDetectada(
                            tipo="VALOR_SUSPEITO",
                            gravidade="ALTA",
                            matricula=str(row.get('matricula', 'N/A')),
                            campo_afetado=col_valor,
                            valor_original=row[col_valor],
                            valor_sugerido=valor_mediano,
                            confianca=0.8,
                            detalhes=f"Valor suspeito em {nome_base}: {row[col_valor]}",
                            corrigivel_automaticamente=True
                        ))
        
        return inconsistencias
    
    
    def _validar_integridade_dados(self, bases_dados: Dict[str, pd.DataFrame]) -> List[InconsistenciaDetectada]:
        """
        Valida integridade geral dos dados
        """
        inconsistencias = []
        
        for nome_base, df in bases_dados.items():
            # Verificar duplicatas
            if 'matricula' in df.columns:
                duplicatas = df[df.duplicated(subset=['matricula'], keep=False)]
                
                for matricula in duplicatas['matricula'].unique():
                    inconsistencias.append(InconsistenciaDetectada(
                        tipo="MATRICULA_DUPLICADA",
                        gravidade="ALTA",
                        matricula=str(matricula),
                        campo_afetado="matricula",
                        valor_original=f"DUPLICADA_EM_{nome_base}",
                        valor_sugerido="REMOVER_DUPLICATAS",
                        confianca=1.0,
                        detalhes=f"Matrícula {matricula} duplicada em {nome_base}",
                        corrigivel_automaticamente=True
                    ))
            
            # Verificar campos obrigatórios vazios
            campos_obrigatorios = ['matricula']
            if nome_base in ['ativos', 'desligados']:
                if 'nome' in df.columns:
                    campos_obrigatorios.append('nome')
            
            for campo in campos_obrigatorios:
                if campo in df.columns:
                    vazios = df[df[campo].isna() | (df[campo] == '')]
                    
                    for _, row in vazios.iterrows():
                        inconsistencias.append(InconsistenciaDetectada(
                            tipo="CAMPO_OBRIGATORIO_VAZIO",
                            gravidade="CRITICA",
                            matricula=str(row.get('matricula', 'N/A')),
                            campo_afetado=campo,
                            valor_original="VAZIO",
                            valor_sugerido="PREENCHER_MANUALMENTE",
                            confianca=1.0,
                            detalhes=f"Campo obrigatório '{campo}' vazio em {nome_base}",
                            corrigivel_automaticamente=False
                        ))
        
        return inconsistencias
    
    
    def _detectar_padroes_suspeitos(self, bases_dados: Dict[str, pd.DataFrame]) -> List[InconsistenciaDetectada]:
        """
        Detecta padrões suspeitos usando análise estatística
        """
        inconsistencias = []
        
        # Detectar matrículas com padrão suspeito
        for nome_base, df in bases_dados.items():
            if 'matricula' in df.columns:
                matriculas = df['matricula'].astype(str).tolist()
                
                for matricula in matriculas:
                    if self._is_padrao_suspeito_matricula(matricula):
                        inconsistencias.append(InconsistenciaDetectada(
                            tipo="MATRICULA_PADRAO_SUSPEITO",
                            gravidade="MEDIA",
                            matricula=matricula,
                            campo_afetado="matricula",
                            valor_original=matricula,
                            valor_sugerido="VERIFICAR_MANUALMENTE",
                            confianca=0.7,
                            detalhes=f"Matrícula com padrão suspeito: {matricula}",
                            corrigivel_automaticamente=False
                        ))
        
        return inconsistencias
    
    
    def _analise_estatistica_avancada(self, bases_dados: Dict[str, pd.DataFrame]) -> List[InconsistenciaDetectada]:
        """
        Análise estatística avançada para detectar anomalias
        """
        inconsistencias = []
        
        for nome_base, df in bases_dados.items():
            # Analisar colunas numéricas
            colunas_numericas = df.select_dtypes(include=[np.number]).columns
            
            for coluna in colunas_numericas:
                if len(df[coluna].dropna()) > 5:  # Mínimo de 5 valores
                    # Detectar outliers usando Z-score
                    z_scores = np.abs(stats.zscore(df[coluna].dropna()))
                    outliers_indices = np.where(z_scores > 3)[0]  # Z-score > 3
                    
                    for idx in outliers_indices:
                        valor_original = df.iloc[idx][coluna]
                        valor_mediano = df[coluna].median()
                        
                        inconsistencias.append(InconsistenciaDetectada(
                            tipo="OUTLIER_ESTATISTICO",
                            gravidade="MEDIA",
                            matricula=str(df.iloc[idx].get('matricula', 'N/A')),
                            campo_afetado=coluna,
                            valor_original=valor_original,
                            valor_sugerido=valor_mediano,
                            confianca=min(z_scores[idx] / 10, 1.0),  # Normalizar confiança
                            detalhes=f"Outlier estatístico em {nome_base}.{coluna} (Z-score: {z_scores[idx]:.2f})",
                            corrigivel_automaticamente=False
                        ))
        
        return inconsistencias
    
    
    def _is_padrao_suspeito_matricula(self, matricula: str) -> bool:
        """
        Verifica se uma matrícula tem padrão suspeito
        """
        mat = str(matricula).strip().lower()
        
        # Padrões suspeitos
        if len(mat) < 2:
            return True
        
        # Todos os caracteres iguais
        if len(set(mat)) == 1:
            return True
        
        # Sequência simples (123, abc, etc.)
        if len(mat) >= 3 and mat.isdigit():
            is_sequencia = True
            for i in range(1, len(mat)):
                if int(mat[i]) != int(mat[i-1]) + 1:
                    is_sequencia = False
                    break
            if is_sequencia:
                return True
        
        return False
    
    
    def _gerar_relatorio_fraudes(self, inconsistencias: List[InconsistenciaDetectada]) -> RelatorioFraudes:
        """
        Gera relatório consolidado de fraudes e inconsistências
        """
        total = len(inconsistencias)
        criticas = len([i for i in inconsistencias if i.gravidade == 'CRITICA'])
        corrigiveis = len([i for i in inconsistencias if i.corrigivel_automaticamente])
        
        # Calcular score de integridade (0-100)
        if total == 0:
            score_integridade = 100.0
        else:
            # Penalizar mais as críticas
            peso_criticas = criticas * 15
            peso_altas = len([i for i in inconsistencias if i.gravidade == 'ALTA']) * 5
            peso_outras = (total - criticas - peso_altas) * 1
            score_integridade = max(0, 100 - (peso_criticas + peso_altas + peso_outras))
        
        # Gerar recomendações
        recomendacoes = self._gerar_recomendacoes(inconsistencias)
        
        return RelatorioFraudes(
            total_inconsistencias=total,
            inconsistencias_criticas=criticas,
            inconsistencias_corrigidas=corrigiveis,
            score_integridade=score_integridade,
            detalhes=inconsistencias,
            recomendacoes=recomendacoes
        )
    
    
    def _gerar_recomendacoes(self, inconsistencias: List[InconsistenciaDetectada]) -> List[str]:
        """
        Gera recomendações baseadas nas inconsistências encontradas
        """
        recomendacoes = []
        tipos_encontrados = set(inc.tipo for inc in inconsistencias)
        
        if "ERRO_NOME_COLUNA" in tipos_encontrados:
            recomendacoes.append("📝 Padronizar nomes de colunas nos arquivos de entrada")
        
        if "OUTLIER_FINANCEIRO" in tipos_encontrados:
            recomendacoes.append("💰 Revisar valores de VR que estão fora do padrão normal")
        
        if "COLABORADOR_ATIVO_E_DESLIGADO" in tipos_encontrados:
            recomendacoes.append("👥 Verificar status de colaboradores duplicados entre bases")
        
        if "MATRICULA_DUPLICADA" in tipos_encontrados:
            recomendacoes.append("🔍 Implementar validação de unicidade de matrículas")
        
        if "CAMPO_OBRIGATORIO_VAZIO" in tipos_encontrados:
            recomendacoes.append("✅ Preencher campos obrigatórios antes do processamento")
        
        if "VALOR_SUSPEITO" in tipos_encontrados:
            recomendacoes.append("💵 Revisar valores zerados, negativos ou excessivamente altos")
        
        if len([i for i in inconsistencias if i.gravidade == 'CRITICA']) > 0:
            recomendacoes.append("🚨 URGENTE: Corrigir inconsistências críticas antes de continuar")
        
        return recomendacoes
    
    
    def aplicar_correcoes_automaticas(
        self, 
        bases_dados: Dict[str, pd.DataFrame],
        inconsistencias: List[InconsistenciaDetectada]
    ) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """
        Aplica correções automáticas nas inconsistências detectadas
        """
        bases_corrigidas = {nome: df.copy() for nome, df in bases_dados.items()}
        correcoes_aplicadas = []
        
        for inconsistencia in inconsistencias:
            if inconsistencia.corrigivel_automaticamente:
                try:
                    if inconsistencia.tipo == "VALOR_SUSPEITO":
                        bases_corrigidas = self._corrigir_valor_suspeito(bases_corrigidas, inconsistencia)
                        correcoes_aplicadas.append(f"✅ Valor corrigido para matrícula {inconsistencia.matricula}")
                    
                    elif inconsistencia.tipo == "MATRICULA_DUPLICADA":
                        bases_corrigidas = self._remover_duplicatas(bases_corrigidas, inconsistencia)
                        correcoes_aplicadas.append(f"✅ Duplicatas removidas para matrícula {inconsistencia.matricula}")
                    
                except Exception as e:
                    self.logger.error(f"Erro ao aplicar correção: {e}")
        
        return bases_corrigidas, correcoes_aplicadas
    
    
    def _corrigir_valor_suspeito(self, bases_dados: Dict[str, pd.DataFrame], inconsistencia: InconsistenciaDetectada) -> Dict[str, pd.DataFrame]:
        """Corrige valores suspeitos substituindo pela mediana"""
        for nome_base, df in bases_dados.items():
            if 'matricula' in df.columns and inconsistencia.campo_afetado in df.columns:
                mask = df['matricula'].astype(str) == str(inconsistencia.matricula)
                if mask.any():
                    df.loc[mask, inconsistencia.campo_afetado] = inconsistencia.valor_sugerido
                    break
        return bases_dados
    
    
    def _remover_duplicatas(self, bases_dados: Dict[str, pd.DataFrame], inconsistencia: InconsistenciaDetectada) -> Dict[str, pd.DataFrame]:
        """Remove duplicatas mantendo apenas o primeiro registro"""
        for nome_base, df in bases_dados.items():
            if 'matricula' in df.columns:
                df_sem_duplicatas = df.drop_duplicates(subset=['matricula'], keep='first')
                bases_dados[nome_base] = df_sem_duplicatas
        return bases_dados
    
    
    def gerar_dashboard_inconsistencias(self, relatorio: RelatorioFraudes, salvar_arquivo: bool = True) -> Optional[str]:
        """Gera dashboard visual das inconsistências usando Plotly"""
        try:
            # Preparar dados para visualização
            dados_gravidade = {'CRITICA': 0, 'ALTA': 0, 'MEDIA': 0, 'BAIXA': 0}
            dados_tipos = {}
            
            for inc in relatorio.detalhes:
                dados_gravidade[inc.gravidade] = dados_gravidade.get(inc.gravidade, 0) + 1
                dados_tipos[inc.tipo] = dados_tipos.get(inc.tipo, 0) + 1
            
            # Remover gravidades com valor 0
            dados_gravidade = {k: v for k, v in dados_gravidade.items() if v > 0}
            
            # Criar dashboard
            from plotly.subplots import make_subplots
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Inconsistências por Gravidade', 'Tipos de Inconsistências',
                              'Score de Integridade', 'Estatísticas'),
                specs=[[{"type": "pie"}, {"type": "bar"}],
                       [{"type": "indicator"}, {"type": "table"}]]
            )
            
            # Gráfico de pizza - Gravidade
            if dados_gravidade:
                fig.add_trace(
                    go.Pie(
                        labels=list(dados_gravidade.keys()),
                        values=list(dados_gravidade.values()),
                        name="Gravidade"
                    ),
                    row=1, col=1
                )
            
            # Gráfico de barras - Tipos
            if dados_tipos:
                fig.add_trace(
                    go.Bar(
                        x=list(dados_tipos.keys()),
                        y=list(dados_tipos.values()),
                        name="Tipos"
                    ),
                    row=1, col=2
                )
            
            # Indicador - Score de Integridade
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=relatorio.score_integridade,
                    title={'text': "Score de Integridade"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "red"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}
                        ]
                    }
                ),
                row=2, col=1
            )
            
            # Tabela de estatísticas
            fig.add_trace(
                go.Table(
                    header=dict(values=['Métrica', 'Valor']),
                    cells=dict(values=[
                        ['Total Inconsistências', 'Críticas', 'Corrigíveis', 'Score Integridade'],
                        [relatorio.total_inconsistencias, relatorio.inconsistencias_criticas, 
                         relatorio.inconsistencias_corrigidas, f"{relatorio.score_integridade:.1f}%"]
                    ])
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title="🔍 Dashboard de Análise de Inconsistências - Sistema VR",
                height=800
            )
            
            if salvar_arquivo:
                from pathlib import Path
                output_dir = Path("data/output")
                output_dir.mkdir(parents=True, exist_ok=True)
                arquivo_dashboard = output_dir / f"dashboard_inconsistencias_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                fig.write_html(arquivo_dashboard)
                self.logger.info(f"📊 Dashboard salvo: {arquivo_dashboard}")
                return str(arquivo_dashboard)
            
            return fig.to_html()
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar dashboard: {e}")
            return None


# Função utilitária para uso direto
def analisar_fraudes_vr(
    bases_dados: Dict[str, pd.DataFrame],
    calculos_vr: Optional[List] = None,
    aplicar_correcoes: bool = True
) -> Tuple[RelatorioFraudes, Dict[str, pd.DataFrame]]:
    """
    Função utilitária para análise completa de fraudes e inconsistências
    
    Args:
        bases_dados: Dicionário com as bases de dados
        calculos_vr: Lista de cálculos VR (opcional)
        aplicar_correcoes: Se deve aplicar correções automáticas
    
    Returns:
        Tupla com (RelatorioFraudes, bases_dados_corrigidas)
    """
    # Inicializar sistema de análise
    analyzer = AIAnalyticsVR()
    
    # Detectar inconsistências
    relatorio, bases_corrigidas = analyzer.detectar_fraudes_inconsistencias(bases_dados, calculos_vr)
    
    # Aplicar correções se solicitado
    if aplicar_correcoes:
        bases_corrigidas, correcoes = analyzer.aplicar_correcoes_automaticas(
            bases_corrigidas, relatorio.detalhes
        )
        if correcoes:
            print("🔧 Correções aplicadas automaticamente:")
            for correcao in correcoes:
                print(f"  {correcao}")
    
    # Gerar dashboard
    analyzer.gerar_dashboard_inconsistencias(relatorio)
    
    return relatorio, bases_corrigidas


# Exemplo de uso
if __name__ == "__main__":
    import logging
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🤖 Testando AI Analytics VR...")
    
    # Dados de teste com erros propositais
    dados_teste = {
        'ativos': pd.DataFrame({
            'matricla': ['123', '456', '789', '123'],  # Erro proposital no nome + duplicata
            'nome': ['João', 'Maria', 'Pedro', 'João'],
            'valor_diario': [50.0, 0.0, 500.0, 50.0]  # Valor suspeito (0 e 500)
        }),
        'desligados': pd.DataFrame({
            'matricula': ['123', '999'],  # Duplicata com ativos
            'nome': ['João', 'Ana'],
            'data_demissao': ['2025-01-15', '2025-02-20']
        })
    }
    
    # Analisar
    relatorio, dados_corrigidos = analisar_fraudes_vr(
        dados_teste, 
        aplicar_correcoes=True
    )
    
    # Mostrar resultados
    print(f"\n📊 RELATÓRIO DE INCONSISTÊNCIAS:")
    print(f"   Total: {relatorio.total_inconsistencias}")
    print(f"   Críticas: {relatorio.inconsistencias_criticas}")
    print(f"   Corrigíveis: {relatorio.inconsistencias_corrigidas}")
    print(f"   Score de Integridade: {relatorio.score_integridade:.1f}%")
    
    print(f"\n🔍 DETALHES DAS INCONSISTÊNCIAS:")
    for inc in relatorio.detalhes[:5]:  # Mostrar apenas as 5 primeiras
        print(f"   {inc.tipo}: {inc.detalhes}")
    
    print(f"\n💡 RECOMENDAÇÕES:")
    for rec in relatorio.recomendacoes:
        print(f"   {rec}")
    
    print(f"\n📋 COLUNAS CORRIGIDAS:")
    for nome_base in dados_corrigidos:
        print(f"   {nome_base}: {list(dados_corrigidos[nome_base].columns)}")
