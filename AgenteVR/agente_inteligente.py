"""
Módulo agente_inteligente.py
Sistema de Agente Inteligente com LLMs gratuitas para análise de VR.

Autor: Agente VR IA 2.0
Data: 2025-08-26
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np

# LLMs Gratuitas
try:
    import ollama  # Ollama local
    OLLAMA_DISPONIVEL = True
except ImportError:
    OLLAMA_DISPONIVEL = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_DISPONIVEL = True
except ImportError:
    TRANSFORMERS_DISPONIVEL = False

try:
    import google.generativeai as genai  # Gemini gratuito
    GEMINI_DISPONIVEL = True
except ImportError:
    GEMINI_DISPONIVEL = False

try:
    from groq import Groq  # API gratuita
    GROQ_DISPONIVEL = True
except ImportError:
    GROQ_DISPONIVEL = False

# LangChain para orquestração
try:
    from langchain.llms import Ollama
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.memory import ConversationBufferMemory
    from langchain.agents import initialize_agent, Tool, AgentType
    LANGCHAIN_DISPONIVEL = True
except ImportError:
    LANGCHAIN_DISPONIVEL = False

# Módulos internos
from modules.ai_analytics import AIAnalyticsVR, RelatorioFraudes, InconsistenciaDetectada


@dataclass
class RespostaAgente:
    """Resposta estruturada do agente inteligente"""
    tipo_resposta: str  # 'analise', 'recomendacao', 'correcao', 'insight'
    conteudo: str
    dados_estruturados: Optional[Dict[str, Any]] = None
    confianca: float = 0.0
    acoes_sugeridas: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class AgenteInteligenteVR:
    """
    Agente Inteligente para análise de dados VR usando LLMs gratuitas
    """
    
    def __init__(self, modelo_preferido: str = "ollama", api_keys: Optional[Dict[str, str]] = None):
        """
        Inicializa o agente inteligente
        
        Args:
            modelo_preferido: 'ollama', 'transformers', 'gemini', 'groq'
            api_keys: Dicionário com chaves API opcionais
        """
        self.logger = logging.getLogger(__name__)
        self.modelo_preferido = modelo_preferido
        self.api_keys = api_keys or {}
        
        # Configurar LLM baseado na preferência
        self.llm = None
        self.llm_ativo = None
        self._configurar_llm()
        
        # Integração com analytics existente
        self.analytics = AIAnalyticsVR()
        
        # Memória de conversação
        self.memoria = []
        
        # Prompt templates especializados
        self._configurar_prompts()
        
        # Ferramentas disponíveis para o agente
        self._configurar_ferramentas()
        
        self.logger.info(f"🤖 Agente Inteligente inicializado com {self.llm_ativo}")
    
    def _configurar_llm(self):
        """Configura o LLM baseado na preferência e disponibilidade"""
        
        # Tentar Ollama primeiro (recomendado - gratuito e local)
        if self.modelo_preferido == "ollama" and OLLAMA_DISPONIVEL:
            try:
                # Verificar se Ollama está rodando
                models = ollama.list()
                if models:
                    self.llm = ollama
                    self.llm_ativo = "ollama"
                    self.logger.info("✅ Ollama configurado com sucesso")
                    return
            except Exception as e:
                self.logger.warning(f"⚠️ Ollama não disponível: {e}")
        
        # Tentar Gemini (gratuito com limites)
        if self.modelo_preferido == "gemini" and GEMINI_DISPONIVEL:
            api_key = self.api_keys.get('gemini') or os.getenv('GEMINI_API_KEY')
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    self.llm = genai.GenerativeModel('gemini-1.5-flash')
                    self.llm_ativo = "gemini"
                    self.logger.info("✅ Gemini configurado com sucesso")
                    return
                except Exception as e:
                    self.logger.warning(f"⚠️ Gemini não disponível: {e}")
        
        # Tentar Groq (API gratuita)
        if self.modelo_preferido == "groq" and GROQ_DISPONIVEL:
            api_key = self.api_keys.get('groq') or os.getenv('GROQ_API_KEY')
            if api_key:
                try:
                    self.llm = Groq(api_key=api_key)
                    self.llm_ativo = "groq"
                    self.logger.info("✅ Groq configurado com sucesso")
                    return
                except Exception as e:
                    self.logger.warning(f"⚠️ Groq não disponível: {e}")
        
        # Fallback: Transformers local (sem internet)
        if TRANSFORMERS_DISPONIVEL:
            try:
                # Usar modelo pequeno para economizar recursos
                self.llm = pipeline(
                    "text-generation",
                    model="microsoft/DialoGPT-small",
                    device=-1  # CPU
                )
                self.llm_ativo = "transformers"
                self.logger.info("✅ Transformers configurado como fallback")
                return
            except Exception as e:
                self.logger.warning(f"⚠️ Transformers não disponível: {e}")
        
        # Sem LLM disponível
        self.llm_ativo = "nenhum"
        self.logger.error("❌ Nenhum LLM disponível! Usando modo analítico apenas.")
    
    def _configurar_prompts(self):
        """Configura templates de prompts especializados"""
        
        self.prompts = {
            "analise_dados": """
            Você é um especialista em análise de dados de Vale Refeição corporativo.
            
            DADOS FORNECIDOS:
            {dados_resumo}
            
            INCONSISTÊNCIAS DETECTADAS:
            {inconsistencias}
            
            Por favor, analise os dados e forneça:
            1. Diagnóstico principal dos problemas
            2. Possíveis causas raiz
            3. Impacto nos custos
            4. Recomendações prioritárias
            5. Ações preventivas
            
            Seja conciso e técnico. Foque em insights acionáveis.
            """,
            
            "deteccao_fraudes": """
            Você é um auditor especializado em detecção de fraudes em sistemas de benefícios.
            
            PADRÕES SUSPEITOS ENCONTRADOS:
            {padroes_suspeitos}
            
            DADOS FINANCEIROS:
            {dados_financeiros}
            
            Analise se há indicadores de fraude e classifique o risco:
            1. BAIXO: Inconsistências normais
            2. MÉDIO: Padrões que requerem atenção
            3. ALTO: Possível fraude, investigar imediatamente
            
            Para cada caso suspeito, explique o motivo da classificação.
            """,
            
            "otimizacao_custos": """
            Você é um consultor em otimização de custos corporativos.
            
            DADOS DE CUSTOS VR:
            {dados_custos}
            
            DISTRIBUIÇÃO POR REGIÃO:
            {distribuicao_regional}
            
            Identifique oportunidades de otimização:
            1. Redução de custos sem impacto nos benefícios
            2. Renegociação com fornecedores
            3. Ajustes em políticas internas
            4. Economia potencial estimada
            """,
            
            "correcao_dados": """
            Você é um especialista em qualidade de dados.
            
            DADOS INCONSISTENTES:
            {dados_inconsistentes}
            
            Para cada inconsistência, sugira:
            1. Correção automática (se possível)
            2. Validação manual necessária
            3. Regras para prevenir recorrência
            4. Script de correção (se aplicável)
            
            Priorize correções que podem ser automatizadas.
            """
        }
    
    def _configurar_ferramentas(self):
        """Configura ferramentas disponíveis para o agente"""
        
        self.ferramentas = {
            "analisar_inconsistencias": self._ferramenta_analisar_inconsistencias,
            "detectar_outliers": self._ferramenta_detectar_outliers,
            "corrigir_dados": self._ferramenta_corrigir_dados,
            "gerar_relatorio": self._ferramenta_gerar_relatorio,
            "calcular_metricas": self._ferramenta_calcular_metricas,
            "sugerir_otimizacoes": self._ferramenta_sugerir_otimizacoes
        }
    
    def processar_solicitacao(self, solicitacao: str, dados: Optional[Dict[str, Any]] = None) -> RespostaAgente:
        """
        Processa uma solicitação do usuário usando o agente inteligente
        
        Args:
            solicitacao: Pergunta ou comando do usuário
            dados: Dados contextuais opcionais
            
        Returns:
            RespostaAgente com análise e recomendações
        """
        self.logger.info(f"🤖 Processando: {solicitacao[:100]}...")
        
        # Adicionar à memória
        self.memoria.append({
            "timestamp": datetime.now(),
            "tipo": "solicitacao",
            "conteudo": solicitacao,
            "dados": dados
        })
        
        # Classificar tipo de solicitação
        tipo_solicitacao = self._classificar_solicitacao(solicitacao)
        
        # Processar baseado no tipo
        if tipo_solicitacao == "analise_dados":
            return self._processar_analise_dados(solicitacao, dados)
        elif tipo_solicitacao == "deteccao_fraudes":
            return self._processar_deteccao_fraudes(solicitacao, dados)
        elif tipo_solicitacao == "otimizacao_custos":
            return self._processar_otimizacao_custos(solicitacao, dados)
        elif tipo_solicitacao == "correcao_dados":
            return self._processar_correcao_dados(solicitacao, dados)
        else:
            return self._processar_consulta_geral(solicitacao, dados)
    
    def _classificar_solicitacao(self, solicitacao: str) -> str:
        """Classifica o tipo de solicitação baseado em palavras-chave"""
        
        solicitacao_lower = solicitacao.lower()
        
        # Palavras-chave para cada categoria
        keywords = {
            "analise_dados": ["analis", "dados", "estatistic", "resumo", "overview", "métricas"],
            "deteccao_fraudes": ["fraud", "suspeito", "irregular", "anomal", "outlier", "inconsistenc"],
            "otimizacao_custos": ["custo", "econom", "otimiz", "reduz", "valor", "gasto"],
            "correcao_dados": ["corrig", "ajust", "fix", "erro", "problem", "clean"]
        }
        
        scores = {}
        for categoria, palavras in keywords.items():
            scores[categoria] = sum(1 for palavra in palavras if palavra in solicitacao_lower)
        
        # Retornar categoria com maior score
        return max(scores, key=scores.get) if max(scores.values()) > 0 else "consulta_geral"
    
    def _processar_analise_dados(self, solicitacao: str, dados: Optional[Dict[str, Any]]) -> RespostaAgente:
        """Processa solicitações de análise de dados"""
        
        if not dados:
            return RespostaAgente(
                tipo_resposta="erro",
                conteudo="❌ Dados não fornecidos para análise",
                confianca=0.0
            )
        
        # Usar analytics para gerar insights
        relatorio_fraudes = self.analytics.detectar_fraudes_inconsistencias(dados)
        
        # Preparar dados para LLM
        dados_resumo = self._preparar_resumo_dados(dados)
        inconsistencias = self._formatar_inconsistencias(relatorio_fraudes.detalhes)
        
        # Gerar análise com LLM
        analise_llm = self._chamar_llm(
            self.prompts["analise_dados"].format(
                dados_resumo=dados_resumo,
                inconsistencias=inconsistencias
            )
        )
        
        return RespostaAgente(
            tipo_resposta="analise",
            conteudo=analise_llm,
            dados_estruturados={
                "total_inconsistencias": relatorio_fraudes.total_inconsistencias,
                "score_integridade": relatorio_fraudes.score_integridade,
                "recomendacoes": relatorio_fraudes.recomendacoes
            },
            confianca=0.85,
            acoes_sugeridas=[
                "Revisar inconsistências críticas",
                "Aplicar correções automáticas",
                "Implementar validações preventivas"
            ]
        )
    
    def _processar_deteccao_fraudes(self, solicitacao: str, dados: Optional[Dict[str, Any]]) -> RespostaAgente:
        """Processa solicitações de detecção de fraudes"""
        
        if not dados:
            return RespostaAgente(
                tipo_resposta="erro",
                conteudo="❌ Dados não fornecidos para análise de fraudes",
                confianca=0.0
            )
        
        # Detectar padrões suspeitos
        relatorio = self.analytics.detectar_fraudes_inconsistencias(dados)
        
        # Analisar com LLM
        padroes_suspeitos = self._extrair_padroes_suspeitos(relatorio.detalhes)
        dados_financeiros = self._extrair_dados_financeiros(dados)
        
        analise_fraude = self._chamar_llm(
            self.prompts["deteccao_fraudes"].format(
                padroes_suspeitos=padroes_suspeitos,
                dados_financeiros=dados_financeiros
            )
        )
        
        # Classificar risco
        risco = self._classificar_risco_fraude(relatorio.detalhes)
        
        return RespostaAgente(
            tipo_resposta="deteccao_fraudes",
            conteudo=analise_fraude,
            dados_estruturados={
                "nivel_risco": risco,
                "casos_suspeitos": len([i for i in relatorio.detalhes if i.gravidade == "CRITICA"]),
                "valor_em_risco": self._calcular_valor_em_risco(dados, relatorio.detalhes)
            },
            confianca=0.90,
            acoes_sugeridas=[
                "Investigar casos de risco alto",
                "Implementar controles adicionais",
                "Revisar processos de aprovação"
            ]
        )
    
    def _processar_otimizacao_custos(self, solicitacao: str, dados: Optional[Dict[str, Any]]) -> RespostaAgente:
        """Processa solicitações de otimização de custos"""
        
        if not dados:
            return RespostaAgente(
                tipo_resposta="erro",
                conteudo="❌ Dados não fornecidos para análise de custos",
                confianca=0.0
            )
        
        # Calcular métricas de custos
        metricas = self._ferramenta_calcular_metricas(dados)
        
        # Preparar dados para LLM
        dados_custos = self._extrair_dados_financeiros(dados)
        distribuicao_regional = self._calcular_distribuicao_regional(dados)
        
        analise_custos = self._chamar_llm(
            self.prompts["otimizacao_custos"].format(
                dados_custos=dados_custos,
                distribuicao_regional=distribuicao_regional
            )
        )
        
        return RespostaAgente(
            tipo_resposta="otimizacao_custos",
            conteudo=analise_custos,
            dados_estruturados=metricas,
            confianca=0.80,
            acoes_sugeridas=[
                "Revisar valores outliers",
                "Renegociar contratos",
                "Otimizar distribuição regional"
            ]
        )
    
    def _processar_correcao_dados(self, solicitacao: str, dados: Optional[Dict[str, Any]]) -> RespostaAgente:
        """Processa solicitações de correção de dados"""
        
        if not dados:
            return RespostaAgente(
                tipo_resposta="erro",
                conteudo="❌ Dados não fornecidos para correção",
                confianca=0.0
            )
        
        # Identificar problemas
        correcoes = self._ferramenta_corrigir_dados(dados)
        
        # Preparar dados para LLM
        dados_inconsistentes = self._identificar_inconsistencias(dados)
        
        analise_correcao = self._chamar_llm(
            self.prompts["correcao_dados"].format(
                dados_inconsistentes=dados_inconsistentes
            )
        )
        
        return RespostaAgente(
            tipo_resposta="correcao_dados",
            conteudo=analise_correcao,
            dados_estruturados=correcoes,
            confianca=0.85,
            acoes_sugeridas=[
                "Aplicar correções automáticas",
                "Validar dados manualmente",
                "Implementar validações preventivas"
            ]
        )
    
    def _processar_consulta_geral(self, solicitacao: str, dados: Optional[Dict[str, Any]]) -> RespostaAgente:
        """Processa consultas gerais"""
        
        # Preparar contexto
        contexto = "Consulta geral sobre dados de VR."
        if dados:
            contexto += f" Dados disponíveis: {list(dados.keys())}"
        
        # Gerar resposta
        resposta_llm = self._chamar_llm(f"{contexto}\n\nPergunta: {solicitacao}")
        
        return RespostaAgente(
            tipo_resposta="consulta_geral",
            conteudo=resposta_llm,
            confianca=0.70,
            acoes_sugeridas=["Revisar dados", "Fazer análise específica"]
        )
    
    def _chamar_llm(self, prompt: str) -> str:
        """Chama o LLM configurado com o prompt"""
        
        if self.llm_ativo == "nenhum":
            return "⚠️ LLM não disponível. Usando análise estatística apenas."
        
        try:
            if self.llm_ativo == "ollama":
                response = ollama.generate(
                    model="llama3.1:8b",  # Modelo recomendado
                    prompt=prompt
                )
                return response['response']
            
            elif self.llm_ativo == "gemini":
                response = self.llm.generate_content(prompt)
                return response.text
            
            elif self.llm_ativo == "groq":
                response = self.llm.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama3-8b-8192",
                    max_tokens=1000
                )
                return response.choices[0].message.content
            
            elif self.llm_ativo == "transformers":
                # Modelo local mais simples
                response = self.llm(prompt, max_length=500, num_return_sequences=1)
                return response[0]['generated_text']
            
        except Exception as e:
            self.logger.error(f"Erro ao chamar LLM: {e}")
            return f"⚠️ Erro na análise LLM: {str(e)}"
        
        return "⚠️ LLM não configurado corretamente."
    
    # Métodos auxiliares
    def _preparar_resumo_dados(self, dados: Dict[str, Any]) -> str:
        """Prepara resumo dos dados para o LLM"""
        resumo = []
        
        for nome_base, df in dados.items():
            if isinstance(df, pd.DataFrame):
                resumo.append(f"{nome_base}: {len(df)} registros, {len(df.columns)} colunas")
                
                # Valores nulos
                nulos = df.isnull().sum().sum()
                if nulos > 0:
                    resumo.append(f"  - {nulos} valores nulos")
                
                # Duplicatas
                if 'matricula' in df.columns:
                    duplicatas = df.duplicated(subset=['matricula']).sum()
                    if duplicatas > 0:
                        resumo.append(f"  - {duplicatas} matrículas duplicadas")
        
        return "\n".join(resumo)
    
    def _formatar_inconsistencias(self, inconsistencias: List[InconsistenciaDetectada]) -> str:
        """Formata inconsistências para o LLM"""
        if not inconsistencias:
            return "Nenhuma inconsistência detectada."
        
        resumo = []
        for inc in inconsistencias[:10]:  # Limitar para não sobrecarregar
            resumo.append(f"- {inc.tipo}: {inc.detalhes} (Gravidade: {inc.gravidade})")
        
        if len(inconsistencias) > 10:
            resumo.append(f"... e mais {len(inconsistencias) - 10} inconsistências")
        
        return "\n".join(resumo)
    
    def _extrair_padroes_suspeitos(self, inconsistencias: List[InconsistenciaDetectada]) -> str:
        """Extrai padrões suspeitos das inconsistências"""
        suspeitos = [i for i in inconsistencias if i.gravidade in ["CRITICA", "ALTA"]]
        
        if not suspeitos:
            return "Nenhum padrão suspeito detectado."
        
        padroes = []
        for inc in suspeitos:
            padroes.append(f"- Matrícula {inc.matricula}: {inc.tipo} - {inc.detalhes}")
        
        return "\n".join(padroes)
    
    def _classificar_risco_fraude(self, inconsistencias: List[InconsistenciaDetectada]) -> str:
        """Classifica o nível de risco de fraude"""
        criticas = len([i for i in inconsistencias if i.gravidade == "CRITICA"])
        altas = len([i for i in inconsistencias if i.gravidade == "ALTA"])
        
        if criticas > 0:
            return "ALTO"
        elif altas > 2:
            return "MÉDIO"
        else:
            return "BAIXO"
    
    def _extrair_dados_financeiros(self, dados: Dict[str, Any]) -> str:
        """Extrai dados financeiros para análise"""
        financeiros = []
        for nome, df in dados.items():
            if isinstance(df, pd.DataFrame) and 'valor_vr' in df.columns:
                total = df['valor_vr'].sum()
                media = df['valor_vr'].mean()
                financeiros.append(f"{nome}: Total R$ {total:,.2f}, Média R$ {media:.2f}")
        
        return "\n".join(financeiros) if financeiros else "Nenhum dado financeiro encontrado"
    
    def _calcular_valor_em_risco(self, dados: Dict[str, Any], inconsistencias: List[InconsistenciaDetectada]) -> float:
        """Calcula valor total em risco devido a inconsistências"""
        valor_risco = 0.0
        
        for nome, df in dados.items():
            if isinstance(df, pd.DataFrame) and 'valor_vr' in df.columns:
                # Estimar risco baseado em inconsistências críticas
                criticas = len([i for i in inconsistencias if i.gravidade == "CRITICA"])
                if criticas > 0:
                    # Assumir que 10% do valor está em risco por inconsistência crítica
                    valor_risco += df['valor_vr'].sum() * (criticas * 0.1)
        
        return valor_risco
    
    def _calcular_distribuicao_regional(self, dados: Dict[str, Any]) -> str:
        """Calcula distribuição regional dos dados"""
        distribuicao = []
        
        for nome, df in dados.items():
            if isinstance(df, pd.DataFrame):
                if 'departamento' in df.columns:
                    dept_counts = df['departamento'].value_counts()
                    for dept, count in dept_counts.head(5).items():
                        distribuicao.append(f"{nome} - {dept}: {count} funcionários")
        
        return "\n".join(distribuicao) if distribuicao else "Distribuição não disponível"
    
    def _identificar_inconsistencias(self, dados: Dict[str, Any]) -> str:
        """Identifica inconsistências nos dados"""
        inconsistencias = []
        
        for nome, df in dados.items():
            if isinstance(df, pd.DataFrame):
                # Verificar valores nulos
                nulos = df.isnull().sum()
                for col, count in nulos.items():
                    if count > 0:
                        inconsistencias.append(f"{nome}.{col}: {count} valores nulos")
                
                # Verificar duplicatas
                if 'matricula' in df.columns:
                    dups = df.duplicated(subset=['matricula']).sum()
                    if dups > 0:
                        inconsistencias.append(f"{nome}: {dups} matrículas duplicadas")
                
                # Verificar valores extremos
                if 'valor_vr' in df.columns:
                    media = df['valor_vr'].mean()
                    extremos = len(df[df['valor_vr'] > media * 3])
                    if extremos > 0:
                        inconsistencias.append(f"{nome}: {extremos} valores VR extremos")
        
        return "\n".join(inconsistencias) if inconsistencias else "Nenhuma inconsistência detectada"
    
    # Ferramentas do agente
    def _ferramenta_analisar_inconsistencias(self, dados: Dict[str, Any]) -> Dict[str, Any]:
        """Ferramenta para analisar inconsistências"""
        relatorio = self.analytics.detectar_fraudes_inconsistencias(dados)
        return {
            "total": relatorio.total_inconsistencias,
            "criticas": relatorio.inconsistencias_criticas,
            "score": relatorio.score_integridade
        }
    
    def _ferramenta_detectar_outliers(self, dados: Dict[str, Any]) -> Dict[str, Any]:
        """Ferramenta para detectar outliers"""
        outliers = {}
        for nome, df in dados.items():
            if isinstance(df, pd.DataFrame):
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers[f"{nome}_{col}"] = len(df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)])
        return outliers
    
    def _ferramenta_corrigir_dados(self, dados: Dict[str, Any]) -> Dict[str, Any]:
        """Ferramenta para sugerir correções"""
        correcoes = []
        for nome, df in dados.items():
            if isinstance(df, pd.DataFrame):
                # Valores nulos
                nulos = df.isnull().sum()
                if nulos.sum() > 0:
                    correcoes.append(f"{nome}: {nulos.sum()} valores nulos encontrados")
                
                # Duplicatas
                if 'matricula' in df.columns:
                    dups = df.duplicated(subset=['matricula']).sum()
                    if dups > 0:
                        correcoes.append(f"{nome}: {dups} duplicatas em matricula")
        
        return {"correcoes_sugeridas": correcoes}
    
    def _ferramenta_gerar_relatorio(self, dados: Dict[str, Any]) -> Dict[str, Any]:
        """Ferramenta para gerar relatório"""
        relatorio = {}
        for nome, df in dados.items():
            if isinstance(df, pd.DataFrame):
                relatorio[nome] = {
                    "registros": len(df),
                    "colunas": len(df.columns),
                    "valores_nulos": df.isnull().sum().sum(),
                    "memoria_mb": df.memory_usage(deep=True).sum() / 1024**2
                }
        return relatorio
    
    def _ferramenta_calcular_metricas(self, dados: Dict[str, Any]) -> Dict[str, Any]:
        """Ferramenta para calcular métricas"""
        metricas = {}
        for nome, df in dados.items():
            if isinstance(df, pd.DataFrame):
                if 'valor_vr' in df.columns:
                    metricas[f"{nome}_valor_total"] = df['valor_vr'].sum()
                    metricas[f"{nome}_valor_medio"] = df['valor_vr'].mean()
                    metricas[f"{nome}_valor_max"] = df['valor_vr'].max()
        return metricas
    
    def _ferramenta_sugerir_otimizacoes(self, dados: Dict[str, Any]) -> Dict[str, Any]:
        """Ferramenta para sugerir otimizações"""
        otimizacoes = []
        for nome, df in dados.items():
            if isinstance(df, pd.DataFrame) and 'valor_vr' in df.columns:
                media = df['valor_vr'].mean()
                outliers_altos = len(df[df['valor_vr'] > media * 2])
                if outliers_altos > 0:
                    otimizacoes.append(f"{nome}: {outliers_altos} valores muito altos detectados")
        
        return {"otimizacoes": otimizacoes}
    
    def listar_modelos_disponiveis(self) -> Dict[str, bool]:
        """Lista modelos LLM disponíveis no sistema"""
        return {
            "ollama": OLLAMA_DISPONIVEL,
            "transformers": TRANSFORMERS_DISPONIVEL,
            "gemini": GEMINI_DISPONIVEL,
            "groq": GROQ_DISPONIVEL,
            "langchain": LANGCHAIN_DISPONIVEL
        }
    
    def alternar_modelo(self, novo_modelo: str) -> bool:
        """Alterna para um novo modelo LLM"""
        self.modelo_preferido = novo_modelo
        self._configurar_llm()
        return self.llm_ativo == novo_modelo


# Função utilitária para criar agente rapidamente
def criar_agente_vr(
    modelo: str = "ollama",
    api_keys: Optional[Dict[str, str]] = None
) -> AgenteInteligenteVR:
    """
    Cria um agente inteligente VR configurado
    
    Args:
        modelo: 'ollama', 'gemini', 'groq', 'transformers'
        api_keys: Chaves API opcionais
    
    Returns:
        AgenteInteligenteVR configurado
    """
    return AgenteInteligenteVR(modelo, api_keys)


# Exemplo de uso
if __name__ == "__main__":
    import logging
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Criar agente
    agente = criar_agente_vr("ollama")
    
    # Verificar modelos disponíveis
    modelos = agente.listar_modelos_disponiveis()
    print("🤖 Modelos LLM Disponíveis:")
    for modelo, disponivel in modelos.items():
        status = "✅" if disponivel else "❌"
        print(f"   {status} {modelo}")
    
    print(f"\n🎯 Agente ativo com: {agente.llm_ativo}")
    
    # Teste básico
    resposta = agente.processar_solicitacao(
        "Analise os dados de VR e identifique possíveis problemas",
        {"teste": "dados_exemplo"}
    )
    
    print(f"\n📋 Resposta do Agente:")
    print(f"Tipo: {resposta.tipo_resposta}")
    print(f"Conteúdo: {resposta.conteudo}")
    print(f"Confiança: {resposta.confianca:.2%}")
