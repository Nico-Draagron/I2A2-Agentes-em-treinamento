# Classe stub para evitar erro de importação
class VRAgent:
    def __init__(self):
        pass
"""
Assistente Inteligente com Google Gemini API
Utiliza IA generativa para análise e correção de dados VR
"""

import os
from config.settings import GEMINI_API_KEY
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import google.generativeai as genai
from dataclasses import dataclass
import logging


@dataclass
class AnaliseIA:
    """Resultado de análise da IA"""
    tipo: str  # 'validacao', 'correcao', 'anomalia', 'sugestao'
    confianca: float  # 0-1
    mensagem: str
    detalhes: Dict[str, Any]
    acoes_sugeridas: List[str]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class GeminiAssistant:
    """
    Assistente inteligente usando Google Gemini API
    Versão gratuita com rate limiting
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa o assistente Gemini
        
        Args:
            api_key: Chave da API do Google Gemini
        """
        self.logger = logging.getLogger(__name__)
        
        # Configurar API
        self.api_key = api_key or GEMINI_API_KEY or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("❌ GEMINI_API_KEY não configurada!")
        
        # Configurar Gemini
        genai.configure(api_key=self.api_key)
        
        # Modelo gratuito: gemini-1.5-flash
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Configurações de segurança
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
        
        # Rate limiting (versão gratuita tem limites)
        self.requests_per_minute = 15  # Limite conservador
        self.last_request_time = 0
        self.request_count = 0
        
        # Cache de respostas para economizar requisições
        self.cache = {}
        
        # Prompts especializados
        self._configurar_prompts()
        
        self.logger.info("🤖 Gemini Assistant inicializado com sucesso!")
    
    def _configurar_prompts(self):
        """Configura prompts especializados para VR"""
        
        self.prompts = {
            "analise_dados": """
            Você é um especialista em análise de dados de RH e benefícios corporativos.
            
            Analise os seguintes dados de Vale Refeição (VR) e identifique:
            1. Inconsistências ou anomalias
            2. Dados faltantes críticos
            3. Valores suspeitos ou fora do padrão
            4. Problemas de formatação
            5. Sugestões de correção
            
            Dados para análise:
            {dados}
            
            Responda em formato JSON estruturado:
            {{
                "inconsistencias": [...],
                "dados_faltantes": [...],
                "valores_suspeitos": [...],
                "problemas_formatacao": [...],
                "sugestoes_correcao": [...],
                "score_qualidade": 0-100,
                "requer_atencao_urgente": true/false
            }}
            """,
            
            "validar_matricula": """
            Analise as seguintes matrículas e identifique possíveis problemas:
            
            Matrículas: {matriculas}
            
            Verifique:
            - Duplicatas
            - Formatos inconsistentes
            - Valores inválidos (muito altos, muito baixos, sequenciais suspeitos)
            - Padrões anormais
            
            Responda em JSON com matrículas problemáticas e o motivo.
            """,
            
            "corrigir_datas": """
            Corrija as seguintes datas inconsistentes no contexto de RH:
            
            Datas problemáticas:
            {datas}
            
            Contexto: {contexto}
            
            Para cada data, sugira:
            1. Data corrigida mais provável
            2. Nível de confiança (0-100%)
            3. Justificativa da correção
            
            Responda em JSON.
            """,
            
            "detectar_fraudes": """
            Como auditor especializado, analise os seguintes dados de VR para detectar possíveis fraudes:
            
            Dados suspeitos:
            {dados}
            
            Procure por:
            - Valores muito acima da média
            - Padrões repetitivos suspeitos
            - Colaboradores com múltiplos registros
            - Valores que não correspondem ao sindicato
            - Datas manipuladas
            
            Classifique cada caso:
            - BAIXO: Provavelmente erro de digitação
            - MÉDIO: Requer verificação manual
            - ALTO: Possível fraude, investigar imediatamente
            
            Responda em JSON com casos suspeitos e classificação.
            """,
            
            "otimizar_processo": """
            Como consultor de processos, analise o processamento de VR e sugira melhorias:
            
            Estatísticas atuais:
            {estatisticas}
            
            Problemas recorrentes:
            {problemas}
            
            Sugira:
            1. Melhorias no processo
            2. Validações adicionais necessárias
            3. Automações possíveis
            4. Estimativa de economia/eficiência
            
            Responda de forma estruturada e prática.
            """,
            
            "explicar_regra": """
            Explique de forma clara e didática a seguinte regra de VR:
            
            Regra: {regra}
            Contexto: {contexto}
            
            Inclua:
            1. Explicação simples
            2. Exemplos práticos
            3. Casos especiais
            4. Impacto financeiro
            """,
            
            "resumo_executivo": """
            Crie um resumo executivo do processamento de VR:
            
            Dados processados:
            {dados}
            
            O resumo deve conter:
            1. Visão geral (2-3 linhas)
            2. Números principais
            3. Problemas críticos encontrados
            4. Ações recomendadas
            5. Próximos passos
            
            Seja conciso e direto ao ponto.
            """
        }
    
    def _rate_limit(self):
        """Implementa rate limiting para API gratuita"""
        current_time = time.time()
        
        # Reset contador a cada minuto
        if current_time - self.last_request_time > 60:
            self.request_count = 0
            self.last_request_time = current_time
        
        # Verificar limite
        if self.request_count >= self.requests_per_minute:
            wait_time = 60 - (current_time - self.last_request_time)
            if wait_time > 0:
                self.logger.warning(f"⏱️ Rate limit atingido. Aguardando {wait_time:.1f}s...")
                time.sleep(wait_time)
                self.request_count = 0
                self.last_request_time = time.time()
        
        self.request_count += 1
    
    def _chamar_gemini(self, prompt: str, use_cache: bool = True) -> Optional[str]:
        """
        Chama a API do Gemini com tratamento de erros
        
        Args:
            prompt: Prompt para enviar
            use_cache: Se deve usar cache
            
        Returns:
            Resposta do modelo ou None se erro
        """
        # Verificar cache
        if use_cache and prompt in self.cache:
            self.logger.debug("📦 Usando resposta do cache")
            return self.cache[prompt]
        
        try:
            # Rate limiting
            self._rate_limit()
            
            # Gerar resposta
            response = self.model.generate_content(
                prompt,
                safety_settings=self.safety_settings,
                generation_config={
                    'temperature': 0.3,  # Baixa para respostas consistentes
                    'top_p': 0.95,
                    'top_k': 40,
                    'max_output_tokens': 2048,
                }
            )
            
            # Extrair texto
            if response.text:
                # Adicionar ao cache
                if use_cache:
                    self.cache[prompt] = response.text
                
                return response.text
            else:
                self.logger.warning("⚠️ Resposta vazia do Gemini")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Erro ao chamar Gemini: {e}")
            
            # Se for erro de rate limit, aguardar
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                self.logger.warning("⏱️ Aguardando 60s por rate limit...")
                time.sleep(60)
            
            return None
    
    def analisar_dados_entrada(self, dados: Dict[str, pd.DataFrame]) -> AnaliseIA:
        """
        Analisa dados de entrada com IA
        
        Args:
            dados: Dicionário com DataFrames
            
        Returns:
            Análise completa dos dados
        """
        self.logger.info("🔍 Analisando dados com Gemini...")
        
        # Preparar resumo dos dados
        resumo_dados = self._preparar_resumo_dados(dados)
        
        # Montar prompt
        prompt = self.prompts["analise_dados"].format(dados=resumo_dados)
        
        # Chamar Gemini
        resposta = self._chamar_gemini(prompt)
        
        if not resposta:
            return self._analise_fallback(dados)
        
        # Processar resposta
        try:
            # Tentar extrair JSON da resposta
            resultado = self._extrair_json(resposta)
            
            return AnaliseIA(
                tipo='validacao',
                confianca=resultado.get('score_qualidade', 50) / 100,
                mensagem=f"Análise completa: Score {resultado.get('score_qualidade', 'N/A')}/100",
                detalhes=resultado,
                acoes_sugeridas=resultado.get('sugestoes_correcao', [])
            )
            
        except Exception as e:
            self.logger.warning(f"⚠️ Erro ao processar resposta: {e}")
            return self._analise_fallback(dados)
    
    def validar_calculos(self, calculos: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida cálculos de VR com IA
        
        Args:
            calculos: Dicionário com cálculos realizados
            
        Returns:
            Validação com anomalias detectadas
        """
        self.logger.info("💰 Validando cálculos com Gemini...")
        
        # Preparar dados para análise
        dados_analise = {
            'total_colaboradores': len(calculos.get('detalhes', [])),
            'valor_total': sum(c.get('valor_total', 0) for c in calculos.get('detalhes', [])),
            'valores_por_sindicato': self._agrupar_por_sindicato(calculos),
            'distribuicao_valores': self._calcular_distribuicao(calculos)
        }
        
        prompt = f"""
        Analise os seguintes cálculos de VR e identifique anomalias:
        
        {json.dumps(dados_analise, indent=2, default=str)}
        
        Verifique:
        1. Valores muito discrepantes
        2. Sindicatos com valores incorretos
        3. Proporções empresa/colaborador incorretas
        4. Valores totais suspeitos
        
        Responda em JSON com anomalias encontradas e nível de gravidade.
        """
        
        resposta = self._chamar_gemini(prompt)
        
        if resposta:
            try:
                resultado = self._extrair_json(resposta)
                return resultado
            except:
                pass
        
        # Fallback com validação básica
        return self._validacao_basica_calculos(calculos)
    
    def sugerir_correcoes(self, inconsistencias: List[Any]) -> List[str]:
        """
        Sugere correções para inconsistências encontradas
        
        Args:
            inconsistencias: Lista de inconsistências
            
        Returns:
            Lista de sugestões de correção
        """
        if not inconsistencias:
            return []
        
        self.logger.info("💡 Gerando sugestões de correção com Gemini...")
        
        # Preparar dados das inconsistências
        dados_inc = []
        for inc in inconsistencias[:10]:  # Limitar para não exceder tokens
            dados_inc.append({
                'tipo': getattr(inc, 'tipo', 'desconhecido'),
                'descricao': getattr(inc, 'descricao', str(inc)),
                'campo': getattr(inc, 'campo', ''),
                'valor': getattr(inc, 'valor_atual', '')
            })
        
        prompt = f"""
        Como especialista em dados de RH, sugira correções para as seguintes inconsistências:
        
        {json.dumps(dados_inc, indent=2, default=str)}
        
        Para cada problema, forneça:
        1. Correção recomendada
        2. Como prevenir no futuro
        3. Prioridade (Alta/Média/Baixa)
        
        Seja prático e direto.
        """
        
        resposta = self._chamar_gemini(prompt)
        
        if resposta:
            # Processar resposta em lista de sugestões
            sugestoes = []
            linhas = resposta.split('\n')
            for linha in linhas:
                if linha.strip() and not linha.startswith('#'):
                    sugestoes.append(linha.strip())
            return sugestoes[:5]  # Retornar top 5 sugestões
        
        # Fallback
        return ["Revisar dados manualmente", "Verificar fontes originais"]
    
    def detectar_anomalias_valores(self, df: pd.DataFrame, coluna_valor: str = 'valor_vr') -> List[Dict]:
        """
        Detecta anomalias em valores usando IA
        
        Args:
            df: DataFrame com dados
            coluna_valor: Nome da coluna de valores
            
        Returns:
            Lista de anomalias detectadas
        """
        if coluna_valor not in df.columns:
            return []
        
        # Estatísticas básicas
        stats = {
            'media': df[coluna_valor].mean(),
            'mediana': df[coluna_valor].median(),
            'desvio': df[coluna_valor].std(),
            'min': df[coluna_valor].min(),
            'max': df[coluna_valor].max(),
            'q1': df[coluna_valor].quantile(0.25),
            'q3': df[coluna_valor].quantile(0.75)
        }
        
        # Identificar outliers estatísticos
        iqr = stats['q3'] - stats['q1']
        limite_inferior = stats['q1'] - 1.5 * iqr
        limite_superior = stats['q3'] + 1.5 * iqr
        
        outliers = df[(df[coluna_valor] < limite_inferior) | (df[coluna_valor] > limite_superior)]
        
        if len(outliers) == 0:
            return []
        
        # Analisar outliers com IA
        prompt = f"""
        Analise os seguintes valores outliers de VR:
        
        Estatísticas gerais:
        {json.dumps(stats, indent=2, default=float)}
        
        Valores outliers encontrados:
        {outliers[coluna_valor].tolist()[:20]}
        
        Para cada outlier, determine se é:
        1. Erro de digitação provável
        2. Valor válido mas excepcional
        3. Possível fraude
        
        Responda em JSON.
        """
        
        resposta = self._chamar_gemini(prompt)
        
        anomalias = []
        if resposta:
            try:
                resultado = self._extrair_json(resposta)
                # Processar resultado em lista de anomalias
                for idx, row in outliers.iterrows():
                    anomalias.append({
                        'matricula': row.get('matricula', idx),
                        'valor': row[coluna_valor],
                        'tipo': 'outlier',
                        'gravidade': 'ALTA' if row[coluna_valor] > stats['max'] * 0.9 else 'MÉDIA',
                        'sugestao': f"Verificar valor R$ {row[coluna_valor]:.2f}"
                    })
            except:
                pass
        
        return anomalias
    
    def gerar_relatorio_executivo(self, dados_processamento: Dict) -> str:
        """
        Gera relatório executivo do processamento
        
        Args:
            dados_processamento: Dados completos do processamento
            
        Returns:
            Relatório formatado
        """
        self.logger.info("📝 Gerando relatório executivo com Gemini...")
        
        prompt = self.prompts["resumo_executivo"].format(
            dados=json.dumps(dados_processamento, indent=2, default=str)
        )
        
        resposta = self._chamar_gemini(prompt)
        
        if resposta:
            return resposta
        
        # Fallback com relatório básico
        return self._gerar_relatorio_basico(dados_processamento)
    
    def explicar_regra_negocio(self, regra: str, contexto: str = "") -> str:
        """
        Explica uma regra de negócio de forma clara
        
        Args:
            regra: Nome ou descrição da regra
            contexto: Contexto adicional
            
        Returns:
            Explicação clara da regra
        """
        prompt = self.prompts["explicar_regra"].format(
            regra=regra,
            contexto=contexto
        )
        
        resposta = self._chamar_gemini(prompt, use_cache=True)
        
        if resposta:
            return resposta
        
        return f"Regra: {regra}. Consulte o manual para mais detalhes."
    
    # Métodos auxiliares privados
    
    def _preparar_resumo_dados(self, dados: Dict[str, pd.DataFrame]) -> str:
        """Prepara resumo dos dados para análise"""
        resumo = []
        
        for nome_base, df in dados.items():
            info = {
                'base': nome_base,
                'registros': len(df),
                'colunas': list(df.columns),
                'tipos': df.dtypes.to_dict(),
                'nulos': df.isnull().sum().to_dict(),
                'amostra': df.head(3).to_dict() if len(df) > 0 else {}
            }
            resumo.append(info)
        
        return json.dumps(resumo, indent=2, default=str)
    
    def _extrair_json(self, texto: str) -> Dict:
        """Extrai JSON de uma resposta de texto"""
        # Tentar encontrar JSON no texto
        import re
        
        # Procurar por padrão JSON
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, texto, re.DOTALL)
        
        if matches:
            # Tentar parsear o maior match (provavelmente o JSON completo)
            for match in sorted(matches, key=len, reverse=True):
                try:
                    return json.loads(match)
                except:
                    continue
        
        # Se não encontrar, tentar limpar e parsear o texto todo
        texto_limpo = texto.strip()
        if texto_limpo.startswith('```json'):
            texto_limpo = texto_limpo[7:]
        if texto_limpo.endswith('```'):
            texto_limpo = texto_limpo[:-3]
        
        try:
            return json.loads(texto_limpo)
        except:
            # Retornar dict vazio se falhar
            return {}
    
    def _analise_fallback(self, dados: Dict[str, pd.DataFrame]) -> AnaliseIA:
        """Análise fallback quando Gemini não está disponível"""
        problemas = []
        sugestoes = []
        score = 100
        
        for nome_base, df in dados.items():
            # Verificar dados nulos
            nulos = df.isnull().sum().sum()
            if nulos > 0:
                problemas.append(f"{nome_base}: {nulos} valores nulos")
                sugestoes.append(f"Preencher valores nulos em {nome_base}")
                score -= 5
            
            # Verificar duplicatas
            if 'matricula' in df.columns:
                dups = df.duplicated(subset=['matricula']).sum()
                if dups > 0:
                    problemas.append(f"{nome_base}: {dups} matrículas duplicadas")
                    sugestoes.append(f"Remover duplicatas em {nome_base}")
                    score -= 10
        
        return AnaliseIA(
            tipo='validacao',
            confianca=0.7,
            mensagem=f"Análise básica: {len(problemas)} problemas encontrados",
            detalhes={
                'problemas': problemas,
                'score_qualidade': max(score, 0)
            },
            acoes_sugeridas=sugestoes
        )
    
    def _agrupar_por_sindicato(self, calculos: Dict) -> Dict:
        """Agrupa valores por sindicato"""
        agrupamento = {}
        
        for detalhe in calculos.get('detalhes', []):
            sindicato = detalhe.get('sindicato', 'N/A')
            if sindicato not in agrupamento:
                agrupamento[sindicato] = {
                    'quantidade': 0,
                    'valor_total': 0
                }
            agrupamento[sindicato]['quantidade'] += 1
            agrupamento[sindicato]['valor_total'] += detalhe.get('valor_total', 0)
        
        return agrupamento
    
    def _calcular_distribuicao(self, calculos: Dict) -> Dict:
        """Calcula distribuição de valores"""
        valores = [d.get('valor_total', 0) for d in calculos.get('detalhes', [])]
        
        if not valores:
            return {}
        
        return {
            'media': np.mean(valores),
            'mediana': np.median(valores),
            'desvio_padrao': np.std(valores),
            'minimo': min(valores),
            'maximo': max(valores),
            'quartil_1': np.percentile(valores, 25),
            'quartil_3': np.percentile(valores, 75)
        }
    
    def _validacao_basica_calculos(self, calculos: Dict) -> Dict:
        """Validação básica de cálculos sem IA"""
        anomalias = []
        
        for detalhe in calculos.get('detalhes', []):
            # Verificar proporção empresa/colaborador
            valor_total = detalhe.get('valor_total', 0)
            valor_empresa = detalhe.get('valor_empresa', 0)
            valor_colaborador = detalhe.get('valor_colaborador', 0)
            
            if valor_total > 0:
                prop_empresa = valor_empresa / valor_total
                prop_colaborador = valor_colaborador / valor_total
                
                if abs(prop_empresa - 0.8) > 0.01:
                    anomalias.append({
                        'tipo': 'proporcao_incorreta',
                        'matricula': detalhe.get('matricula'),
                        'descricao': f'Proporção empresa incorreta: {prop_empresa:.2%}'
                    })
        
        return {'anomalias': anomalias}
    
    def _gerar_relatorio_basico(self, dados: Dict) -> str:
        """Gera relatório básico sem IA"""
        return f"""
        RELATÓRIO DE PROCESSAMENTO VR
        ==============================
        
        Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}
        
        RESUMO
        ------
        Total de colaboradores: {dados.get('total_colaboradores', 'N/A')}
        Valor total: R$ {dados.get('valor_total', 0):,.2f}
        
        DETALHAMENTO
        -----------
        Empresa (80%): R$ {dados.get('valor_empresa', 0):,.2f}
        Colaboradores (20%): R$ {dados.get('valor_colaborador', 0):,.2f}
        
        Status: Processamento concluído
        """
    
    def limpar_cache(self):
        """Limpa o cache de respostas"""
        self.cache.clear()
        self.logger.info("🧹 Cache limpo")
    
    def get_estatisticas_uso(self) -> Dict:
        """Retorna estatísticas de uso da API"""
        return {
            'requisicoes_realizadas': self.request_count,
            'cache_size': len(self.cache),
            'rate_limit': f"{self.requests_per_minute}/min"
        }


# Função helper para criar assistente rapidamente
def criar_assistente_gemini(api_key: Optional[str] = None) -> GeminiAssistant:
    """
    Cria uma instância do assistente Gemini
    
    Args:
        api_key: Chave da API (opcional se estiver em .env)
        
    Returns:
        Instância configurada do GeminiAssistant
    """
    return GeminiAssistant(api_key)


# Exemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🤖 Teste do Gemini Assistant")
    print("=" * 40)
    
    try:
        # Criar assistente
        assistant = criar_assistente_gemini()
        
        # Teste 1: Validar dados
        print("\n📊 Teste 1: Análise de dados")
        dados_teste = {
            'ativos': pd.DataFrame({
                'matricula': [123, 456, 789, 123],  # Duplicata
                'nome': ['João', 'Maria', None, 'Pedro'],  # Null
                'valor_vr': [500, 600, 10000, 550]  # Outlier
            })
        }
        
        analise = assistant.analisar_dados_entrada(dados_teste)
        print(f"Score de qualidade: {analise.confianca * 100:.0f}%")
        print(f"Mensagem: {analise.mensagem}")
        print(f"Ações sugeridas: {analise.acoes_sugeridas}")
        
        # Teste 2: Detectar anomalias
        print("\n💰 Teste 2: Detectar anomalias em valores")
        anomalias = assistant.detectar_anomalias_valores(dados_teste['ativos'])
        print(f"Anomalias encontradas: {len(anomalias)}")
        for anomalia in anomalias:
            print(f"  - {anomalia}")
        
        # Teste 3: Explicar regra
        print("\n📖 Teste 3: Explicar regra de negócio")
        explicacao = assistant.explicar_regra_negocio(
            "Regra dos 15 dias para desligamento",
            "Sistema de Vale Refeição"
        )
        print(f"Explicação: {explicacao[:200]}...")
        
        # Estatísticas
        print("\n📊 Estatísticas de uso:")
        stats = assistant.get_estatisticas_uso()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"❌ Erro: {e}")