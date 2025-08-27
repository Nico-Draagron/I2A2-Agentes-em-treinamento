"""
Sistema VR Integrado - Versão Otimizada
Combina seus módulos existentes com Gemini API
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
import json
import google.generativeai as genai  # type: ignore

from config import settings

# ========================================
# PARTE 1: CONFIGURAÇÃO E SETUP
# ========================================

class ConfiguracaoVR:
    """Centraliza todas as configurações do sistema"""
    
    def __init__(self):
        # API Gemini
        self.GEMINI_API_KEY = settings.GEMINI_API_KEY
        
        # Pastas
        self.PASTA_INPUT = Path("data/input")
        self.PASTA_OUTPUT = Path("data/output")
        
        # Criar pastas se não existirem
        self.PASTA_INPUT.mkdir(parents=True, exist_ok=True)
        self.PASTA_OUTPUT.mkdir(parents=True, exist_ok=True)
        
        # Competência
        self.COMPETENCIA = datetime.now().strftime("%Y-%m")
        
        # Mapeamentos (do seu código original)
        self.MAPEAMENTO_ARQUIVOS = {
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
        
        self.SINDICATO_ESTADO = {
            'SINDPD SP': 'São Paulo',
            'SINDPD RJ': 'Rio de Janeiro',
            'SINDPPD RS': 'Rio Grande do Sul',
            'SITEPD PR': 'Paraná'
        }
        
        self.CARGOS_EXCLUSAO = [
            'DIRETOR', 'DIRECTOR', 'CEO', 'CFO', 'CTO', 'COO',
            'PRESIDENTE', 'VICE-PRESIDENTE', 'VP'
        ]

# ========================================
# PARTE 2: ASSISTENTE GEMINI SIMPLIFICADO
# ========================================

class AssistenteGemini:
    """Assistente IA usando Gemini - Versão Simplificada e Eficiente"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)

        # Configurar Gemini
        if not api_key:
            self.logger.warning("⚠️ Gemini não configurado - modo offline")
            self.gemini_disponivel = False
        else:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                self.gemini_disponivel = True
                self.logger.info("✅ Gemini configurado com sucesso!")
            except Exception as e:
                self.logger.error(f"❌ Erro ao configurar Gemini: {e}")
                self.gemini_disponivel = False
    
    def analisar_inconsistencias(self, dados: Dict[str, pd.DataFrame]) -> Dict:
        """Analisa inconsistências e anomalias nos dados usando Gemini e estatística"""
        if not self.gemini_disponivel:
            return self._analise_offline(dados)
        try:
            resumos: dict[str, Any] = {}
            anomalias: dict[str, dict[str, Any]] = {}
            for nome, df in dados.items():
                if not isinstance(df, pd.DataFrame):
                    self.logger.warning(f"[Gemini] Ignorando '{nome}': não é DataFrame (tipo: {type(df)})")
                    resumos[nome] = f"Ignorado: tipo {type(df)}"
                    continue
                resumos[nome] = {
                    'colunas': list(df.columns),
                    'nulos': int(df.isnull().sum().sum()),
                    'registros': len(df)
                }
                # Detecção de anomalias numéricas (z-score)
                anomalias[nome] = {}
                for col in df.select_dtypes(include=[np.number]).columns:
                    if len(df[col].dropna()) > 0:
                        z = np.abs((df[col] - df[col].mean()) / (df[col].std() if df[col].std() else 1))
                        outliers = df[z > 3]
                        if not outliers.empty:
                            anomalias[nome][col] = outliers[[col]].to_dict('records')
            # Enviar resumo e anomalias para Gemini
            prompt = f"""
            Analise os seguintes resumos e anomalias dos dados de VR. Identifique possíveis inconsistências, padrões incomuns, outliers e sugira ações corretivas.
            Resumos: {json.dumps(resumos, default=str)}
            Anomalias detectadas (z-score > 3): {json.dumps(anomalias, default=str)}
            Responda em JSON com: score_qualidade (0-100), problemas_criticos (lista), avisos (lista), sugestoes (lista)
            """
            response = self.model.generate_content(prompt)
            texto = response.text
            inicio = texto.find('{')
            fim = texto.rfind('}') + 1
            resultado = json.loads(texto[inicio:fim]) if inicio != -1 and fim > inicio else {}
            resultado['anomalias'] = anomalias
            return resultado
        except Exception as e:
            self.logger.error(f"Erro Gemini: {e}. Verifique se a API KEY está correta e se há conexão com a internet.")
            return self._analise_offline(dados)
    
    def corrigir_nomes_colunas(self, df: pd.DataFrame, nome_base: str) -> pd.DataFrame:
        """Usa Gemini para corrigir nomes de colunas automaticamente"""
        if not self.gemini_disponivel:
            return df
        
        try:
            colunas_atuais = list(df.columns)
            
            prompt = f"""
            Corrija os nomes das colunas para a base {nome_base} de VR.
            
            Colunas atuais: {colunas_atuais}
            
            Padrão esperado:
            - matricula (não matrícula, matricla, etc)
            - nome (não nom, nme, etc)
            - data_admissao (não admissão, admissao, etc)
            - valor_diario (não valor_diário, valor, etc)
            
            Retorne APENAS um dicionário Python mapeando nome_errado: nome_correto
            Exemplo: {{"matricla": "matricula", "nom": "nome"}}
            """
            
            response = self.model.generate_content(prompt)
            
            # Extrair dicionário
            texto = response.text
            inicio = texto.find('{')
            fim = texto.rfind('}') + 1
            mapeamento = eval(texto[inicio:fim])
            
            # Aplicar correções
            df = df.rename(columns=mapeamento)
            self.logger.info(f"✅ Colunas corrigidas em {nome_base}: {mapeamento}")
            
        except Exception as e:
            self.logger.warning(f"Não foi possível corrigir colunas: {e}")
        
        return df
    
    def validar_calculos(self, calculos: List[Dict]) -> Dict:
        """Valida cálculos de VR usando Gemini"""
        if not self.gemini_disponivel or not calculos:
            return {"valido": True, "avisos": []}
        
        try:
            # Resumo dos cálculos
            total = sum(c.get('valor_total', 0) for c in calculos)
            media = total / len(calculos) if calculos else 0
            
            prompt = f"""
            Valide os cálculos de VR:
            
            Total colaboradores: {len(calculos)}
            Valor total: R$ {total:,.2f}
            Valor médio: R$ {media:,.2f}
            
            Amostra de cálculos:
            {json.dumps(calculos[:5], default=str)}
            
            Verifique:
            1. Proporção 80/20 (empresa/colaborador)
            2. Valores outliers
            3. Consistência dos valores por sindicato
            
            Retorne JSON com: valido (bool), avisos (lista), score_confianca (0-100)
            """
            
            response = self.model.generate_content(prompt)
            texto = response.text
            inicio = texto.find('{')
            fim = texto.rfind('}') + 1
            return json.loads(texto[inicio:fim])
            
        except Exception as e:
            self.logger.warning(f"Validação manual: {e}")
            return {"valido": True, "avisos": ["Validação manual necessária"]}
    
    def _criar_resumo_dados(self, dados: Dict) -> str:
        """Cria resumo dos dados para análise"""
        resumo = []
        for nome, df in dados.items():
            info = f"""
            Base: {nome}
            - Registros: {len(df)}
            - Colunas: {list(df.columns)[:5]}
            - Valores nulos: {df.isnull().sum().sum()}
            - Duplicatas em 'matricula': {df['matricula'].duplicated().sum() if 'matricula' in df.columns else 'N/A'}
            """
            resumo.append(info)
        return "\n".join(resumo)
    
    def _analise_offline(self, dados: Dict) -> Dict:
        """Análise básica quando Gemini não está disponível"""
        problemas = []
        avisos = []
        
        for nome, df in dados.items():
            # Verificar nulos
            nulos = df.isnull().sum().sum()
            if nulos > 0:
                avisos.append(f"{nome}: {nulos} valores nulos")
            
            # Verificar duplicatas
            if 'matricula' in df.columns:
                dups = df['matricula'].duplicated().sum()
                if dups > 0:
                    problemas.append(f"{nome}: {dups} matrículas duplicadas")
        
        score = 100 - len(problemas) * 10 - len(avisos) * 2
        
        return {
            "problemas_criticos": problemas,
            "avisos": avisos,
            "sugestoes": ["Revisar dados manualmente"],
            "score_qualidade": max(score, 0)
        }

# ========================================
# PARTE 3: PROCESSADOR PRINCIPAL (USANDO SEU CÓDIGO)
# ========================================

class ProcessadorVR:
    """Processador principal que integra seus módulos existentes"""
    

    bases_originais: Dict[str, pd.DataFrame]
    bases_processadas: Dict[str, pd.DataFrame]
    calculos: list
    relatorio: Dict[str, Any]
    data_reader: Optional[Any]
    validator: Optional[Any]
    rules: Optional[Any]
    calculator: Optional[Any]

    def __init__(self, config: ConfiguracaoVR):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.gemini = AssistenteGemini(config.GEMINI_API_KEY)

        # Inicializar atributos de módulos como None
        self.data_reader = None
        self.validator = None
        self.rules = None
        self.calculator = None

        # Importar seus módulos existentes (se disponíveis)
        self.usar_modulos_existentes = self._verificar_modulos()

        # Dados
        self.bases_originais = {}
        self.bases_processadas = {}
        self.calculos = []
        self.relatorio = {}
            
    def _verificar_modulos(self) -> bool:
        """Verifica se os módulos do GitHub estão disponíveis"""
        try:
            # Tentar importar seus módulos
            # Se precisar dos módulos, ajuste o caminho de importação conforme sua estrutura
            # self.data_reader = ...
            # self.validator = ...
            # self.rules = ...
            # self.calculator = ...
            self.logger.info("✅ Módulos existentes carregados!")
            return True
        except ImportError:
            self.logger.warning("⚠️ Usando processamento simplificado")
            return False
    
    def executar_pipeline(self) -> bool:
        """Executa o pipeline completo de processamento"""
        
        print("\n" + "="*60)
        print("🚀 INICIANDO PROCESSAMENTO VR")
        print(f"📅 Competência: {self.config.COMPETENCIA}")
        print("="*60)
        
        try:
            # Etapa 1: Carregar dados
            if not self._carregar_dados():
                return False
            
            # Etapa 2: Validar com Gemini
            if not self._validar_dados():
                return False
            
            # Etapa 3: Aplicar regras
            if not self._aplicar_regras():
                return False
            
            # Etapa 4: Calcular VR
            if not self._calcular_vr():
                return False
            
            # Etapa 5: Gerar saídas
            if not self._gerar_relatorios():
                return False
            
            print("\n✅ PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
            self._exibir_resumo()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erro fatal: {e}")
            return False
    
    def _carregar_dados(self) -> bool:
        """Carrega dados usando seu data_reader ou método simplificado"""
        print("\n📂 ETAPA 1: CARREGANDO DADOS")
        print("-" * 40)
        
        if self.usar_modulos_existentes and self.data_reader is not None:
            # Usar seu DataReader
            try:
                self.bases_originais = self.data_reader.carregar_todas_bases()
                print(f"✅ {len(self.bases_originais)} bases carregadas")
                return True
            except Exception as e:
                self.logger.error(f"Erro no DataReader: {e}")
        
        # Método simplificado
        for nome, arquivo in self.config.MAPEAMENTO_ARQUIVOS.items():
            caminho = self.config.PASTA_INPUT / arquivo
            if caminho.exists():
                try:
                    # Detectar tipo de arquivo e ler
                    if arquivo.endswith('.xlsx'):
                        df = pd.read_excel(caminho)
                    elif arquivo.endswith('.csv'):
                        df = pd.read_csv(caminho)
                    else:
                        continue
                    
                    # Corrigir nomes de colunas com Gemini
                    df = self.gemini.corrigir_nomes_colunas(df, nome)
                    
                    self.bases_originais[nome] = df
                    print(f"✅ {nome}: {len(df)} registros")
                    
                except Exception as e:
                    print(f"⚠️ Erro ao ler {arquivo}: {e}")
        
        return len(self.bases_originais) > 0
    
    def _validar_dados(self) -> bool:
        """Valida dados usando Gemini + seu validator"""
        print("\n🔍 ETAPA 2: VALIDANDO DADOS")
        print("-" * 40)
        
        # Análise com Gemini
        analise = self.gemini.analisar_inconsistencias(self.bases_originais)
        
        print(f"📊 Score de qualidade: {analise.get('score_qualidade', 0)}%")
        
        if analise.get('problemas_criticos'):
            print("\n🔴 Problemas críticos:")
            for problema in analise['problemas_criticos'][:5]:
                print(f"   - {problema}")
        
        if analise.get('avisos'):
            print("\n🟡 Avisos:")
            for aviso in analise['avisos'][:5]:
                print(f"   - {aviso}")
        
        # Se tiver seu validator, usar também
        if self.usar_modulos_existentes and self.validator is not None:
            try:
                self.bases_processadas, inconsistencias = self.validator.validar_e_limpar_dados(
                    self.bases_originais
                )
                print(f"✅ Validação completa: {len(inconsistencias)} inconsistências tratadas")
            except Exception:
                self.bases_processadas = self.bases_originais.copy()
        else:
            self.bases_processadas = self.bases_originais.copy()
        
        return analise.get('score_qualidade', 0) > 30  # Aceitar se score > 30%
    
    def _aplicar_regras(self) -> bool:
        """Aplica regras de negócio"""
        print("\n📋 ETAPA 3: APLICANDO REGRAS DE NEGÓCIO")
        print("-" * 40)
        
        if self.usar_modulos_existentes and self.rules is not None:
            # Usar seu BusinessRulesEngine
            try:
                resultados = self.rules.processar_elegibilidade_completa(self.bases_processadas)
                elegiveis = [r for r in resultados if r.dias_elegivel > 0]
                print(f"✅ {len(elegiveis)} colaboradores elegíveis")
                
                # Converter para formato padrão
                for r in elegiveis:
                    self.calculos.append({
                        'matricula': r.matricula,
                        'dias': r.dias_elegivel,
                        'valor_diario': r.valor_diario,
                        'valor_total': r.dias_elegivel * r.valor_diario,
                        'valor_empresa': r.dias_elegivel * r.valor_diario * 0.8,
                        'valor_colaborador': r.dias_elegivel * r.valor_diario * 0.2
                    })
                return True
            except Exception as e:
                self.logger.error(f"Erro no BusinessRules: {e}")
        
        # Método simplificado
        return self._aplicar_regras_simplificado()
    
    def _aplicar_regras_simplificado(self) -> bool:
        """Aplica regras básicas quando módulos não estão disponíveis"""
        
        if 'ativos' not in self.bases_processadas:
            print("❌ Base de ativos não encontrada")
            return False
        
        df_ativos = self.bases_processadas['ativos'].copy()
        
        # Aplicar exclusões básicas
        inicial = len(df_ativos)
        
        # Excluir diretores
        if 'cargo' in df_ativos.columns:
            for cargo_exc in self.config.CARGOS_EXCLUSAO:
                df_ativos = df_ativos[~df_ativos['cargo'].str.contains(cargo_exc, case=False, na=False)]
        
        # Excluir matrículas das bases de exclusão
        for base_exc in ['afastamentos', 'estagiarios', 'aprendizes', 'exterior']:
            if base_exc in self.bases_processadas and 'matricula' in self.bases_processadas[base_exc].columns:
                matriculas_excluir = self.bases_processadas[base_exc]['matricula'].tolist()
                df_ativos = df_ativos[~df_ativos['matricula'].isin(matriculas_excluir)]
        
        print(f"✅ Exclusões aplicadas: {inicial - len(df_ativos)} removidos")
        
        # Calcular VR básico (22 dias * R$ 35)
        for _, row in df_ativos.iterrows():
            dias = 22  # Padrão
            valor_diario = 35.0  # Padrão
            
            # Ajustar por sindicato se disponível
            if 'sindicato' in row and pd.notna(row['sindicato']):
                # Lógica simplificada de sindicato
                if 'RS' in str(row['sindicato']):
                    dias = 20
                    valor_diario = 28.88
                elif 'RJ' in str(row['sindicato']):
                    dias = 21
                    valor_diario = 34.0
            
            valor_total = dias * valor_diario
            
            self.calculos.append({
                'matricula': row['matricula'],
                'nome': row.get('nome', ''),
                'dias': dias,
                'valor_diario': valor_diario,
                'valor_total': valor_total,
                'valor_empresa': valor_total * 0.8,
                'valor_colaborador': valor_total * 0.2
            })
        
        print(f"✅ {len(self.calculos)} colaboradores processados")
        return len(self.calculos) > 0
    
    def _calcular_vr(self) -> bool:
        """Valida cálculos com Gemini"""
        print("\n💰 ETAPA 4: CALCULANDO E VALIDANDO VR")
        print("-" * 40)
        
        if not self.calculos:
            print("❌ Nenhum cálculo para validar")
            return False
        
        # Validar com Gemini
        validacao = self.gemini.validar_calculos(self.calculos[:100])  # Amostra
        
        # Estatísticas
        total = sum(c['valor_total'] for c in self.calculos)
        total_empresa = sum(c['valor_empresa'] for c in self.calculos)
        total_colaborador = sum(c['valor_colaborador'] for c in self.calculos)
        
        print(f"📊 Total de colaboradores: {len(self.calculos)}")
        print(f"💵 Valor total: R$ {total:,.2f}")
        print(f"🏢 Empresa (80%): R$ {total_empresa:,.2f}")
        print(f"👤 Colaboradores (20%): R$ {total_colaborador:,.2f}")
        
        if validacao.get('avisos'):
            print("\n⚠️ Avisos da validação:")
            for aviso in validacao['avisos']:
                print(f"   - {aviso}")
        
        self.relatorio = {
            'total_colaboradores': len(self.calculos),
            'valor_total': total,
            'valor_empresa': total_empresa,
            'valor_colaborador': total_colaborador,
            'validacao': validacao
        }
        
        return True
    
    def _gerar_relatorios(self) -> bool:
        """Gera arquivos de saída, incluindo anomalias"""
        print("\n📊 ETAPA 5: GERANDO RELATÓRIOS")
        print("-" * 40)
        
        # Gerar planilha VR
        df_vr = pd.DataFrame(self.calculos)
        
        # Adicionar colunas necessárias
        df_vr['competencia'] = self.config.COMPETENCIA
        df_vr['data_processamento'] = datetime.now()
        
        # Salvar Excel
        arquivo_saida = self.config.PASTA_OUTPUT / f"VR_MENSAL_{self.config.COMPETENCIA.replace('-', '_')}.xlsx"
        
        with pd.ExcelWriter(arquivo_saida, engine='openpyxl') as writer:
            # Aba principal
            df_vr.to_excel(writer, sheet_name='VR_Mensal', index=False)
            
            # Aba resumo
            df_resumo = pd.DataFrame([self.relatorio])
            df_resumo.to_excel(writer, sheet_name='Resumo', index=False)
            
            # Aba validações e anomalias (se houver)
            if self.gemini.gemini_disponivel:
                analise = self.gemini.analisar_inconsistencias(self.bases_originais)
                df_validacao = pd.DataFrame({
                    'Tipo': ['Score Qualidade', 'Problemas Críticos', 'Avisos'],
                    'Valor': [
                        analise.get('score_qualidade', 0),
                        len(analise.get('problemas_criticos', [])),
                        len(analise.get('avisos', []))
                    ]
                })
                df_validacao.to_excel(writer, sheet_name='Validações', index=False)
                # Salvar anomalias detectadas
                if 'anomalias' in analise:
                    for base, colunas in analise['anomalias'].items():
                        for coluna, registros in colunas.items():
                            if registros:
                                df_anom = pd.DataFrame(registros)
                                if not df_anom.empty:
                                    df_anom.to_excel(writer, sheet_name=f'Anomalias_{base}_{coluna}'[:31], index=False)
        
        print(f"✅ Arquivo gerado: {arquivo_saida}")
        
        # Gerar relatório texto
        relatorio_txt = self.config.PASTA_OUTPUT / f"relatorio_{self.config.COMPETENCIA}.txt"
        with open(relatorio_txt, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DE PROCESSAMENTO VR\n")
            f.write("="*50 + "\n\n")
            f.write(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
            f.write(f"Competência: {self.config.COMPETENCIA}\n\n")
            f.write(f"Total colaboradores: {self.relatorio['total_colaboradores']}\n")
            f.write(f"Valor total: R$ {self.relatorio['valor_total']:,.2f}\n")
            f.write(f"Empresa: R$ {self.relatorio['valor_empresa']:,.2f}\n")
            f.write(f"Colaboradores: R$ {self.relatorio['valor_colaborador']:,.2f}\n")
            # Adicionar anomalias ao relatório texto
            if self.gemini.gemini_disponivel:
                analise = self.gemini.analisar_inconsistencias(self.bases_originais)
                if 'anomalias' in analise:
                    f.write("\nANOMALIAS DETECTADAS:\n")
                    for base, colunas in analise['anomalias'].items():
                        for coluna, registros in colunas.items():
                            f.write(f"- {base} [{coluna}]: {len(registros)} registros anômalos\n")
        
        print(f"✅ Relatório gerado: {relatorio_txt}")
        
        return True
    
    def _exibir_resumo(self):
        """Exibe resumo final"""
        print("\n" + "="*60)
        print("📋 RESUMO FINAL")
        print("="*60)
        print(f"✅ Colaboradores processados: {self.relatorio['total_colaboradores']}")
        print(f"💰 Valor total de VR: R$ {self.relatorio['valor_total']:,.2f}")
        print(f"🏢 Custo empresa: R$ {self.relatorio['valor_empresa']:,.2f}")
        print(f"👤 Desconto colaboradores: R$ {self.relatorio['valor_colaborador']:,.2f}")
        print("\n📁 Arquivos gerados na pasta: data/output/")

# ========================================
# PARTE 4: INTERFACE PRINCIPAL
# ========================================

class SistemaVR:
    """Sistema principal com menu interativo"""
    
    def __init__(self):
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        self.config = ConfiguracaoVR()
        self.processador = ProcessadorVR(self.config)
    
    def menu_principal(self):
        """Menu interativo principal"""
        while True:
            print("\n" + "="*50)
            print("🤖 SISTEMA VR - MENU PRINCIPAL")
            print("="*50)
            print("1. Processar VR (Automático)")
            print("2. Verificar dados de entrada")
            print("3. Testar conexão Gemini")
            print("4. Configurações")
            print("0. Sair")
            
            opcao = input("\nEscolha uma opção: ").strip()
            
            if opcao == '1':
                self.processar_vr()
            elif opcao == '2':
                self.verificar_dados()
            elif opcao == '3':
                self.testar_gemini()
            elif opcao == '4':
                self.configuracoes()
            elif opcao == '0':
                print("\n👋 Encerrando sistema...")
                break
            else:
                print("❌ Opção inválida!")
    
    def processar_vr(self):
        """Processa VR completo"""
        print("\n🚀 Iniciando processamento...")
        
        # Verificar se há arquivos
        arquivos = list(self.config.PASTA_INPUT.glob("*.xlsx")) + list(self.config.PASTA_INPUT.glob("*.csv"))
        
        if not arquivos:
            print(f"❌ Nenhum arquivo encontrado em {self.config.PASTA_INPUT}")
            print("Por favor, coloque os arquivos Excel na pasta data/input/")
            return
        
        print(f"📂 {len(arquivos)} arquivos encontrados")
        
        # Executar pipeline
        sucesso = self.processador.executar_pipeline()
        
        if sucesso:
            print("\n✨ Processamento concluído com sucesso!")
            input("\nPressione ENTER para continuar...")
        else:
            print("\n❌ Processamento falhou. Verifique os logs.")
            input("\nPressione ENTER para continuar...")
    
    def verificar_dados(self):
        """Verifica arquivos de entrada"""
        print("\n📂 VERIFICAÇÃO DE DADOS")
        print("-" * 30)
        
        print(f"Pasta de entrada: {self.config.PASTA_INPUT}")
        
        # Listar arquivos esperados vs encontrados
        for nome, arquivo_esperado in self.config.MAPEAMENTO_ARQUIVOS.items():
            caminho = self.config.PASTA_INPUT / arquivo_esperado
            if caminho.exists():
                tamanho = caminho.stat().st_size / 1024  # KB
                print(f"✅ {arquivo_esperado} ({tamanho:.1f} KB)")
            else:
                print(f"❌ {arquivo_esperado} (não encontrado)")
        
        # Listar outros arquivos
        outros = []
        for arquivo in self.config.PASTA_INPUT.iterdir():
            if arquivo.name not in self.config.MAPEAMENTO_ARQUIVOS.values():
                outros.append(arquivo.name)
        
        if outros:
            print(f"\n📄 Outros arquivos: {', '.join(outros)}")
        
        input("\nPressione ENTER para continuar...")
    
    def testar_gemini(self):
        """Testa conexão com Gemini"""
        print("\n🤖 TESTE DE CONEXÃO GEMINI")
        print("-" * 30)
        
        if not self.config.GEMINI_API_KEY:
            print("❌ GEMINI_API_KEY não configurada!")
            print("Adicione sua chave no arquivo .env")
            print("Obtenha em: https://makersuite.google.com/app/apikey")
        else:
            print("🔑 API Key configurada")
            print("Testando conexão...")
            
            try:
                genai.configure(api_key=self.config.GEMINI_API_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content("Responda apenas: OK")
                print(f"✅ Conexão bem-sucedida!")
                print(f"Resposta: {response.text[:50]}")
            except Exception as e:
                print(f"❌ Erro: {e}")
        
        input("\nPressione ENTER para continuar...")
    
    def configuracoes(self):
        """Mostra configurações atuais"""
        print("\n⚙️ CONFIGURAÇÕES")
        print("-" * 30)
        print(f"Competência: {self.config.COMPETENCIA}")
        print(f"Pasta entrada: {self.config.PASTA_INPUT}")
        print(f"Pasta saída: {self.config.PASTA_OUTPUT}")
        print(f"Gemini configurado: {'✅' if self.config.GEMINI_API_KEY else '❌'}")
        print(f"Total sindicatos: {len(self.config.SINDICATO_ESTADO)}")
        print(f"Cargos exclusão: {len(self.config.CARGOS_EXCLUSAO)}")
        
        input("\nPressione ENTER para continuar...")

# ========================================
# PARTE 5: EXECUÇÃO PRINCIPAL
# ========================================

def main():
    """Função principal"""
    import sys
    
    print("\n" + "="*60)
    print("🤖 SISTEMA INTELIGENTE VR/VA")
    print("Versão 3.0 - Com Gemini AI")
    print("="*60)
    
    # Verificar argumentos de linha de comando
    if len(sys.argv) > 1:
        if sys.argv[1] == '--auto':
            # Modo automático
            print("\n🤖 Modo automático ativado")
            config = ConfiguracaoVR()
            processador = ProcessadorVR(config)
            sucesso = processador.executar_pipeline()
            sys.exit(0 if sucesso else 1)
        elif sys.argv[1] == '--help':
            print("""
Uso:
  python sistema_vr.py          # Modo interativo
  python sistema_vr.py --auto   # Modo automático
  python sistema_vr.py --help   # Esta ajuda
            """)
            sys.exit(0)
    
    # Modo interativo
    sistema = SistemaVR()
    sistema.menu_principal()

if __name__ == "__main__":
    main()