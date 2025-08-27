"""
Menu Interativo Híbrido com IA - Sistema VR
Sistema avançado de análise de dados com detecção inteligente de problemas.

Autor: Agente VR + IA
Data: 2025-08
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Adicionar módulos ao path
sys.path.append(str(Path(__file__).parent / "modules"))

from modules.data_reader import carregar_bases_vr
from modules.ai_analytics import AIAnalyticsVR, analisar_fraudes_vr, RelatorioFraudes


class MenuInterativoVR:
    """
    Sistema de menu interativo híbrido com IA para análise de dados VR
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.bases_dados = {}
        self.bases_corrigidas = {}
        self.relatorio_ia = None
        self.analyzer = AIAnalyticsVR()
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        print("🤖 SISTEMA VR - MENU INTERATIVO HÍBRIDO COM IA")
        print("=" * 60)
        print("Sistema avançado de análise e correção de dados")
        print("Versão: 2.0 (Híbrido ML + IA Generativa)")
        print("=" * 60)
    
    def executar(self):
        """Executa o menu principal com escolha de modo (teste/manual) e sugestão IA"""
        while True:
            try:
                # MENU INICIAL: escolha de modo
                print("\n🤖 SISTEMA VR - INÍCIO")
                print("=" * 60)
                print("Escolha o modo de operação:")
                print("1️⃣  Modo de Teste (usa automaticamente o arquivo train.csv do dataset)")
                print("2️⃣  Modo Manual (informar caminho do arquivo)")
                print("3️⃣  Sugerir arquivo automaticamente com IA")
                print("0️⃣  Sair")
                modo = input("\n🔍 Escolha uma opção: ").strip()

                if modo == '1':
                    # Carregar train.csv automaticamente
                    caminho_teste = Path("atividade/dataset/train.csv")
                    if caminho_teste.exists():
                        from modules.data_reader import DataReader
                        self.bases_dados = {'ativos': DataReader().read_file(str(caminho_teste))}
                        print(f"✅ Arquivo de teste carregado: {caminho_teste}")
                        # Análise automática inicial
                        print("\n🤖 Executando análise inicial com IA...")
                        self.relatorio_ia, self.bases_corrigidas = analisar_fraudes_vr(
                            self.bases_dados, aplicar_correcoes=False
                        )
                        print(f"🎯 Score de Integridade: {self.relatorio_ia.score_integridade:.1f}%")
                        print(f"⚠️ {self.relatorio_ia.total_inconsistencias} inconsistências detectadas")
                        break
                    else:
                        print("❌ Arquivo de teste não encontrado: atividade/dataset/train.csv")
                        continue
                elif modo == '2':
                    self.carregar_dados()
                    break
                elif modo == '3':
                    # Sugestão IA: procurar arquivo mais provável
                    print("\n🤖 Buscando sugestão de arquivo com IA...")
                    arquivos = list(Path("data/input").glob("*.csv")) + list(Path("data/input").glob("*.xlsx"))
                    if not arquivos:
                        print("❌ Nenhum arquivo encontrado em data/input")
                        continue
                    # Simples heurística: prioriza nomes com 'ativo', 'train', 'base'
                    sugestao = None
                    for nome in ['ativo', 'train', 'base']:
                        for arq in arquivos:
                            if nome in arq.stem.lower():
                                sugestao = arq
                                break
                        if sugestao:
                            break
                    if not sugestao:
                        sugestao = arquivos[0]
                    print(f"🤖 Sugestão IA: {sugestao}")
                    usar = input("Usar este arquivo? (S/n): ").strip().lower()
                    if usar in ['', 's', 'sim', 'yes']:
                        from modules.data_reader import DataReader
                        self.bases_dados = {'ativos': DataReader().read_file(str(sugestao))}
                        print(f"✅ Arquivo carregado: {sugestao}")
                        # Análise automática inicial
                        print("\n🤖 Executando análise inicial com IA...")
                        self.relatorio_ia, self.bases_corrigidas = analisar_fraudes_vr(
                            self.bases_dados, aplicar_correcoes=False
                        )
                        print(f"🎯 Score de Integridade: {self.relatorio_ia.score_integridade:.1f}%")
                        print(f"⚠️ {self.relatorio_ia.total_inconsistencias} inconsistências detectadas")
                        break
                    else:
                        print("Operação cancelada.")
                        continue
                else:
                    print("❌ Não foi possível identificar sugestão do Gemini.")
                    continue
                elif modo == '0':
                    print("\n👋 Saindo do sistema...")
                    return
                else:
                    print("❌ Opção inválida! Tente novamente.")
                    continue

                input("\n⏸️ Pressione ENTER para continuar...")
            except KeyboardInterrupt:
                print("\n\n👋 Sistema encerrado pelo usuário.")
                return
            except Exception as e:
                print(f"\n❌ Erro inesperado: {e}")
                input("⏸️ Pressione ENTER para continuar...")

        # Após carregar dados, segue para o menu principal normal
        while True:
            try:
                opcao = self.mostrar_menu_principal()
                if opcao == '1':
                    self.carregar_dados()
                elif opcao == '2':
                    self.analisar_dados_ausentes()
                elif opcao == '3':
                    self.analisar_dados_duplicados()
                elif opcao == '4':
                    self.analisar_inconsistencias_ia()
                elif opcao == '5':
                    self.corrigir_problemas_automatico()
                elif opcao == '6':
                    self.relatorio_completo()
                elif opcao == '7':
                    self.exportar_analises()
                elif opcao == '8':
                    self.menu_llm_gemini()
                elif opcao == '0':
                    print("\n👋 Saindo do sistema...")
                    break
                else:
                    print("❌ Opção inválida! Tente novamente.")
                input("\n⏸️ Pressione ENTER para continuar...")
            except KeyboardInterrupt:
                print("\n\n👋 Sistema encerrado pelo usuário.")
                break
            except Exception as e:
                print(f"\n❌ Erro inesperado: {e}")
                input("⏸️ Pressione ENTER para continuar...")
    
    def mostrar_menu_principal(self) -> str:
        """Mostra o menu principal e retorna a opção escolhida"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("🤖 SISTEMA VR - MENU PRINCIPAL")
        print("=" * 50)
        
        # Status dos dados
        if self.bases_dados:
            print(f"📊 Status: {len(self.bases_dados)} bases carregadas")
            if self.relatorio_ia:
                score = self.relatorio_ia.score_integridade
                emoji = "🟢" if score > 80 else "🟡" if score > 50 else "🔴"
                print(f"🎯 Integridade: {emoji} {score:.1f}%")
        else:
            print("📊 Status: Nenhuma base carregada")
        
        print("\n📋 OPÇÕES DISPONÍVEIS:")
        print("1️⃣  Carregar Dados")
        print("2️⃣  Analisar Dados Ausentes (NaN)")
        print("3️⃣  Analisar Dados Duplicados")
        print("4️⃣  Análise com IA (Inconsistências)")
        print("5️⃣  Correção Automática")
        print("6️⃣  Relatório Completo")
        print("7️⃣  Exportar Análises")
        print("8️⃣  Análise Inteligente com LLM (Gemini)")
        print("0️⃣  Sair")
        
        print("\n" + "="*50)
        return input("🔍 Escolha uma opção: ").strip()

    def menu_llm_gemini(self):
        """Menu para análise inteligente com LLM (Gemini)"""
        try:
            from modules.agente_inteligente import criar_agente_vr
                elif modo == '3':
                    # Sugestão IA REAL: ler colunas/abas e consultar Gemini
                    print("\n🤖 Buscando sugestão de arquivo com Gemini...")
                    arquivos = list(Path("data/input").glob("*.csv")) + list(Path("data/input").glob("*.xlsx"))
                    if not arquivos:
                        print("❌ Nenhum arquivo encontrado em data/input")
                        continue
                    arquivos_info = []
                    import pandas as pd
                    for arq in arquivos:
                        info = {'nome': arq.name, 'tipo': arq.suffix, 'colunas': [], 'abas': []}
                        try:
                            if arq.suffix.lower() == '.csv':
                                df = pd.read_csv(arq, nrows=1)
                                info['colunas'] = list(df.columns)
                            elif arq.suffix.lower() in ['.xlsx', '.xls']:
                                xls = pd.ExcelFile(arq)
                                info['abas'] = xls.sheet_names
                                # Pega colunas da primeira aba
                                df = pd.read_excel(xls, sheet_name=xls.sheet_names[0], nrows=1)
                                info['colunas'] = list(df.columns)
                        except Exception as e:
                            info['erro'] = str(e)
                        arquivos_info.append(info)

                    # Montar prompt para Gemini
                    prompt = """
                print("❌ Opção inválida!")

    def _exibir_resposta_llm(self, resposta):
                    for info in arquivos_info:
                        prompt += f"Arquivo: {info['nome']}\n"
                        if info['abas']:
                            prompt += f"  Abas: {', '.join(info['abas'])}\n"
                        if info['colunas']:
                            prompt += f"  Colunas: {', '.join(info['colunas'])}\n"
                        if 'erro' in info:
                            prompt += f"  [ERRO leitura: {info['erro']}]\n"
                        prompt += "\n"
                    prompt += "\nResponda apenas o nome do arquivo mais indicado e uma frase de justificativa."

                    # Chamar Gemini
                    from modules.agente_inteligente import criar_agente_vr
                    agente = criar_agente_vr("gemini")
                    resposta = agente._chamar_llm(prompt)
                    print(f"🤖 Resposta do Gemini:\n{resposta}")
                    # Tentar extrair nome do arquivo sugerido
                    arquivo_sugerido = None
                    for info in arquivos_info:
                        if info['nome'].lower() in resposta.lower():
                            arquivo_sugerido = info
                            break
                    if arquivo_sugerido:
                        usar = input(f"Usar o arquivo sugerido '{arquivo_sugerido['nome']}'? (S/n): ").strip().lower()
                        if usar in ['', 's', 'sim', 'yes']:
                            from modules.data_reader import DataReader
                            caminho_arquivo = Path("data/input") / arquivo_sugerido['nome']
                            self.bases_dados = {'ativos': DataReader().read_file(str(caminho_arquivo))}
                            print(f"✅ Arquivo carregado: {caminho_arquivo}")
                            print("\n🤖 Executando análise inicial com IA...")
                            self.relatorio_ia, self.bases_corrigidas = analisar_fraudes_vr(
                                self.bases_dados, aplicar_correcoes=False
                            )
                            print(f"🎯 Score de Integridade: {self.relatorio_ia.score_integridade:.1f}%")
                            print(f"⚠️ {self.relatorio_ia.total_inconsistencias} inconsistências detectadas")
                            break
                        else:
                            print("Operação cancelada.")
                            continue
                    else:
                        print("❌ Não foi possível identificar sugestão do Gemini.")
                        continue
                elif modo == '0':
                    print("\n👋 Saindo do sistema...")
                    return
                else:
                    print("❌ Opção inválida! Tente novamente.")
                    continue
        print("\n📊 Resultado da LLM (Gemini):")
        print(f"Tipo: {resposta.tipo_resposta}")
        print(f"Confiança: {resposta.confianca:.1%}")
        print(f"\n💬 Análise:")
        print(resposta.conteudo)
        if resposta.acoes_sugeridas:
            print(f"\n✅ Ações sugeridas:")
            for acao in resposta.acoes_sugeridas:
                print(f"• {acao}")
    
    def carregar_dados(self):
        """Carrega os dados do sistema"""
        print("\n📚 CARREGANDO DADOS...")
        print("-" * 30)
        
        try:
            self.bases_dados = carregar_bases_vr("data/input")
            
            print(f"✅ {len(self.bases_dados)} bases carregadas com sucesso!")
            print("\n📊 RESUMO DAS BASES:")
            
            for nome, df in self.bases_dados.items():
                print(f"   📋 {nome}: {len(df)} registros, {len(df.columns)} colunas")
            
            # Análise automática inicial
            print("\n🤖 Executando análise inicial com IA...")
            self.relatorio_ia, self.bases_corrigidas = analisar_fraudes_vr(
                self.bases_dados, aplicar_correcoes=False
            )
            
            print(f"🎯 Score de Integridade: {self.relatorio_ia.score_integridade:.1f}%")
            print(f"⚠️ {self.relatorio_ia.total_inconsistencias} inconsistências detectadas")
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
    
    def analisar_dados_ausentes(self):
        """Analisa dados ausentes (NaN) de forma detalhada"""
        if not self.bases_dados:
            print("❌ Nenhuma base carregada! Execute a opção 1 primeiro.")
            return
        
        while True:
            print("\n🔍 ANÁLISE DE DADOS AUSENTES (NaN)")
            print("=" * 40)
            
            # Mostrar resumo geral
            self._mostrar_resumo_nan()
            
            print("\n📋 OPÇÕES:")
            print("1️⃣  Ver detalhes por base")
            print("2️⃣  Ver dados ausentes por matrícula")
            print("3️⃣  Analisar padrões de ausência")
            print("4️⃣  Sugerir correções")
            print("0️⃣  Voltar ao menu principal")
            
            opcao = input("\n🔍 Escolha uma opção: ").strip()
            
            if opcao == '1':
                self._detalhar_nan_por_base()
            elif opcao == '2':
                self._analisar_nan_por_matricula()
            elif opcao == '3':
                self._analisar_padroes_ausencia()
            elif opcao == '4':
                self._sugerir_correcoes_nan()
            elif opcao == '0':
                break
            else:
                print("❌ Opção inválida!")
            
            input("\n⏸️ Pressione ENTER para continuar...")
    
    def _mostrar_resumo_nan(self):
        """Mostra resumo geral de dados ausentes"""
        total_nan = 0
        total_registros = 0
        
        print("\n📊 RESUMO GERAL DE DADOS AUSENTES:")
        for nome_base, df in self.bases_dados.items():
            nan_count = df.isnull().sum().sum()
            total_count = df.size
            percentual = (nan_count / total_count) * 100 if total_count > 0 else 0
            
            emoji = "🔴" if percentual > 10 else "🟡" if percentual > 5 else "🟢"
            print(f"   {emoji} {nome_base}: {nan_count} NaN ({percentual:.1f}%)")
            
            total_nan += nan_count
            total_registros += total_count
        
        percentual_geral = (total_nan / total_registros) * 100 if total_registros > 0 else 0
        print(f"\n🎯 TOTAL GERAL: {total_nan} NaN de {total_registros} células ({percentual_geral:.1f}%)")
    
    def _detalhar_nan_por_base(self):
        """Detalha dados ausentes por base"""
        print("\n📋 ESCOLHA UMA BASE PARA ANÁLISE DETALHADA:")
        
        bases_lista = list(self.bases_dados.keys())
        for i, nome in enumerate(bases_lista, 1):
            print(f"   {i}️⃣  {nome}")
        
        try:
            escolha = int(input("\n🔍 Número da base: ")) - 1
            if 0 <= escolha < len(bases_lista):
                nome_base = bases_lista[escolha]
                df = self.bases_dados[nome_base]
                
                print(f"\n📊 ANÁLISE DETALHADA - BASE: {nome_base}")
                print("-" * 50)
                
                # Análise por coluna
                nan_por_coluna = df.isnull().sum()
                nan_com_dados = nan_por_coluna[nan_por_coluna > 0]
                
                if len(nan_com_dados) == 0:
                    print("✅ Esta base não possui dados ausentes!")
                else:
                    print("🔍 DADOS AUSENTES POR COLUNA:")
                    for coluna, count in nan_com_dados.items():
                        percentual = (count / len(df)) * 100
                        emoji = "🔴" if percentual > 50 else "🟡" if percentual > 20 else "🟠"
                        print(f"   {emoji} {coluna}: {count} NaN ({percentual:.1f}%)")
                    
                    # Perguntar se quer ver registros específicos
                    if input("\n❓ Ver registros com dados ausentes? (s/n): ").lower() == 's':
                        self._mostrar_registros_nan(df, nome_base)
            else:
                print("❌ Número inválido!")
        except ValueError:
            print("❌ Por favor, digite um número válido!")
    
    def _analisar_nan_por_matricula(self):
        """Analisa dados ausentes agrupados por matrícula"""
        print("\n👤 ANÁLISE DE DADOS AUSENTES POR MATRÍCULA")
        print("-" * 45)
        
        matriculas_com_nan = {}
        
        for nome_base, df in self.bases_dados.items():
            if 'matricula' in df.columns:
                # Encontrar registros com NaN
                registros_nan = df[df.isnull().any(axis=1)]
                
                for _, row in registros_nan.iterrows():
                    matricula = str(row['matricula'])
                    if matricula not in matriculas_com_nan:
                        matriculas_com_nan[matricula] = {}
                    
                    # Contar NaN por matrícula
                    nan_count = row.isnull().sum()
                    matriculas_com_nan[matricula][nome_base] = {
                        'nan_count': nan_count,
                        'colunas_nan': row[row.isnull()].index.tolist()
                    }
        
        if not matriculas_com_nan:
            print("✅ Nenhuma matrícula encontrada com dados ausentes!")
            return
        
        # Mostrar top 10 matrículas com mais problemas
        matriculas_ordenadas = sorted(
            matriculas_com_nan.items(),
            key=lambda x: sum(dados['nan_count'] for dados in x[1].values()),
            reverse=True
        )
        
        print(f"🎯 TOP 10 MATRÍCULAS COM MAIS DADOS AUSENTES:")
        for i, (matricula, dados) in enumerate(matriculas_ordenadas[:10], 1):
            total_nan = sum(dados['nan_count'] for dados in dados.values())
            bases_afetadas = list(dados.keys())
            print(f"   {i:2d}. Matrícula {matricula}: {total_nan} NaN em {bases_afetadas}")
        
        # Perguntar se quer ver detalhes de uma matrícula específica
        if input("\n❓ Ver detalhes de uma matrícula específica? (s/n): ").lower() == 's':
            matricula_busca = input("🔍 Digite a matrícula: ").strip()
            if matricula_busca in matriculas_com_nan:
                self._detalhar_matricula_nan(matricula_busca, matriculas_com_nan[matricula_busca])
            else:
                print(f"❌ Matrícula '{matricula_busca}' não encontrada com dados ausentes.")
    
    def _detalhar_matricula_nan(self, matricula: str, dados: Dict):
        """Mostra detalhes de dados ausentes para uma matrícula específica"""
        print(f"\n👤 DETALHES - MATRÍCULA: {matricula}")
        print("-" * 40)
        
        for nome_base, info in dados.items():
            print(f"\n📋 Base: {nome_base}")
            print(f"   🔢 Dados ausentes: {info['nan_count']}")
            print(f"   📝 Colunas afetadas: {', '.join(info['colunas_nan'])}")
    
    def _analisar_padroes_ausencia(self):
        """Analisa padrões de ausência usando IA"""
        print("\n🤖 ANÁLISE DE PADRÕES COM IA")
        print("-" * 35)
        
        # Usar IA para detectar padrões
        padroes_detectados = self.analyzer._detectar_padroes_ausencia_ia(self.bases_dados)
        
        if padroes_detectados:
            print("🔍 PADRÕES DETECTADOS:")
            for i, padrao in enumerate(padroes_detectados, 1):
                print(f"   {i}. {padrao}")
        else:
            print("✅ Nenhum padrão suspeito de ausência detectado.")
    
    def _sugerir_correcoes_nan(self):
        """Sugere correções para dados ausentes"""
        print("\n💡 SUGESTÕES DE CORREÇÃO PARA DADOS AUSENTES")
        print("-" * 50)
        
        sugestoes = []
        
        for nome_base, df in self.bases_dados.items():
            nan_por_coluna = df.isnull().sum()
            nan_com_dados = nan_por_coluna[nan_por_coluna > 0]
            
            for coluna, count in nan_com_dados.items():
                percentual = (count / len(df)) * 100
                
                if percentual < 5:
                    sugestoes.append(f"✅ {nome_base}.{coluna}: Preencher manualmente ({count} registros)")
                elif percentual < 20:
                    if df[coluna].dtype in ['int64', 'float64']:
                        sugestoes.append(f"🔢 {nome_base}.{coluna}: Substituir pela mediana/média")
                    else:
                        sugestoes.append(f"📝 {nome_base}.{coluna}: Preencher com valor padrão")
                else:
                    sugestoes.append(f"⚠️ {nome_base}.{coluna}: CRÍTICO - Verificar fonte dos dados")
        
        if sugestoes:
            for sugestao in sugestoes:
                print(f"   {sugestao}")
        else:
            print("✅ Nenhuma correção necessária!")
    
    def _mostrar_registros_nan(self, df: pd.DataFrame, nome_base: str):
        """Mostra registros específicos com dados ausentes"""
        registros_nan = df[df.isnull().any(axis=1)]
        
        if len(registros_nan) == 0:
            print("✅ Nenhum registro com dados ausentes encontrado!")
            return
        
        print(f"\n📋 REGISTROS COM DADOS AUSENTES ({len(registros_nan)} encontrados):")
        print("-" * 60)
        
        # Mostrar apenas os primeiros 10 para não sobrecarregar
        for i, (idx, row) in enumerate(registros_nan.head(10).iterrows(), 1):
            print(f"\n🔍 Registro {i} (Índice: {idx}):")
            
            # Mostrar apenas colunas com identificação + colunas com NaN
            colunas_mostrar = []
            if 'matricula' in row.index:
                colunas_mostrar.append('matricula')
            if 'nome' in row.index:
                colunas_mostrar.append('nome')
            
            # Adicionar colunas com NaN
            colunas_nan = row[row.isnull()].index.tolist()
            colunas_mostrar.extend(colunas_nan)
            
            for coluna in set(colunas_mostrar):
                valor = row[coluna]
                if pd.isnull(valor):
                    print(f"   ❌ {coluna}: [DADOS AUSENTES]")
                else:
                    print(f"   ✅ {coluna}: {valor}")
        
        if len(registros_nan) > 10:
            print(f"\n... e mais {len(registros_nan) - 10} registros.")
    
    def analisar_dados_duplicados(self):
        """Analisa dados duplicados"""
        if not self.bases_dados:
            print("❌ Nenhuma base carregada! Execute a opção 1 primeiro.")
            return
        
        while True:
            print("\n🔄 ANÁLISE DE DADOS DUPLICADOS")
            print("=" * 35)
            
            # Mostrar resumo geral
            self._mostrar_resumo_duplicados()
            
            print("\n📋 OPÇÕES:")
            print("1️⃣  Ver detalhes por base")
            print("2️⃣  Ver duplicatas por matrícula")
            print("3️⃣  Analisar duplicatas cruzadas")
            print("4️⃣  Sugerir remoção de duplicatas")
            print("0️⃣  Voltar ao menu principal")
            
            opcao = input("\n🔍 Escolha uma opção: ").strip()
            
            if opcao == '1':
                self._detalhar_duplicados_por_base()
            elif opcao == '2':
                self._analisar_duplicados_por_matricula()
            elif opcao == '3':
                self._analisar_duplicatas_cruzadas()
            elif opcao == '4':
                self._sugerir_remocao_duplicatas()
            elif opcao == '0':
                break
            else:
                print("❌ Opção inválida!")
            
            input("\n⏸️ Pressione ENTER para continuar...")
    
    def _mostrar_resumo_duplicados(self):
        """Mostra resumo geral de dados duplicados"""
        print("\n📊 RESUMO GERAL DE DUPLICATAS:")
        
        for nome_base, df in self.bases_dados.items():
            if 'matricula' in df.columns:
                duplicatas = df[df.duplicated(subset=['matricula'], keep=False)]
                matriculas_duplicadas = df['matricula'].duplicated().sum()
                
                emoji = "🔴" if matriculas_duplicadas > 0 else "🟢"
                print(f"   {emoji} {nome_base}: {matriculas_duplicadas} duplicatas de matrícula")
                
                # Duplicatas completas
                duplicatas_completas = df.duplicated().sum()
                if duplicatas_completas > 0:
                    print(f"      🔄 Registros completamente duplicados: {duplicatas_completas}")
            else:
                print(f"   ⚪ {nome_base}: Sem coluna 'matricula' para análise")
    
    def _detalhar_duplicados_por_base(self):
        """Detalha duplicados por base específica"""
        print("\n📋 ESCOLHA UMA BASE PARA ANÁLISE DETALHADA:")
        
        bases_lista = list(self.bases_dados.keys())
        for i, nome in enumerate(bases_lista, 1):
            print(f"   {i}️⃣  {nome}")
        
        try:
            escolha = int(input("\n🔍 Número da base: ")) - 1
            if 0 <= escolha < len(bases_lista):
                nome_base = bases_lista[escolha]
                df = self.bases_dados[nome_base]
                
                if 'matricula' not in df.columns:
                    print(f"❌ Base '{nome_base}' não possui coluna 'matricula'!")
                    return
                
                duplicatas = df[df.duplicated(subset=['matricula'], keep=False)]
                
                if len(duplicatas) == 0:
                    print(f"✅ Base '{nome_base}' não possui duplicatas de matrícula!")
                else:
                    print(f"\n🔄 DUPLICATAS ENCONTRADAS EM '{nome_base}': {len(duplicatas)} registros")
                    print("-" * 60)
                    
                    # Agrupar por matrícula
                    matriculas_dup = duplicatas.groupby('matricula')
                    
                    for matricula, grupo in matriculas_dup:
                        print(f"\n👤 Matrícula: {matricula} ({len(grupo)} ocorrências)")
                        for i, (idx, row) in enumerate(grupo.iterrows(), 1):
                            print(f"   🔍 Ocorrência {i} (Índice: {idx}):")
                            # Mostrar principais campos
                            for col in ['nome', 'admissao', 'situacao', 'valor'][:3]:
                                if col in row.index:
                                    print(f"      {col}: {row[col]}")
            else:
                print("❌ Número inválido!")
        except ValueError:
            print("❌ Por favor, digite um número válido!")
    
    def _analisar_duplicados_por_matricula(self):
        """Analisa duplicados agrupados por matrícula"""
        print("\n👤 DUPLICATAS POR MATRÍCULA (TODAS AS BASES)")
        print("-" * 45)
        
        all_matriculas = {}
        
        # Coletar todas as matrículas de todas as bases
        for nome_base, df in self.bases_dados.items():
            if 'matricula' in df.columns:
                for _, row in df.iterrows():
                    matricula = str(row['matricula'])
                    if matricula not in all_matriculas:
                        all_matriculas[matricula] = []
                    all_matriculas[matricula].append((nome_base, row.to_dict()))
        
        # Encontrar matrículas que aparecem em múltiplas bases ou múltiplas vezes
        matriculas_duplicadas = {
            mat: ocorrencias for mat, ocorrencias in all_matriculas.items()
            if len(ocorrencias) > 1
        }
        
        if not matriculas_duplicadas:
            print("✅ Nenhuma matrícula duplicada encontrada!")
            return
        
        print(f"🔍 {len(matriculas_duplicadas)} matrículas com duplicatas:")
        
        for matricula, ocorrencias in list(matriculas_duplicadas.items())[:10]:
            bases_envolvidas = [oc[0] for oc in ocorrencias]
            print(f"   👤 {matricula}: {len(ocorrencias)} ocorrências em {set(bases_envolvidas)}")
        
        if len(matriculas_duplicadas) > 10:
            print(f"   ... e mais {len(matriculas_duplicadas) - 10} matrículas.")
    
    def _analisar_duplicatas_cruzadas(self):
        """Analisa duplicatas entre diferentes bases"""
        print("\n🔄 ANÁLISE DE DUPLICATAS CRUZADAS")
        print("-" * 40)
        
        bases_com_matricula = {
            nome: df for nome, df in self.bases_dados.items()
            if 'matricula' in df.columns
        }
        
        if len(bases_com_matricula) < 2:
            print("❌ Necessário pelo menos 2 bases com coluna 'matricula'!")
            return
        
        # Analisar intersecções entre bases
        bases_lista = list(bases_com_matricula.keys())
        
        for i in range(len(bases_lista)):
            for j in range(i + 1, len(bases_lista)):
                base1, base2 = bases_lista[i], bases_lista[j]
                
                matriculas1 = set(bases_com_matricula[base1]['matricula'].astype(str))
                matriculas2 = set(bases_com_matricula[base2]['matricula'].astype(str))
                
                intersecao = matriculas1.intersection(matriculas2)
                
                if intersecao:
                    print(f"\n🔄 {base1} ↔ {base2}:")
                    print(f"   📊 {len(intersecao)} matrículas em comum")
                    
                    if len(intersecao) <= 5:
                        print(f"   👥 Matrículas: {', '.join(sorted(intersecao))}")
                    else:
                        primeiras = sorted(list(intersecao))[:3]
                        print(f"   👥 Exemplos: {', '.join(primeiras)}... (+{len(intersecao)-3})")
    
    def _sugerir_remocao_duplicatas(self):
        """Sugere estratégias para remoção de duplicatas"""
        print("\n💡 SUGESTÕES PARA REMOÇÃO DE DUPLICATAS")
        print("-" * 45)
        
        sugestoes = []
        
        for nome_base, df in self.bases_dados.items():
            if 'matricula' in df.columns:
                duplicatas_matricula = df['matricula'].duplicated().sum()
                duplicatas_completas = df.duplicated().sum()
                
                if duplicatas_completas > 0:
                    sugestoes.append(
                        f"🔄 {nome_base}: Remover {duplicatas_completas} registros completamente duplicados"
                    )
                
                if duplicatas_matricula > duplicatas_completas:
                    diff = duplicatas_matricula - duplicatas_completas
                    sugestoes.append(
                        f"⚠️ {nome_base}: {diff} matrículas duplicadas com dados diferentes - REVISAR MANUALMENTE"
                    )
        
        if sugestoes:
            for sugestao in sugestoes:
                print(f"   {sugestao}")
            
            print(f"\n🤖 SUGESTÃO DA IA:")
            print(f"   1. Remover duplicatas completas automaticamente")
            print(f"   2. Para matrículas com dados diferentes: manter registro mais recente")
            print(f"   3. Investigar fonte dos dados para prevenir futuras duplicatas")
        else:
            print("✅ Nenhuma duplicata encontrada!")
    
    def analisar_inconsistencias_ia(self):
        """Análise completa com IA de inconsistências"""
        if not self.bases_dados:
            print("❌ Nenhuma base carregada! Execute a opção 1 primeiro.")
            return
        
        print("\n🤖 ANÁLISE AVANÇADA COM IA")
        print("=" * 35)
        
        if not self.relatorio_ia:
            print("🔄 Executando análise completa com IA...")
            self.relatorio_ia, self.bases_corrigidas = analisar_fraudes_vr(
                self.bases_dados, aplicar_correcoes=False
            )
        
        # Mostrar resultado
        print(f"\n📊 RESULTADO DA ANÁLISE:")
        print(f"   🎯 Score de Integridade: {self.relatorio_ia.score_integridade:.1f}%")
        print(f"   ⚠️ Total de Inconsistências: {self.relatorio_ia.total_inconsistencias}")
        print(f"   🔴 Críticas: {self.relatorio_ia.inconsistencias_criticas}")
        print(f"   🔧 Corrigíveis: {self.relatorio_ia.inconsistencias_corrigidas}")
        
        # Mostrar tipos de inconsistências
        tipos_count = {}
        for inc in self.relatorio_ia.detalhes:
            tipos_count[inc.tipo] = tipos_count.get(inc.tipo, 0) + 1
        
                elif modo == '3':
                    # Sugestão IA REAL: ler colunas/abas e consultar Gemini
                    print("\n🤖 Buscando sugestão de arquivo com Gemini...")
                    arquivos = list(Path("data/input").glob("*.csv")) + list(Path("data/input").glob("*.xlsx"))
                    if not arquivos:
                        print("❌ Nenhum arquivo encontrado em data/input")
                        continue
                    arquivos_info = []
                    import pandas as pd
                    for arq in arquivos:
                        info = {'nome': arq.name, 'tipo': arq.suffix, 'colunas': [], 'abas': []}
                        try:
                            if arq.suffix.lower() == '.csv':
                                df = pd.read_csv(arq, nrows=1)
                                info['colunas'] = list(df.columns)
                            elif arq.suffix.lower() in ['.xlsx', '.xls']:
                                xls = pd.ExcelFile(arq)
                                info['abas'] = xls.sheet_names
                                # Pega colunas da primeira aba
                                df = pd.read_excel(xls, sheet_name=xls.sheet_names[0], nrows=1)
                                info['colunas'] = list(df.columns)
                        except Exception as e:
                            info['erro'] = str(e)
                        arquivos_info.append(info)

                    # Montar prompt para Gemini
                    prompt = """
                print(f"\n{emoji} {gravidade} ({len(inconsistencias)} itens):")
                
                for i, inc in enumerate(inconsistencias[:5], 1):  # Máximo 5 por categoria
                    for info in arquivos_info:
                        prompt += f"Arquivo: {info['nome']}\n"
                        elif modo == '3':
                            # Sugestão IA REAL: ler colunas/abas e consultar Gemini
                            print("\n🤖 Buscando sugestão de arquivo com Gemini...")
                            arquivos = list(Path("data/input").glob("*.csv")) + list(Path("data/input").glob("*.xlsx"))
                            if not arquivos:
                                print("❌ Nenhum arquivo encontrado em data/input")
                                continue
                            arquivos_info = []
                            import pandas as pd
                            for arq in arquivos:
                                info = {'nome': arq.name, 'tipo': arq.suffix, 'colunas': [], 'abas': []}
                                try:
                                    if arq.suffix.lower() == '.csv':
                                        df = pd.read_csv(arq, nrows=1)
                                        info['colunas'] = list(df.columns)
                                    elif arq.suffix.lower() in ['.xlsx', '.xls']:
                                        xls = pd.ExcelFile(arq)
                                        info['abas'] = xls.sheet_names
                                        # Pega colunas da primeira aba
                                        df = pd.read_excel(xls, sheet_name=xls.sheet_names[0], nrows=1)
                                        info['colunas'] = list(df.columns)
                                except Exception as e:
                                    info['erro'] = str(e)
                                arquivos_info.append(info)

                            # Montar prompt para Gemini
                            prompt = """
                            print("Operação cancelada.")
                            continue
                    else:
                            for info in arquivos_info:
                                prompt += f"Arquivo: {info['nome']}\n"
                                if info['abas']:
                                    prompt += f"  Abas: {', '.join(info['abas'])}\n"
                                if info['colunas']:
                                    prompt += f"  Colunas: {', '.join(info['colunas'])}\n"
                                if 'erro' in info:
                                    prompt += f"  [ERRO leitura: {info['erro']}]\n"
                                prompt += "\n"
                            prompt += "\nResponda apenas o nome do arquivo mais indicado e uma frase de justificativa."

                            # Chamar Gemini
                            from modules.agente_inteligente import criar_agente_vr
                            agente = criar_agente_vr("gemini")
                            resposta = agente._chamar_llm(prompt)
                            print(f"🤖 Resposta do Gemini:\n{resposta}")
                            # Tentar extrair nome do arquivo sugerido
                            arquivo_sugerido = None
                            for info in arquivos_info:
                                if info['nome'].lower() in resposta.lower():
                                    arquivo_sugerido = info
                                    break
                            if arquivo_sugerido:
                                usar = input(f"Usar o arquivo sugerido '{arquivo_sugerido['nome']}'? (S/n): ").strip().lower()
                                if usar in ['', 's', 'sim', 'yes']:
                                    from modules.data_reader import DataReader
                                    caminho_arquivo = Path("data/input") / arquivo_sugerido['nome']
                                    self.bases_dados = {'ativos': DataReader().read_file(str(caminho_arquivo))}
                                    print(f"✅ Arquivo carregado: {caminho_arquivo}")
                                    print("\n🤖 Executando análise inicial com IA...")
                                    self.relatorio_ia, self.bases_corrigidas = analisar_fraudes_vr(
                                        self.bases_dados, aplicar_correcoes=False
                                    )
                                    print(f"🎯 Score de Integridade: {self.relatorio_ia.score_integridade:.1f}%")
                                    print(f"⚠️ {self.relatorio_ia.total_inconsistencias} inconsistências detectadas")
                                    break
                                else:
                                    print("Operação cancelada.")
                                    continue
                        print("❌ Não foi possível identificar sugestão do Gemini.")
                        continue
                    matricula_str = f" (Mat: {inc.matricula})" if inc.matricula else ""
                    print(f"   {i}. {inc.detalhes}{matricula_str}")
                
                if len(inconsistencias) > 5:
                    print(f"   ... e mais {len(inconsistencias) - 5} itens.")
    
    def corrigir_problemas_automatico(self):
        """Sistema de correção automática"""
        if not self.bases_dados:
            print("❌ Nenhuma base carregada! Execute a opção 1 primeiro.")
            return
        
        print("\n🔧 SISTEMA DE CORREÇÃO AUTOMÁTICA")
        print("=" * 40)
        
        if not self.relatorio_ia:
            print("🔄 Executando análise para identificar problemas...")
            self.relatorio_ia, self.bases_corrigidas = analisar_fraudes_vr(
                self.bases_dados, aplicar_correcoes=False
            )
        
        corrigiveis = [inc for inc in self.relatorio_ia.detalhes if inc.corrigivel_automaticamente]
        
        if not corrigiveis:
            print("✅ Nenhum problema corrigível automaticamente encontrado!")
            return
        
        print(f"🔍 {len(corrigiveis)} problemas podem ser corrigidos automaticamente:")
        
        # Mostrar resumo do que será corrigido
        tipos_correcao = {}
        for inc in corrigiveis:
            tipos_correcao[inc.tipo] = tipos_correcao.get(inc.tipo, 0) + 1
        
        for tipo, count in tipos_correcao.items():
            print(f"   • {tipo.replace('_', ' ').title()}: {count} correções")
        
        # Perguntar se quer aplicar correções
        if input(f"\n❓ Aplicar {len(corrigiveis)} correções automáticas? (s/n): ").lower() == 's':
            print("\n🔄 Aplicando correções...")
            
            # Aplicar correções
            bases_corrigidas_novas, correcoes_aplicadas = self.analyzer.aplicar_correcoes_automaticas(
                self.bases_dados, corrigiveis
            )
            
            self.bases_corrigidas = bases_corrigidas_novas
            
            print(f"\n✅ {len(correcoes_aplicadas)} correções aplicadas:")
            for correcao in correcoes_aplicadas:
                print(f"   {correcao}")
            
            # Reanalizar após correções
            print("\n🔄 Reanalise após correções...")
            self.relatorio_ia, _ = analisar_fraudes_vr(
                self.bases_corrigidas, aplicar_correcoes=False
            )
            
            print(f"🎯 Novo Score de Integridade: {self.relatorio_ia.score_integridade:.1f}%")
            
            # Perguntar se quer salvar
            if input("\n❓ Salvar dados corrigidos? (s/n): ").lower() == 's':
                self._salvar_dados_corrigidos()
        else:
            print("❌ Correções não aplicadas.")
    
    def _salvar_dados_corrigidos(self):
        """Salva dados corrigidos"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pasta_output = Path("data/output/corrigidos")
        pasta_output.mkdir(parents=True, exist_ok=True)
        
        arquivos_salvos = []
        
        for nome_base, df in self.bases_corrigidas.items():
            arquivo = pasta_output / f"{nome_base}_corrigido_{timestamp}.xlsx"
            df.to_excel(arquivo, index=False)
            arquivos_salvos.append(str(arquivo))
        
        print(f"\n💾 {len(arquivos_salvos)} arquivos salvos em 'data/output/corrigidos/':")
        for arquivo in arquivos_salvos:
            print(f"   📄 {Path(arquivo).name}")
    
    def relatorio_completo(self):
        """Gera relatório completo de análise"""
        if not self.bases_dados:
            print("❌ Nenhuma base carregada! Execute a opção 1 primeiro.")
            return
        
        print("\n📋 RELATÓRIO COMPLETO DO SISTEMA")
        print("=" * 40)
        
        # Executar análise se não foi feita
        if not self.relatorio_ia:
            print("🔄 Executando análise completa...")
            self.relatorio_ia, self.bases_corrigidas = analisar_fraudes_vr(
                self.bases_dados, aplicar_correcoes=False
            )
        
        # Seção 1: Resumo das Bases
        print("\n📊 1. RESUMO DAS BASES DE DADOS")
        print("-" * 30)
        total_registros = sum(len(df) for df in self.bases_dados.values())
        print(f"   📋 Total de bases: {len(self.bases_dados)}")
        print(f"   📄 Total de registros: {total_registros}")
        
        for nome, df in self.bases_dados.items():
            print(f"   • {nome}: {len(df)} registros, {len(df.columns)} colunas")
        
        # Seção 2: Qualidade dos Dados
        print(f"\n🎯 2. QUALIDADE DOS DADOS")
        print("-" * 25)
        print(f"   Score de Integridade: {self.relatorio_ia.score_integridade:.1f}%")
        
        if self.relatorio_ia.score_integridade >= 80:
            print("   Status: 🟢 EXCELENTE")
        elif self.relatorio_ia.score_integridade >= 60:
            print("   Status: 🟡 BOM")
        else:
            print("   Status: 🔴 NECESSITA ATENÇÃO")
        
        # Seção 3: Problemas Identificados
        print(f"\n⚠️ 3. PROBLEMAS IDENTIFICADOS")
        print("-" * 28)
        print(f"   Total de inconsistências: {self.relatorio_ia.total_inconsistencias}")
        print(f"   Críticas: {self.relatorio_ia.inconsistencias_criticas}")
        print(f"   Corrigíveis automaticamente: {self.relatorio_ia.inconsistencias_corrigidas}")
        
        # Seção 4: Recomendações
        if self.relatorio_ia.recomendacoes:
            print(f"\n💡 4. RECOMENDAÇÕES")
            print("-" * 18)
            for i, rec in enumerate(self.relatorio_ia.recomendacoes, 1):
                print(f"   {i}. {rec}")
        
        # Perguntar se quer exportar
        if input(f"\n❓ Exportar relatório completo? (s/n): ").lower() == 's':
            self.exportar_analises()
    
    def exportar_analises(self):
        """Exporta análises para arquivos"""
        if not self.bases_dados:
            print("❌ Nenhuma base carregada! Execute a opção 1 primeiro.")
            return
        
        print("\n📤 EXPORTAÇÃO DE ANÁLISES")
        print("=" * 30)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pasta_output = Path("data/output/analises")
        pasta_output.mkdir(parents=True, exist_ok=True)
        
        arquivos_gerados = []
        
        try:
            # 1. Relatório de inconsistências
            if self.relatorio_ia:
                arquivo_inconsistencias = pasta_output / f"relatorio_inconsistencias_{timestamp}.txt"
                self._gerar_arquivo_relatorio(arquivo_inconsistencias)
                arquivos_gerados.append(arquivo_inconsistencias)
            
            # 2. Planilha com dados ausentes
            arquivo_nan = pasta_output / f"analise_dados_ausentes_{timestamp}.xlsx"
            self._gerar_planilha_nan(arquivo_nan)
            arquivos_gerados.append(arquivo_nan)
            
            # 3. Planilha com duplicatas
            arquivo_dup = pasta_output / f"analise_duplicatas_{timestamp}.xlsx"
            self._gerar_planilha_duplicatas(arquivo_dup)
            arquivos_gerados.append(arquivo_dup)
            
            # 4. Dashboard visual (se disponível)
            if self.relatorio_ia:
                dashboard = self.analyzer.gerar_dashboard_inconsistencias(self.relatorio_ia)
                if dashboard:
                    arquivos_gerados.append(dashboard)
            
            print(f"\n✅ {len(arquivos_gerados)} arquivos gerados:")
            for arquivo in arquivos_gerados:
                print(f"   📄 {Path(arquivo).name}")
            
            print(f"\n📁 Pasta: {pasta_output}")
            
        except Exception as e:
            print(f"❌ Erro ao exportar: {e}")
    
    def _gerar_arquivo_relatorio(self, arquivo: Path):
        """Gera arquivo de texto com relatório detalhado"""
        with open(arquivo, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO DE ANÁLISE DE DADOS - SISTEMA VR\n")
            f.write("=" * 50 + "\n")
            f.write(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
            
            # Resumo das bases
            f.write("1. RESUMO DAS BASES\n")
            f.write("-" * 20 + "\n")
            for nome, df in self.bases_dados.items():
                f.write(f"   {nome}: {len(df)} registros, {len(df.columns)} colunas\n")
            
            # Score de integridade
            f.write(f"\n2. QUALIDADE DOS DADOS\n")
            f.write("-" * 22 + "\n")
            f.write(f"   Score de Integridade: {self.relatorio_ia.score_integridade:.1f}%\n")
            
            # Detalhes das inconsistências
            f.write(f"\n3. INCONSISTÊNCIAS DETECTADAS\n")
            f.write("-" * 30 + "\n")
            for inc in self.relatorio_ia.detalhes:
                f.write(f"   {inc.gravidade}: {inc.detalhes}\n")
            
            # Recomendações
            f.write(f"\n4. RECOMENDAÇÕES\n")
            f.write("-" * 16 + "\n")
            for rec in self.relatorio_ia.recomendacoes:
                f.write(f"   • {rec}\n")
    
    def _gerar_planilha_nan(self, arquivo: Path):
        """Gera planilha com análise de dados ausentes"""
        with pd.ExcelWriter(arquivo, engine='openpyxl') as writer:
            # Aba com resumo
            resumo_data = []
            for nome_base, df in self.bases_dados.items():
                nan_por_coluna = df.isnull().sum()
                for coluna, nan_count in nan_por_coluna.items():
                    if nan_count > 0:
                        resumo_data.append({
                            'Base': nome_base,
                            'Coluna': coluna,
                            'Dados_Ausentes': nan_count,
                            'Total_Registros': len(df),
                            'Percentual': (nan_count / len(df)) * 100
                        })
            
            if resumo_data:
                pd.DataFrame(resumo_data).to_excel(writer, sheet_name='Resumo_NaN', index=False)
            
            # Aba para cada base com dados ausentes
            for nome_base, df in self.bases_dados.items():
                registros_nan = df[df.isnull().any(axis=1)]
                if len(registros_nan) > 0:
                    registros_nan.to_excel(writer, sheet_name=f'NaN_{nome_base}'[:31], index=False)
    
    def _gerar_planilha_duplicatas(self, arquivo: Path):
        """Gera planilha com análise de duplicatas"""
        with pd.ExcelWriter(arquivo, engine='openpyxl') as writer:
            for nome_base, df in self.bases_dados.items():
                if 'matricula' in df.columns:
                    duplicatas = df[df.duplicated(subset=['matricula'], keep=False)]
                    if len(duplicatas) > 0:
                        duplicatas.to_excel(writer, sheet_name=f'Dup_{nome_base}'[:31], index=False)


def main():
    """Função principal do menu interativo"""
    try:
        menu = MenuInterativoVR()
        menu.executar()
    except Exception as e:
        print(f"❌ Erro fatal: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
