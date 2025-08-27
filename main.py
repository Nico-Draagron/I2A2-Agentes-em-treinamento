from config import settings
import os
"""
Sistema Inteligente de Automação VR/VA
Autor: Agente VR 3.0
Data: 2025-08
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import logging
import argparse

import pandas as pd
from typing import Dict, List, Any, Optional

# Adicionar diretório ao path
sys.path.append(str(Path(__file__).parent))



# Importar módulos do projeto

from core.data_loader import DataLoader
from core.validator import DataValidator
from core.rules import BusinessRulesEngine
from core.calculator import VRCalculator
from core.report_generator import ReportGenerator
from agents.vr_agent import VRAgent
from agents.gemini import AssistenteGemini
from utils.logger import setup_logger
from utils.helpers import criar_estrutura_pastas, backup_arquivos


class SistemaVR:
    """Sistema principal de automação VR/VA"""
    
    def __init__(self, modo: str = "interativo", config: Optional[Dict] = None):
        """
        Inicializa o sistema VR
        
        Args:
            modo: 'interativo' ou 'automatico'
            config: Configurações personalizadas
        """
        self.modo = modo
        self.config = config or self._carregar_configuracoes()
        self.logger = setup_logger("SistemaVR")
        
        # Criar estrutura de pastas se não existir
        criar_estrutura_pastas()
        
        # Inicializar componentes
        self.data_loader = DataLoader()
        self.validator = DataValidator()
        self.rules_engine = BusinessRulesEngine()
        self.calculator = VRCalculator()
        self.report_generator = ReportGenerator()
        
        # Inicializar agentes inteligentes
        self.vr_agent = VRAgent()
        self.gemini = self._inicializar_gemini()
        
    
    def _carregar_configuracoes(self) -> Dict:
        """Carrega configurações do sistema"""
        config = {
            'competencia': datetime.now().strftime('%Y-%m'),
            'pasta_input': 'data/input',
            'pasta_output': 'data/output',
            'backup_enabled': True,
            'gemini_api_key': settings.GEMINI_API_KEY,
            'modo_debug': False
        }
        return config
    
    def _inicializar_gemini(self) -> Optional[AssistenteGemini]:
        """Inicializa assistente Gemini se disponível"""
        api_key = self.config.get('gemini_api_key')
        if not api_key:
            self.logger.warning("⚠️ Gemini API Key não configurada. IA limitada.")
            return None
        try:
            return AssistenteGemini(api_key)
        except Exception as e:
            self.logger.error(f"❌ Erro ao inicializar Gemini: {e}")
            return None
    
    def executar(self) -> bool:
        """
        Executa o pipeline completo do sistema VR
        
        Returns:
            True se executado com sucesso, False caso contrário
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("🤖 INICIANDO PROCESSAMENTO VR")
            self.logger.info(f"📅 Competência: {self.config['competencia']}")
            self.logger.info("=" * 60)
            
            # Pipeline principal
            sucesso = (
                self._etapa_1_carregar_dados() and
                self._etapa_2_validar_dados() and
                self._etapa_3_aplicar_regras() and
                self._etapa_4_calcular_vr() and
                self._etapa_5_gerar_relatorios()
            )
            
            if sucesso:
                self.logger.info("✅ PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
                self._exibir_resumo_final()
            else:
                self.logger.error("❌ PROCESSAMENTO FALHOU!")
            
            return sucesso
            
        except Exception as e:
            self.logger.error(f"❌ Erro fatal: {e}")
            return False
    
    def _etapa_1_carregar_dados(self) -> bool:
        """Etapa 1: Carregar todos os arquivos de entrada"""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("📂 ETAPA 1: CARREGANDO DADOS")
        self.logger.info("=" * 50)
        try:
            # Fazer backup se configurado
            if self.config.get('backup_enabled'):
                backup_arquivos(self.config['pasta_input'])
            # Carregar dados
            self.dados_carregados = self.data_loader.carregar_todas_bases()
            if not self.dados_carregados:
                self.logger.error("❌ Nenhum dado foi carregado!")
                return False
            self.logger.info("📊 Resumo dos dados carregados:")
            for nome_base, df in self.dados_carregados.items():
                self.logger.info(f"   ✅ {nome_base}: {len(df)} registros")
            # Análise inicial com Gemini se disponível
            if self.gemini:
                self.logger.info("🤖 Analisando dados com IA...")
                analise_ia = self.gemini.analisar_inconsistencias(self.dados_carregados)
                if analise_ia:
                    self.logger.info(f"   💡 Sugestão IA: {analise_ia}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar dados: {e}")
            return False
    
    def _etapa_2_validar_dados(self) -> bool:
        """Etapa 2: Validar e corrigir dados"""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("🔍 ETAPA 2: VALIDANDO DADOS")
        self.logger.info("=" * 50)
        try:
            # Validar dados
            self.dados_validados, inconsistencias = self.validator.validar_e_limpar_dados(self.dados_carregados)
            # Mostrar inconsistências encontradas
            if inconsistencias:
                self.logger.warning(f"⚠️ {len(inconsistencias)} inconsistências encontradas:")
                criticas = [i for i in inconsistencias if getattr(i, 'gravidade', None) == 'CRITICA']
                altas = [i for i in inconsistencias if getattr(i, 'gravidade', None) == 'ALTA']
                outras = [i for i in inconsistencias if getattr(i, 'gravidade', None) not in ['CRITICA', 'ALTA']]
                if criticas:
                    self.logger.error(f"   🔴 CRÍTICAS: {len(criticas)}")
                    for inc in criticas[:3]:
                        self.logger.error(f"      - {getattr(inc, 'descricao', inc)}")
                if altas:
                    self.logger.warning(f"   🟡 ALTAS: {len(altas)}")
                    for inc in altas[:3]:
                        self.logger.warning(f"      - {getattr(inc, 'descricao', inc)}")
                if outras:
                    self.logger.info(f"   🟢 OUTRAS: {len(outras)}")
                # Perguntar se deseja continuar com críticas
                if criticas and self.modo == "interativo":
                    resposta = input("\n❓ Existem erros críticos. Continuar? (s/n): ")
                    if resposta.lower() != 's':
                        return False
            else:
                self.logger.info("✅ Todos os dados estão válidos!")
            return True
        except Exception as e:
            self.logger.error(f"❌ Erro na validação: {e}")
            return False
    
    def _etapa_3_aplicar_regras(self) -> bool:
        """Etapa 3: Aplicar regras de negócio"""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("📋 ETAPA 3: APLICANDO REGRAS DE NEGÓCIO")
        self.logger.info("=" * 50)
        try:
            # Exemplo: processar elegibilidade completa (ajuste conforme seu core)
            resultados = self.rules_engine.processar_elegibilidade_completa(self.dados_validados)
            self.dados_validados = resultados  # ou ajuste conforme esperado nas próximas etapas
            self.logger.info(f"✅ Regras aplicadas com sucesso! Total processados: {len(resultados)}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Erro ao aplicar regras: {e}")
            return False
    
    def _etapa_4_calcular_vr(self) -> bool:
        """Etapa 4: Calcular valores de VR"""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("💰 ETAPA 4: CALCULANDO VR")
        self.logger.info("=" * 50)
        try:
            # Exemplo: calcular VR completo (ajuste conforme seu core)
            self.calculos = self.calculator.calcular_vr_completo(self.dados_validados, self.config['competencia'])
            self.logger.info(f"✅ Cálculo de VR realizado com sucesso!")
            return True
        except Exception as e:
            self.logger.error(f"❌ Erro nos cálculos: {e}")
            return False
    
    def _etapa_5_gerar_relatorios(self) -> bool:
        """Etapa 5: Gerar relatórios finais"""
        self.logger.info("\n" + "=" * 50)
        self.logger.info("📊 ETAPA 5: GERANDO RELATÓRIOS")
        self.logger.info("=" * 50)
        try:
            # Gerar planilha final (stub)
            arquivo_vr = self.report_generator.gerar_planilha_vr(self.calculos, self.config['competencia'], self.config['pasta_output'])
            self.logger.info(f"✅ Planilha VR gerada: {arquivo_vr}")
            self.relatorio_final = {
                'planilha_vr': arquivo_vr,
                'timestamp': datetime.now()
            }
            return True
        except Exception as e:
            self.logger.error(f"❌ Erro ao gerar relatórios: {e}")
            return False
    
    def _exibir_resumo_final(self):
        """Exibe resumo final do processamento"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📋 RESUMO FINAL DO PROCESSAMENTO")
        self.logger.info("=" * 60)
        
        # Estatísticas gerais
        stats = self.vr_agent.get_estatisticas_completas()
        
        self.logger.info("📊 Estatísticas Gerais:")
        self.logger.info(f"   📂 Arquivos processados: {stats['arquivos_processados']}")
        self.logger.info(f"   👥 Colaboradores analisados: {stats['colaboradores_total']}")
        self.logger.info(f"   ✅ Elegíveis para VR: {stats['colaboradores_elegiveis']}")
        self.logger.info(f"   ❌ Excluídos: {stats['colaboradores_excluidos']}")
        
        self.logger.info("\n💰 Valores Calculados:")
        self.logger.info(f"   💵 Total VR: R$ {stats['valor_total']:,.2f}")
        self.logger.info(f"   🏢 Empresa: R$ {stats['valor_empresa']:,.2f}")
        self.logger.info(f"   👤 Colaboradores: R$ {stats['valor_colaboradores']:,.2f}")
        
        self.logger.info("\n📁 Arquivos Gerados:")
        for nome, caminho in self.relatorio_final.items():
            if nome != 'timestamp':
                self.logger.info(f"   📄 {nome}: {caminho}")
        
        self.logger.info("\n✨ Processamento concluído com sucesso!")
        self.logger.info(f"⏱️ Tempo total: {stats.get('tempo_processamento', 'N/A')}")
    
    def executar_interativo(self):
        """Executa o sistema em modo interativo com menu"""
        from utils.menu import MenuInterativo
        menu = MenuInterativo(self)
        menu.executar()
    
    def executar_automatico(self):
        """Executa o sistema em modo automático"""
        self.logger.info("🤖 Executando em modo automático...")
        return self.executar()


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description='Sistema Inteligente de Automação VR/VA'
    )
    
    parser.add_argument(
        '--modo',
        choices=['interativo', 'automatico'],
        default='interativo',
        help='Modo de execução do sistema'
    )
    
    parser.add_argument(
        '--competencia',
        type=str,
        default=datetime.now().strftime('%Y-%m'),
        help='Competência no formato YYYY-MM'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/input',
        help='Pasta com arquivos de entrada'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/output',
        help='Pasta para arquivos de saída'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Ativar modo debug'
    )
    
    args = parser.parse_args()
    
    # Configurar sistema
    config = {
        'competencia': args.competencia,
        'pasta_input': args.input,
        'pasta_output': args.output,
        'modo_debug': args.debug,
        'gemini_api_key': settings.GEMINI_API_KEY
    }
    
    # Criar e executar sistema
    sistema = SistemaVR(modo=args.modo, config=config)
    
    if args.modo == 'interativo':
        sistema.executar_interativo()
    else:
        sucesso = sistema.executar_automatico()
        sys.exit(0 if sucesso else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Sistema encerrado pelo usuário.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")
        sys.exit(1)