# Aplicação das regras de negócio
"""
Módulo business_rules.py
Responsável por aplicar regras específicas de negócio do sistema VR.

Autor: Agente VR
Data: 2025-08
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set, Any
import logging
from dataclasses import dataclass
from enum import Enum

class TipoExclusao(Enum):
    """Tipos de exclusão de colaboradores"""
    DIRETOR = "diretor"
    ESTAGIARIO = "estagiario"
    APRENDIZ = "aprendiz"
    AFASTADO = "afastado"
    EXTERIOR = "exterior"
    DESLIGADO_ANTES_15 = "desligado_antes_15"
    SEM_SINDICATO = "sem_sindicato"
    CARGO_EXCLUSAO = "cargo_exclusao"

class StatusElegibilidade(Enum):
    """Status de elegibilidade para VR"""
    ELEGIVEL = "elegivel"
    EXCLUIDO = "excluido"
    DESLIGADO_SEM_DIREITO = "desligado_sem_direito"
    PROPORCIONAL = "proporcional"

@dataclass
class ResultadoElegibilidade:
    """Resultado da análise de elegibilidade de um colaborador"""
    matricula: int
    status: StatusElegibilidade
    tipo_exclusao: Optional[TipoExclusao]
    dias_elegivel: int
    dias_uteis_sindicato: int
    dias_ferias: int
    data_inicio_periodo: datetime
    data_fim_periodo: datetime
    valor_diario: float
    estado_sindicato: str
    observacoes: List[str]
    detalhes_calculo: Dict[str, Any]

class BusinessRulesEngine:
    """
    Motor de regras de negócio para determinar elegibilidade e cálculo de VR.
    """
    
    def __init__(self, data_competencia: str = "2025-05"):
        """
        Inicializa o motor de regras de negócio.
        
        Args:
            data_competencia: Competência no formato YYYY-MM
        """
        self.logger = logging.getLogger(__name__)
        self.data_competencia = datetime.strptime(f"{data_competencia}-01", "%Y-%m-%d")
        
        # Calcular período da competência (do dia 15 do mês anterior ao dia 15 do mês atual)
        if self.data_competencia.month == 1:
            mes_anterior = 12
            ano_anterior = self.data_competencia.year - 1
        else:
            mes_anterior = self.data_competencia.month - 1
            ano_anterior = self.data_competencia.year
        
        self.data_inicio_periodo = datetime(ano_anterior, mes_anterior, 15)
        self.data_fim_periodo = datetime(self.data_competencia.year, self.data_competencia.month, 15)
        
        # Dia 15 como corte para regra de desligamento
        self.dia_corte_desligamento = 15
        
        # Mapeamento de sindicatos para estados
        self.mapeamento_sindicato_estado = {
            'SINDPD SP': 'São Paulo',
            'SINDPD RJ': 'Rio de Janeiro',
            'SINDPPD RS': 'Rio Grande do Sul', 
            'SITEPD PR': 'Paraná'
        }
        
        # Cargos que devem ser excluídos
        self.cargos_exclusao = {
            'DIRETOR', 'DIRECTOR', 'CEO', 'CFO', 'CTO', 'COO',
            'PRESIDENTE', 'VICE-PRESIDENTE', 'VP'
        }
        
        # Estatísticas de processamento
        self.estatisticas = {
            'total_processados': 0,
            'elegiveis': 0,
            'excluidos_por_tipo': {},
            'com_ferias': 0,
            'proporcionais': 0
        }
    
    def processar_elegibilidade_completa(self, bases_validadas: Dict[str, Any]) -> List[ResultadoElegibilidade]:
        """
        Processa elegibilidade de todos os colaboradores aplicando todas as regras.
        
        Args:
            bases_validadas: Bases validadas pelo data_validator
            
        Returns:
            Lista com resultado de elegibilidade de cada colaborador
        """
        self.logger.info("📋 Iniciando processamento de regras de negócio...")
        
        try:
            # Preparar dados auxiliares
            configuracoes = self._preparar_configuracoes(bases_validadas)
            listas_exclusao = self._preparar_listas_exclusao(bases_validadas)
            
            # Processar cada colaborador ativo
            resultados = []
            colaboradores_ativos = bases_validadas['ativos']
            
            self.logger.info(f"   Processando {len(colaboradores_ativos)} colaboradores ativos...")
            
            for _, colaborador in colaboradores_ativos.iterrows():
                resultado = self._processar_colaborador_individual(
                    colaborador, 
                    bases_validadas,
                    configuracoes,
                    listas_exclusao
                )
                resultados.append(resultado)
                
                # Atualizar estatísticas
                self._atualizar_estatisticas(resultado)
            
            # Log do resumo
            self._log_resumo_processamento()
            
            return resultados
            
        except Exception as e:
            self.logger.error(f"❌ Erro no processamento de regras: {e}")
            raise
    
    def _preparar_configuracoes(self, bases: Dict[str, Any]) -> Dict[str, Any]:
        """Prepara configurações de sindicatos, valores e dias úteis"""
        self.logger.info("⚙️ Preparando configurações de sindicatos...")
        
        configuracoes = {
            'valores_por_estado': {},
            'dias_uteis_por_sindicato': {},
            'sindicato_para_estado': {}
        }
        
        # Valores por estado
        if 'sindicatos_valores' in bases:
            for _, row in bases['sindicatos_valores'].iterrows():
                estado = row['estado'].strip()
                valor = float(row['valor_diario'])
                configuracoes['valores_por_estado'][estado] = valor
        
        # Dias úteis por sindicato
        if 'dias_uteis' in bases:
            for _, row in bases['dias_uteis'].iterrows():
                nome_sindicato = row['sindicato'].strip()
                dias = int(row['dias_uteis'])
                
                # Mapear nome completo para sigla
                for sigla, estado in self.mapeamento_sindicato_estado.items():
                    if sigla in nome_sindicato:
                        configuracoes['dias_uteis_por_sindicato'][sigla] = dias
                        configuracoes['sindicato_para_estado'][sigla] = estado
                        break
        
        self.logger.info(f"   ✅ Configurações preparadas:")
        self.logger.info(f"      Estados com valor: {len(configuracoes['valores_por_estado'])}")
        self.logger.info(f"      Sindicatos com dias úteis: {len(configuracoes['dias_uteis_por_sindicato'])}")
        
        return configuracoes
    
    def _preparar_listas_exclusao(self, bases: Dict[str, Any]) -> Dict[str, Set[int]]:
        """Prepara listas de matrículas para exclusão"""
        self.logger.info("🚫 Preparando listas de exclusão...")
        
        listas = {
            'afastados': set(),
            'estagiarios': set(),
            'aprendizes': set(),
            'exterior': set(),
            'desligados': set()
        }
        
        # Afastados
        if 'exclusoes' in bases and 'afastamentos' in bases['exclusoes']:
            listas['afastados'] = set(bases['exclusoes']['afastamentos']['matricula'])
        
        # Estagiários
        if 'exclusoes' in bases and 'estagiarios' in bases['exclusoes']:
            listas['estagiarios'] = set(bases['exclusoes']['estagiarios']['matricula'])
        
        # Aprendizes
        if 'exclusoes' in bases and 'aprendizes' in bases['exclusoes']:
            listas['aprendizes'] = set(bases['exclusoes']['aprendizes']['matricula'])
        
        # Exterior
        if 'exclusoes' in bases and 'exterior' in bases['exclusoes']:
            listas['exterior'] = set(bases['exclusoes']['exterior']['matricula'])
        
        # Desligados
        if 'desligados' in bases:
            listas['desligados'] = set(bases['desligados']['matricula'])
        
        total_exclusoes = sum(len(lista) for lista in listas.values())
        self.logger.info(f"   ✅ Listas de exclusão preparadas: {total_exclusoes} matrículas")
        
        return listas
    
    def _processar_colaborador_individual(
        self, 
        colaborador: pd.Series,
        bases: Dict[str, Any],
        configuracoes: Dict[str, Any],
        listas_exclusao: Dict[str, Set[int]]
    ) -> ResultadoElegibilidade:
        """Processa elegibilidade de um colaborador individual"""
        
        matricula = int(colaborador['matricula'])
        observacoes = []
        detalhes_calculo = {}
        
        # Verificar exclusões básicas
        exclusao = self._verificar_exclusoes_basicas(matricula, colaborador, listas_exclusao)
        if exclusao:
            return self._criar_resultado_excluido(matricula, exclusao[0], exclusao[1], observacoes)
        
        # Verificar desligamento
        resultado_desligamento = self._verificar_regra_desligamento(matricula, bases)
        if resultado_desligamento['excluido']:
            return self._criar_resultado_excluido(
                matricula, 
                TipoExclusao.DESLIGADO_ANTES_15,
                resultado_desligamento['motivo'],
                observacoes
            )
        
        # Obter configurações do sindicato
        config_sindicato = self._obter_configuracao_sindicato(colaborador, configuracoes)
        if not config_sindicato['valido']:
            return self._criar_resultado_excluido(
                matricula,
                TipoExclusao.SEM_SINDICATO,
                config_sindicato['motivo'],
                observacoes
            )
        
        # Calcular período elegível
        periodo_elegivel = self._calcular_periodo_elegivel(matricula, bases, resultado_desligamento)
        
        # Calcular dias de férias no período
        dias_ferias = self._calcular_dias_ferias(matricula, bases, periodo_elegivel)
        
        # Calcular dias elegíveis
        dias_uteis_sindicato = config_sindicato['dias_uteis']
        dias_elegivel = max(0, dias_uteis_sindicato - dias_ferias)
        
        # Ajustar para período proporcional se necessário
        if periodo_elegivel['proporcional']:
            fator_proporcional = periodo_elegivel['dias_periodo'] / 30  # Assumindo mês de 30 dias
            dias_elegivel = int(dias_elegivel * fator_proporcional)
            observacoes.append(f"Cálculo proporcional aplicado (fator: {fator_proporcional:.2f})")
        
        # Detalhes do cálculo
        detalhes_calculo = {
            'dias_uteis_sindicato': dias_uteis_sindicato,
            'dias_ferias_periodo': dias_ferias,
            'periodo_inicio': periodo_elegivel['inicio'],
            'periodo_fim': periodo_elegivel['fim'],
            'proporcional': periodo_elegivel['proporcional'],
            'fator_proporcional': periodo_elegivel.get('fator_proporcional', 1.0)
        }
        
        # Adicionar observações específicas
        if dias_ferias > 0:
            observacoes.append(f"Desconto de {dias_ferias} dias por férias")
        
        if resultado_desligamento['proporcional']:
            observacoes.append(f"Desligamento após dia 15 - mantido proporcional")
        
        # Determinar status final
        status = StatusElegibilidade.ELEGIVEL
        if dias_elegivel == 0:
            status = StatusElegibilidade.EXCLUIDO
            observacoes.append("Zero dias elegíveis após aplicação das regras")
        elif periodo_elegivel['proporcional']:
            status = StatusElegibilidade.PROPORCIONAL
        
        return ResultadoElegibilidade(
            matricula=matricula,
            status=status,
            tipo_exclusao=None,
            dias_elegivel=dias_elegivel,
            dias_uteis_sindicato=dias_uteis_sindicato,
            dias_ferias=dias_ferias,
            data_inicio_periodo=periodo_elegivel['inicio'],
            data_fim_periodo=periodo_elegivel['fim'],
            valor_diario=config_sindicato['valor_diario'],
            estado_sindicato=config_sindicato['estado'],
            observacoes=observacoes,
            detalhes_calculo=detalhes_calculo
        )
    
    def _verificar_exclusoes_basicas(
        self, 
        matricula: int, 
        colaborador: pd.Series, 
        listas_exclusao: Dict[str, Set[int]]
    ) -> Optional[Tuple[TipoExclusao, str]]:
        """Verifica exclusões básicas (afastados, estagiários, etc.)"""
        
        # Verificar cargo de diretor
        cargo = str(colaborador.get('cargo', '')).upper()
        for cargo_exclusao in self.cargos_exclusao:
            if cargo_exclusao in cargo:
                return (TipoExclusao.DIRETOR, f"Cargo de diretor: {cargo}")
        
        # Verificar listas de exclusão
        if matricula in listas_exclusao['afastados']:
            return (TipoExclusao.AFASTADO, "Colaborador afastado")
        
        if matricula in listas_exclusao['estagiarios']:
            return (TipoExclusao.ESTAGIARIO, "Estagiário")
        
        if matricula in listas_exclusao['aprendizes']:
            return (TipoExclusao.APRENDIZ, "Aprendiz")
        
        if matricula in listas_exclusao['exterior']:
            return (TipoExclusao.EXTERIOR, "Colaborador no exterior")
        
        return None
    
    def _verificar_regra_desligamento(self, matricula: int, bases: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplica regra dos 15 dias para desligamentos.
        
        Regra: Se comunicado até dia 15 + OK → NÃO pagar
               Se comunicado após dia 15 → Pagar proporcional
        """
        resultado = {
            'excluido': False,
            'proporcional': False,
            'motivo': '',
            'data_desligamento': None
        }
        
        if 'desligados' not in bases:
            return resultado
        
        # Buscar colaborador na base de desligados
        desligado = bases['desligados'][bases['desligados']['matricula'] == matricula]
        
        if desligado.empty:
            return resultado
        
        desligado_info = desligado.iloc[0]
        data_demissao = desligado_info['data_demissao']
        comunicado_ok = desligado_info.get('comunicado_ok', '').upper()
        
        resultado['data_desligamento'] = data_demissao
        
        # Se não há data de demissão, não aplicar regra
        if pd.isna(data_demissao):
            return resultado
        
        data_demissao = pd.to_datetime(data_demissao)
        dia_demissao = data_demissao.day
        
        # Regra: comunicado até dia 15 + OK = não pagar
        if dia_demissao <= self.dia_corte_desligamento and comunicado_ok == 'OK':
            resultado['excluido'] = True
            resultado['motivo'] = f"Desligado dia {dia_demissao} com comunicado OK"
        else:
            # Após dia 15 ou sem OK = pagar proporcional
            resultado['proporcional'] = True
            resultado['motivo'] = f"Desligado dia {dia_demissao}, mantido proporcional"
        
        return resultado
    
    def _obter_configuracao_sindicato(self, colaborador: pd.Series, configuracoes: Dict[str, Any]) -> Dict[str, Any]:
        """Obtém configurações de valor e dias úteis do sindicato"""
        
        sindicato = str(colaborador.get('sindicato', ''))
        
        resultado = {
            'valido': False,
            'sigla_sindicato': '',
            'estado': '',
            'valor_diario': 0.0,
            'dias_uteis': 0,
            'motivo': ''
        }
        
        # Identificar sigla do sindicato
        sigla_encontrada = None
        for sigla in self.mapeamento_sindicato_estado.keys():
            if sigla in sindicato:
                sigla_encontrada = sigla
                break
        
        if not sigla_encontrada:
            resultado['motivo'] = f"Sindicato não reconhecido: {sindicato}"
            return resultado
        
        # Obter estado
        estado = self.mapeamento_sindicato_estado[sigla_encontrada]
        
        # Obter valor diário
        if estado not in configuracoes['valores_por_estado']:
            resultado['motivo'] = f"Estado {estado} sem valor diário configurado"
            return resultado
        
        # Obter dias úteis
        if sigla_encontrada not in configuracoes['dias_uteis_por_sindicato']:
            resultado['motivo'] = f"Sindicato {sigla_encontrada} sem dias úteis configurados"
            return resultado
        
        resultado.update({
            'valido': True,
            'sigla_sindicato': sigla_encontrada,
            'estado': estado,
            'valor_diario': configuracoes['valores_por_estado'][estado],
            'dias_uteis': configuracoes['dias_uteis_por_sindicato'][sigla_encontrada]
        })
        
        return resultado
    
    def _calcular_periodo_elegivel(self, matricula: int, bases: Dict[str, Any], resultado_desligamento: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula período elegível considerando admissão e desligamento"""
        
        inicio = self.data_inicio_periodo
        fim = self.data_fim_periodo
        proporcional = False
        
        # Verificar admissão no período
        if 'admissoes' in bases:
            admissao = bases['admissoes'][bases['admissoes']['matricula'] == matricula]
            if not admissao.empty:
                data_admissao = pd.to_datetime(admissao.iloc[0]['data_admissao'])
                if data_admissao > inicio:
                    inicio = data_admissao
                    proporcional = True
        
        # Verificar desligamento no período
        if resultado_desligamento['data_desligamento']:
            data_desligamento = pd.to_datetime(resultado_desligamento['data_desligamento'])
            if data_desligamento < fim:
                fim = data_desligamento
                proporcional = True
        
        dias_periodo = (fim - inicio).days
        
        return {
            'inicio': inicio,
            'fim': fim,
            'proporcional': proporcional,
            'dias_periodo': dias_periodo
        }
    
    def _calcular_dias_ferias(self, matricula: int, bases: Dict[str, Any], periodo_elegivel: Dict[str, Any]) -> int:
        """Calcula dias de férias no período elegível"""
        
        if 'ferias' not in bases:
            return 0
        
        ferias_colaborador = bases['ferias'][bases['ferias']['matricula'] == matricula]
        
        if ferias_colaborador.empty:
            return 0
        
        # Somar todos os dias de férias do colaborador
        total_dias_ferias = ferias_colaborador['dias_ferias'].sum()
        
        # Se o período é proporcional, aplicar proporção
        if periodo_elegivel['proporcional']:
            fator_proporcional = periodo_elegivel['dias_periodo'] / 30
            total_dias_ferias = int(total_dias_ferias * fator_proporcional)
        
        return total_dias_ferias
    
    def _criar_resultado_excluido(self, matricula: int, tipo_exclusao: TipoExclusao, motivo: str, observacoes: List[str]) -> ResultadoElegibilidade:
        """Cria resultado para colaborador excluído"""
        
        observacoes.append(f"EXCLUÍDO: {motivo}")
        
        return ResultadoElegibilidade(
            matricula=matricula,
            status=StatusElegibilidade.EXCLUIDO,
            tipo_exclusao=tipo_exclusao,
            dias_elegivel=0,
            dias_uteis_sindicato=0,
            dias_ferias=0,
            data_inicio_periodo=self.data_inicio_periodo,
            data_fim_periodo=self.data_fim_periodo,
            valor_diario=0.0,
            estado_sindicato='',
            observacoes=observacoes,
            detalhes_calculo={}
        )
    
    def _atualizar_estatisticas(self, resultado: ResultadoElegibilidade) -> None:
        """Atualiza estatísticas de processamento"""
        self.estatisticas['total_processados'] += 1
        
        if resultado.status == StatusElegibilidade.ELEGIVEL:
            self.estatisticas['elegiveis'] += 1
        elif resultado.status == StatusElegibilidade.PROPORCIONAL:
            self.estatisticas['proporcionais'] += 1
        
        if resultado.tipo_exclusao:
            tipo = resultado.tipo_exclusao.value
            self.estatisticas['excluidos_por_tipo'][tipo] = self.estatisticas['excluidos_por_tipo'].get(tipo, 0) + 1
        
        if resultado.dias_ferias > 0:
            self.estatisticas['com_ferias'] += 1
    
    def _log_resumo_processamento(self) -> None:
        """Registra resumo do processamento de regras"""
        stats = self.estatisticas
        
        self.logger.info("📊 RESUMO DO PROCESSAMENTO DE REGRAS:")
        self.logger.info(f"   Total processados: {stats['total_processados']}")
        self.logger.info(f"   ✅ Elegíveis: {stats['elegiveis']}")
        self.logger.info(f"   📊 Proporcionais: {stats['proporcionais']}")
        self.logger.info(f"   🏖️ Com férias: {stats['com_ferias']}")
        
        if stats['excluidos_por_tipo']:
            self.logger.info("   🚫 Exclusões por tipo:")
            for tipo, count in stats['excluidos_por_tipo'].items():
                self.logger.info(f"      {tipo}: {count}")
    
    def gerar_relatorio_elegibilidade(self, resultados: List[ResultadoElegibilidade]) -> pd.DataFrame:
        """
        Gera relatório detalhado de elegibilidade.
        
        Args:
            resultados: Lista de resultados de elegibilidade
            
        Returns:
            DataFrame com relatório detalhado
        """
        dados_relatorio = []
        
        for resultado in resultados:
            dados_relatorio.append({
                'Matrícula': resultado.matricula,
                'Status': resultado.status.value,
                'Tipo Exclusão': resultado.tipo_exclusao.value if resultado.tipo_exclusao else '',
                'Dias Elegível': resultado.dias_elegivel,
                'Dias Úteis Sindicato': resultado.dias_uteis_sindicato,
                'Dias Férias': resultado.dias_ferias,
                'Valor Diário': resultado.valor_diario,
                'Estado/Sindicato': resultado.estado_sindicato,
                'Período Início': resultado.data_inicio_periodo.strftime('%Y-%m-%d'),
                'Período Fim': resultado.data_fim_periodo.strftime('%Y-%m-%d'),
                'Observações': ' | '.join(resultado.observacoes)
            })
        
        return pd.DataFrame(dados_relatorio)
    
    def filtrar_elegiveis(self, resultados: List[ResultadoElegibilidade]) -> List[ResultadoElegibilidade]:
        """
        Filtra apenas colaboradores elegíveis ou proporcionais.
        
        Args:
            resultados: Lista completa de resultados
            
        Returns:
            Lista apenas com elegíveis e proporcionais
        """
        return [
            r for r in resultados 
            if r.status in [StatusElegibilidade.ELEGIVEL, StatusElegibilidade.PROPORCIONAL]
            and r.dias_elegivel > 0
        ]


# Função utilitária para uso direto
def processar_regras_negocio_vr(bases_validadas: Dict[str, Any], data_competencia: str = "2025-05") -> List[ResultadoElegibilidade]:
    """
    Função utilitária para processar regras de negócio de VR.
    
    Args:
        bases_validadas: Bases validadas pelo data_validator
        data_competencia: Competência no formato YYYY-MM
        
    Returns:
        Lista com resultados de elegibilidade
    """
    engine = BusinessRulesEngine(data_competencia)
    return engine.processar_elegibilidade_completa(bases_validadas)


# Exemplo de uso
if __name__ == "__main__":
    import logging
    from data_reader import carregar_bases_vr
    from data_validator import validar_bases_vr
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Testar regras de negócio
    try:
        # Carregar e validar dados
        bases = carregar_bases_vr("data/input")
        bases_validadas, _ = validar_bases_vr(bases)
        
        # Aplicar regras de negócio
        resultados = processar_regras_negocio_vr(bases_validadas)
        
        print("✅ Regras de negócio aplicadas!")
        print(f"📊 Total de resultados: {len(resultados)}")
        
        # Filtrar elegíveis
        engine = BusinessRulesEngine()
        elegiveis = engine.filtrar_elegiveis(resultados)
        excluidos = [r for r in resultados if r.status == StatusElegibilidade.EXCLUIDO]
        
        print(f"✅ Elegíveis para VR: {len(elegiveis)}")
        print(f"🚫 Excluídos: {len(excluidos)}")
        
        # Mostrar estatísticas de exclusão
        if excluidos:
            tipos_exclusao = {}
            for resultado in excluidos:
                if resultado.tipo_exclusao:
                    tipo = resultado.tipo_exclusao.value
                    tipos_exclusao[tipo] = tipos_exclusao.get(tipo, 0) + 1
            
            print("\n📋 Exclusões por tipo:")
            for tipo, count in sorted(tipos_exclusao.items()):
                print(f"   {tipo}: {count}")
        
        # Mostrar alguns exemplos
        print("\n📝 Exemplos de colaboradores elegíveis:")
        for resultado in elegiveis[:3]:
            print(f"   Matrícula {resultado.matricula}: {resultado.dias_elegivel} dias, R$ {resultado.valor_diario:.2f}/dia")
        
    except Exception as e:
        print(f"❌ Erro: {e}")